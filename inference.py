"""
OpenEnv Inference Agent for T1D Environment
Meta PyTorch Hackathon - Round 1

Compliant with OpenEnv Hackathon Submission Guidelines:
- Uses OpenAI Client for LLM calls
- Reads API_BASE_URL, MODEL_NAME, HF_TOKEN env vars
- Emits [START], [STEP], [END] to stdout
"""

import os
import sys
import json
import time
import logging
from typing import Dict, Any, List, Optional

import httpx
from openai import OpenAI

# ---------------------------------------------------------------------------
# Required Environment Variables (per hackathon guidelines)
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Server / timeout config
ENV_SERVER_URL = os.environ.get("ENV_SERVER_URL", "http://localhost:7860")
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "10"))
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "15"))
TOTAL_TIMEOUT = float(os.environ.get("TOTAL_TIMEOUT", "1500"))
TASKS = os.environ.get("TASKS", "easy,medium,hard").split(",")

# Logging to stderr so stdout stays clean for [START]/[STEP]/[END]
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("inference")

# ---------------------------------------------------------------------------
# OpenAI Client (per guideline #2: must use OpenAI Client)
# ---------------------------------------------------------------------------

client = OpenAI(
    api_key=HF_TOKEN or "no-key",
    base_url=API_BASE_URL,
    timeout=LLM_TIMEOUT,
    max_retries=1,
)

SYSTEM_PROMPT = """You are an insulin dosing controller for a Type 1 Diabetes patient.

Given the patient's current state, decide the insulin bolus dose (0-20 units).

RULES:
- If glucose < 80 mg/dL: give 0 insulin (patient is already low)
- If glucose is dropping fast (trend < -10): give 0 or very little insulin
- If a meal is announced: give meal_carbs / 10.0 units as a meal bolus
- If glucose > 130 mg/dL: give a small correction = (glucose - 100) / 50.0
- Subtract active_insulin * 0.4 from any correction to avoid insulin stacking
- During exercise (exercise_level > 0.3): reduce dose by 50%

Respond with ONLY a JSON object: {"insulin_bolus": <number>}
No explanation, no extra text."""


def call_llm(observation: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """Call LLM via OpenAI Client to decide action. Returns None on failure."""
    try:
        user_msg = json.dumps({
            "glucose": round(observation.get("glucose", 120), 1),
            "active_insulin": round(observation.get("active_insulin", 0), 2),
            "active_carbs": round(observation.get("active_carbs", 0), 1),
            "time_of_day": round(observation.get("time_of_day", 0), 2),
            "meal_announced": round(observation.get("meal_announced", 0), 1),
            "exercise_level": round(observation.get("exercise_level", 0), 2),
            "glucose_trend": round(
                observation.get("glucose", 120)
                - (observation.get("glucose_history", [120])[-3]
                   if len(observation.get("glucose_history", [])) >= 3
                   else observation.get("glucose", 120)),
                1,
            ),
        })

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=50,
        )

        content = response.choices[0].message.content.strip()
        # Parse JSON from response (handle markdown fences)
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        action = json.loads(content)
        bolus = float(action.get("insulin_bolus", 0.0))
        bolus = max(0.0, min(bolus, 20.0))
        return {"insulin_bolus": round(bolus, 3)}

    except Exception as exc:
        log.warning("LLM call failed (%s), falling back to algorithmic agent", exc)
        return None


# ---------------------------------------------------------------------------
# Fast Algorithmic Fallback Agent (used when LLM fails or times out)
# ---------------------------------------------------------------------------

def algorithmic_agent(obs: Dict[str, Any]) -> Dict[str, float]:
    """Fast adaptive insulin controller — runs in microseconds."""
    glucose = obs.get("glucose", 120.0)
    active_insulin = obs.get("active_insulin", 0.0)
    meal_announced = obs.get("meal_announced", 0.0)
    exercise_level = obs.get("exercise_level", 0.0)
    time_of_day = obs.get("time_of_day", 0.0)
    glucose_history = obs.get("glucose_history", [])

    carb_ratio = 10.0

    if 4 <= time_of_day < 10:
        target, correction_factor, iob_factor = 95.0, 45.0, 0.3
    elif 10 <= time_of_day < 18:
        target, correction_factor, iob_factor = 100.0, 50.0, 0.4
    elif 18 <= time_of_day < 24:
        target, correction_factor, iob_factor = 110.0, 55.0, 0.6
    else:
        target, correction_factor, iob_factor = 115.0, 60.0, 0.7

    if glucose < 80:
        return {"insulin_bolus": 0.0}

    trend = 0.0
    if len(glucose_history) >= 3:
        trend = glucose - glucose_history[-3]

    if trend < -15 and glucose < 130:
        return {"insulin_bolus": 0.0}

    bolus = 0.0

    if meal_announced > 0:
        bolus += meal_announced / carb_ratio

    glucose_error = glucose - target
    if glucose_error > 30:
        correction = glucose_error / correction_factor
        correction = max(0.0, correction - active_insulin * iob_factor)
        bolus += correction

    if exercise_level > 0.3:
        bolus *= (1 - exercise_level * 0.5)

    if trend < -10:
        bolus *= 0.3
    elif trend < -5:
        bolus *= 0.6

    bolus = max(0.0, min(bolus, 15.0))
    return {"insulin_bolus": round(bolus, 3)}


# ---------------------------------------------------------------------------
# Decide action: try LLM first, fall back to algorithmic
# Circuit breaker: stop calling LLM after consecutive failures
# ---------------------------------------------------------------------------

_llm_consecutive_failures = 0
_LLM_MAX_FAILURES = 3  # After 3 consecutive failures, use algorithmic only


def decide_action(observation: Dict[str, Any]) -> Dict[str, float]:
    """Get action from LLM, fall back to algorithmic agent on failure."""
    global _llm_consecutive_failures

    if _llm_consecutive_failures >= _LLM_MAX_FAILURES:
        return algorithmic_agent(observation)

    action = call_llm(observation)
    if action is not None:
        _llm_consecutive_failures = 0
        return action

    _llm_consecutive_failures += 1
    if _llm_consecutive_failures >= _LLM_MAX_FAILURES:
        log.info("LLM circuit breaker tripped — switching to algorithmic agent")
    return algorithmic_agent(observation)


# ---------------------------------------------------------------------------
# Environment Client (sync httpx — simple, reliable)
# ---------------------------------------------------------------------------

def wait_for_server(http: httpx.Client, deadline: float):
    """Wait for the environment server to become available."""
    log.info("Waiting for environment server at %s ...", ENV_SERVER_URL)
    while time.monotonic() < deadline:
        try:
            resp = http.get(f"{ENV_SERVER_URL}/", timeout=3.0)
            if resp.status_code == 200:
                log.info("Server is ready.")
                return
        except (httpx.ConnectError, httpx.TimeoutException):
            pass
        time.sleep(0.5)
    raise RuntimeError("Environment server did not become ready in time.")


def run_episode(http: httpx.Client, task: str, deadline: float):
    """Run a single episode, emitting [START]/[STEP]/[END] to stdout."""
    global _llm_consecutive_failures

    rewards: List[float] = []
    steps = 0
    success = False
    last_error = None

    # --- [START] ---
    print(
        f"[START] task={task} env=t1d-insulin-management model={MODEL_NAME}",
        flush=True,
    )

    try:
        # Reset environment
        resp = http.post(
            f"{ENV_SERVER_URL}/reset",
            json={"task": task},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        observation = data["observation"]
        info = data.get("info", {})
        episode_length = info.get("episode_length", 288)

        for step_num in range(1, episode_length + 1):
            if time.monotonic() > deadline:
                last_error = "timeout"
                log.warning("Approaching total timeout at step %d", step_num)
                break

            # Decide action
            action = decide_action(observation)
            action_str = f"insulin_bolus({action['insulin_bolus']})"

            # Step environment
            resp = http.post(
                f"{ENV_SERVER_URL}/step",
                json={"action": action},
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            step_data = resp.json()

            observation = step_data["observation"]
            reward = float(step_data.get("reward", 0.0))
            done = bool(step_data.get("done", False))
            step_info = step_data.get("info", {})

            rewards.append(reward)
            steps += 1
            last_error = None

            # --- [STEP] ---
            print(
                f"[STEP] step={step_num}"
                f" action={action_str}"
                f" reward={reward:.2f}"
                f" done={'true' if done else 'false'}"
                f" error=null",
                flush=True,
            )

            if done:
                tir = step_info.get("time_in_range", 0)
                success = tir >= 0.60
                break

        if not done:
            success = False

    except Exception as exc:
        last_error = str(exc)
        log.error("Episode error for task=%s: %s", task, exc)

    # --- [END] ---
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else ""
    print(
        f"[END] success={'true' if success else 'false'}"
        f" steps={steps}"
        f" rewards={rewards_str}",
        flush=True,
    )

    return {
        "task": task,
        "steps": steps,
        "success": success,
        "total_reward": sum(rewards),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    start = time.monotonic()
    deadline = start + TOTAL_TIMEOUT

    log.info("Inference starting — model=%s base_url=%s", MODEL_NAME, API_BASE_URL)
    log.info("Tasks: %s", TASKS)

    with httpx.Client() as http:
        wait_for_server(http, deadline)

        results = []
        for task in TASKS:
            task = task.strip()
            if not task:
                continue
            if time.monotonic() > deadline - 60:
                log.warning("Skipping task %s — not enough time remaining.", task)
                break
            result = run_episode(http, task, deadline)
            results.append(result)

    elapsed = time.monotonic() - start
    log.info("Inference complete in %.1f seconds", elapsed)
    for r in results:
        log.info(
            "  Task=%-8s Steps=%-4d Success=%-5s Reward=%.2f",
            r["task"], r["steps"], str(r["success"]), r["total_reward"],
        )


if __name__ == "__main__":
    main()
