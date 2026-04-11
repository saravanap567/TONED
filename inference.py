"""
OpenEnv Inference Agent for T1D Environment
Meta PyTorch Hackathon - Round 1

"""

import os
import sys
import json
import time
import asyncio
import logging
from typing import Dict, Any, List, Optional

import httpx
from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# 3. Required Environment Variables (per submission guidelines)
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

# ---------------------------------------------------------------------------
# Timeout & performance config
# ---------------------------------------------------------------------------

ENV_SERVER_URL = os.environ.get("ENV_SERVER_URL", "http://localhost:7860")
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "5"))   # A. env /step timeout
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "8"))          # A. hard cap per LLM call
TOTAL_TIMEOUT = float(os.environ.get("TOTAL_TIMEOUT", "1080"))    # 18 min — 2 min buffer (limit is 20 min)
LLM_CALL_INTERVAL = int(os.environ.get("LLM_CALL_INTERVAL", "10"))  # B. LLM every N steps
STEP_TIME_BUDGET = float(os.environ.get("STEP_TIME_BUDGET", "3.0"))  # D. per-step budget
TASKS = os.environ.get("TASKS", "easy,medium,hard").split(",")

# Logging to stderr — stdout reserved for [START]/[STEP]/[END]
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("inference")

# ---------------------------------------------------------------------------
# 2. OpenAI Async Client (C. non-blocking I/O)
# ---------------------------------------------------------------------------

llm_client = AsyncOpenAI(
    api_key=HF_TOKEN or os.getenv("OPENAI_API_KEY") or "no-key",
    base_url=API_BASE_URL,
    timeout=LLM_TIMEOUT,
    max_retries=0,  # Fail fast — algorithmic fallback is instant
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


# ---------------------------------------------------------------------------
# A. LLM call with hard timeout via asyncio.wait_for
# ---------------------------------------------------------------------------

async def call_llm(observation: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """Call LLM via OpenAI async client. Returns None on any failure/timeout."""
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

        # Hard timeout guarantee — even if SDK timeout fails, this will cancel
        response = await asyncio.wait_for(
            llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=50,
            ),
            timeout=LLM_TIMEOUT,
        )

        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        action = json.loads(content)
        bolus = float(action.get("insulin_bolus", 0.0))
        bolus = max(0.0, min(bolus, 20.0))
        return {"insulin_bolus": round(bolus, 3)}

    except asyncio.TimeoutError:
        log.warning("LLM hard timeout (>%ss)", LLM_TIMEOUT)
        return None
    except Exception as exc:
        log.warning("LLM call failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Algorithmic fallback agent (instant, no network)
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
# B. Batched LLM calls + circuit breaker
#    - LLM called every LLM_CALL_INTERVAL steps (not every step)
#    - Always called on step 1 and meal events (critical decisions)
#    - Circuit breaker trips after 3 consecutive failures
#    - Skipped if previous step exceeded time budget (D.)
# ---------------------------------------------------------------------------

_llm_consecutive_failures = 0
_LLM_MAX_FAILURES = 3
_last_llm_action: Optional[Dict[str, float]] = None


async def decide_action(
    observation: Dict[str, Any],
    step_num: int,
    time_budget_ok: bool,
) -> Dict[str, float]:
    """Pick action. Calls LLM only on key steps; algorithmic otherwise."""
    global _llm_consecutive_failures, _last_llm_action

    # Circuit breaker tripped — algorithmic only
    if _llm_consecutive_failures >= _LLM_MAX_FAILURES:
        return algorithmic_agent(observation)

    # B. Decide whether this step warrants an LLM call
    meal_announced = observation.get("meal_announced", 0.0)
    is_llm_step = (
        step_num == 1                           # First step of episode
        or step_num % LLM_CALL_INTERVAL == 0    # Every N steps
        or meal_announced > 0                    # Meal event (critical)
    )

    # D. Skip LLM if last step was slow
    if not time_budget_ok:
        is_llm_step = False

    if not is_llm_step:
        return _last_llm_action if _last_llm_action is not None else algorithmic_agent(observation)

    # A. Call LLM with hard timeout
    action = await call_llm(observation)
    if action is not None:
        _llm_consecutive_failures = 0
        _last_llm_action = action
        return action

    # LLM failed — increment circuit breaker
    _llm_consecutive_failures += 1
    if _llm_consecutive_failures >= _LLM_MAX_FAILURES:
        log.info("Circuit breaker tripped after %d LLM failures — algorithmic only", _LLM_MAX_FAILURES)
    return algorithmic_agent(observation)


# ---------------------------------------------------------------------------
# C. Async environment client
# ---------------------------------------------------------------------------

async def wait_for_server(http: httpx.AsyncClient, deadline: float):
    """Non-blocking wait for server readiness."""
    log.info("Waiting for environment server at %s ...", ENV_SERVER_URL)
    while time.monotonic() < deadline:
        try:
            resp = await http.get(f"{ENV_SERVER_URL}/", timeout=3.0)
            if resp.status_code == 200:
                log.info("Server is ready.")
                return
        except (httpx.ConnectError, httpx.TimeoutException):
            pass
        await asyncio.sleep(0.5)
    raise RuntimeError("Environment server did not become ready in time.")


async def run_episode(
    http: httpx.AsyncClient,
    task: str,
    deadline: float,
):
    """Run one episode. Emits [START], [STEP]..., [END] to stdout."""
    global _last_llm_action
    _last_llm_action = None  # Reset cached action per episode

    rewards: List[float] = []
    steps = 0
    score = 0.0
    success = False
    done = False
    step_info: Dict[str, Any] = {}
    time_budget_ok = True

    # --- 4. [START] line ---
    print(
        f"[START] task={task} env=t1d-insulin-management model={MODEL_NAME}",
        flush=True,
    )

    try:
        # Reset environment
        resp = await http.post(
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
                log.warning("Total timeout at step %d", step_num)
                break

            step_start = time.monotonic()

            # Decide action (async, batched, with circuit breaker)
            action = await decide_action(observation, step_num, time_budget_ok)
            action_str = f"insulin_bolus({action['insulin_bolus']})"

            # D. Step environment + measure latency
            env_start = time.monotonic()
            resp = await http.post(
                f"{ENV_SERVER_URL}/step",
                json={"action": action},
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            step_data = resp.json()
            env_elapsed = time.monotonic() - env_start

            if env_elapsed > 1.0:
                log.warning("Slow /step: %.2fs at step %d (task=%s)", env_elapsed, step_num, task)

            observation = step_data["observation"]
            reward = float(step_data.get("reward", 0.0))
            done = bool(step_data.get("done", False))
            step_info = step_data.get("info", {})

            rewards.append(reward)
            steps += 1

            # --- 4. [STEP] line ---
            print(
                f"[STEP] step={step_num}"
                f" action={action_str}"
                f" reward={reward:.2f}"
                f" done={'true' if done else 'false'}"
                f" error=null",
                flush=True,
            )

            # D. Track step timing for next iteration
            step_elapsed = time.monotonic() - step_start
            time_budget_ok = step_elapsed < STEP_TIME_BUDGET

            if done:
                break

        # Compute normalized score in [0, 1] from time_in_range
        tir = step_info.get("time_in_range", 0) if step_info else 0
        score = min(max(float(tir), 0.0), 1.0)
        success = score >= 0.60

    except Exception as exc:
        log.error("Episode error for task=%s: %s", task, exc)

    # --- 4. [END] line (always emitted, even on exception) ---
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else ""
    print(
        f"[END] success={'true' if success else 'false'}"
        f" steps={steps}"
        f" score={score:.3f}"
        f" rewards={rewards_str}",
        flush=True,
    )

    return {
        "task": task,
        "steps": steps,
        "score": score,
        "success": success,
        "total_reward": sum(rewards),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def main():
    start = time.monotonic()
    deadline = start + TOTAL_TIMEOUT

    log.info("Inference starting")
    log.info("  model=%s  base_url=%s", MODEL_NAME, API_BASE_URL)
    log.info("  LLM_TIMEOUT=%ss  LLM_CALL_INTERVAL=%d  TOTAL_TIMEOUT=%ss",
             LLM_TIMEOUT, LLM_CALL_INTERVAL, TOTAL_TIMEOUT)

    async with httpx.AsyncClient() as http:
        await wait_for_server(http, deadline)

        results = []
        for task in TASKS:
            task = task.strip()
            if not task:
                continue
            if time.monotonic() > deadline - 60:
                log.warning("Skipping task %s — not enough time.", task)
                break
            try:
                result = await run_episode(http, task, deadline)
                results.append(result)
            except Exception as exc:
                log.error("Fatal error in task=%s: %s", task, exc)

    elapsed = time.monotonic() - start
    log.info("Inference complete in %.1f seconds", elapsed)
    for r in results:
        log.info(
            "  Task=%-8s Steps=%-4d Score=%.3f Success=%-5s Reward=%.2f",
            r["task"], r["steps"], r["score"], str(r["success"]), r["total_reward"],
        )


if __name__ == "__main__":
    asyncio.run(main())
