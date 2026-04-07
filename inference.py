#!/usr/bin/env python3
"""
inference.py - T1D Environment Inference Script
Meta PyTorch Hackathon - Round 1 Submission

This script runs an RL agent on the Type 1 Diabetes environment
and outputs results in the required format for automated grading.
"""

import os
import sys
import argparse

# ============================================================================
# ENVIRONMENT VARIABLES (Required by Hackathon)
# ============================================================================

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")  # Required only for --use-llm

# ============================================================================
# IMPORT ENVIRONMENT
# ============================================================================

from t1d_env import T1DEnv, TaskDifficulty, Action, State
from baseline_agents import (
    baseline_basal_only,
    baseline_fixed_bolus,
    baseline_proportional_controller,
    baseline_pid_controller,
    baseline_model_predictive,
    baseline_adaptive
)

# ============================================================================
# LLM-POWERED AGENT (Optional - only used with --use-llm flag)
# ============================================================================

def llm_agent(state: State, task_name: str) -> Action:
    """
    LLM-powered agent that uses GPT to make insulin dosing decisions
    
    This demonstrates using the OpenAI client as required by hackathon.
    Only called when --use-llm flag is set.
    """
    # Lazy import to avoid requiring openai when not needed
    from openai import OpenAI
    
    # Initialize client
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    
    # Create prompt for LLM
    prompt = f"""You are managing blood glucose for a Type 1 diabetes patient.

Current State:
- Glucose: {state.glucose:.1f} mg/dL (target: 70-180)
- Active Insulin (IOB): {state.active_insulin:.1f} units
- Active Carbs (COB): {state.active_carbs:.1f} grams
- Time: {state.time_of_day:.1f} hours
- Upcoming Meal: {state.meal_announced:.1f} grams carbs
- Exercise Level: {state.exercise_level:.2f}
- Recent Glucose: {[f'{g:.0f}' for g in state.glucose_history[-3:]]}

Task: {task_name}

Decide insulin bolus dose (0-20 units). Respond with ONLY a number.

Safety Rules:
- If glucose < 120: Give 0 (avoid low glucose)
- If meal announced: Give meal_carbs/10 (carb ratio)
- If glucose > 160 and no meal: Give (glucose-100)/50 - IOB*0.5 (correction)
- Never dose if trending down or high IOB

Your decision (just the number, 0-20 units):"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=10
        )
        
        # Parse response
        content = response.choices[0].message.content.strip()
        # Extract first number found
        import re
        numbers = re.findall(r'\d+\.?\d*', content)
        if numbers:
            insulin_dose = float(numbers[0])
            insulin_dose = max(0.0, min(20.0, insulin_dose))
        else:
            # Fallback to safe default
            insulin_dose = 0.0
        
        return Action(insulin_bolus=insulin_dose)
    
    except Exception as e:
        # Fallback to baseline agent on LLM failure
        print(f"# LLM call failed: {e}, using baseline agent", file=sys.stderr)
        return baseline_adaptive(state)


# ============================================================================
# MAIN INFERENCE RUNNER
# ============================================================================

def run_inference(task_name: str = "easy", use_llm: bool = False, agent_name: str = "adaptive"):
    """
    Run inference on T1D environment with required output format
    
    Args:
        task_name: "easy", "medium", or "hard"
        use_llm: If True, use LLM agent; if False, use baseline agent
        agent_name: Which baseline agent to use (if not using LLM)
    """
    # Check HF_TOKEN only if using LLM
    if use_llm:
        if HF_TOKEN is None:
            print("ERROR: HF_TOKEN environment variable is required when using --use-llm", file=sys.stderr)
            print("Set it with: export HF_TOKEN='your-token'", file=sys.stderr)
            sys.exit(1)
    
    # Map task name to difficulty
    task_map = {
        "easy": TaskDifficulty.EASY,
        "medium": TaskDifficulty.MEDIUM,
        "hard": TaskDifficulty.HARD,
    }
    
    task_difficulty = task_map.get(task_name, TaskDifficulty.EASY)
    
    # Create environment
    env = T1DEnv(task=task_difficulty)
    state = env.reset()
    
    # Choose agent
    if use_llm:
        agent_fn = lambda s: llm_agent(s, task_name)
        model_display = MODEL_NAME
    else:
        # Map agent name to function
        agent_map = {
            'basal': baseline_basal_only,
            'fixed': baseline_fixed_bolus,
            'proportional': baseline_proportional_controller,
            'pid': baseline_pid_controller,
            'mpc': baseline_model_predictive,
            'adaptive': baseline_adaptive,
        }
        agent_fn = agent_map.get(agent_name, baseline_adaptive)
        model_display = f"baseline_{agent_name}"
    
    # ========================================================================
    # [START] - Required format
    # ========================================================================
    print(f"[START] task={task_name} env=t1d-diabetes model={model_display}")
    
    # Run episode
    rewards = []
    steps = 0
    success = True
    
    try:
        for step in range(env.episode_length):
            # Agent decides action
            action = agent_fn(state)
            
            # Format action as string
            action_str = f"insulin_bolus({action.insulin_bolus:.2f})"
            
            # Step environment
            try:
                state, reward, done, info = env.step(action)
                error_msg = "null"
                done_str = "true" if done else "false"
            except Exception as e:
                error_msg = str(e).replace("\n", " ")
                done_str = "true"
                success = False
                reward = 0.0
            
            # ====================================================================
            # [STEP] - Required format
            # ====================================================================
            print(f"[STEP] step={step+1} action={action_str} reward={reward:.2f} done={done_str} error={error_msg}")
            
            rewards.append(reward)
            steps = step + 1
            
            if done:
                break
    
    except Exception as e:
        success = False
        print(f"# Episode failed: {e}", file=sys.stderr)
    
    # ========================================================================
    # [END] - Required format
    # ========================================================================
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    success_str = "true" if success else "false"
    print(f"[END] success={success_str} steps={steps} rewards={rewards_str}")
    
    # Additional info (not part of required format, but useful for debugging)
    if success:
        final_score = env.get_score()
        tir = env._calculate_time_in_range()
        print(f"\n# Final Score: {final_score:.3f}", file=sys.stderr)
        print(f"# Time in Range: {tir:.1%}", file=sys.stderr)
        print(f"# Severe Hypo Events: {env.severe_hypo_events}", file=sys.stderr)


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run T1D environment inference for Meta PyTorch Hackathon"
    )
    parser.add_argument(
        "--task", 
        type=str, 
        default="easy", 
        choices=["easy", "medium", "hard"],
        help="Task difficulty (default: easy)"
    )
    parser.add_argument(
        "--use-llm", 
        action="store_true",
        help="Use LLM agent (requires HF_TOKEN environment variable)"
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="adaptive",
        choices=["basal", "fixed", "proportional", "pid", "mpc", "adaptive"],
        help="Baseline agent to use (default: adaptive)"
    )
    
    args = parser.parse_args()
    
    # Run inference
    run_inference(task_name=args.task, use_llm=args.use_llm, agent_name=args.agent)
