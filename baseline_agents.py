"""
Baseline Agents for T1D Environment
Multiple strategies for insulin dosing decisions
"""

from t1d_env import State, Action
import numpy as np


def baseline_basal_only(state: State) -> Action:
    """
    Simplest baseline: No bolus insulin, only basal
    
    Strategy:
    - Never gives bolus insulin
    - Relies entirely on basal insulin
    
    Expected Performance:
    - Easy: ~0.70 (can maintain fasting glucose)
    - Medium: ~0.20 (fails with meals)
    - Hard: ~0.15 (completely inadequate)
    """
    return Action(insulin_bolus=0.0)


def baseline_fixed_bolus(state: State) -> Action:
    """
    Fixed carb ratio for meal boluses
    
    Strategy:
    - Uses fixed 1:10 carb ratio
    - No correction for high glucose
    - No IOB consideration
    
    Expected Performance:
    - Easy: ~0.70 (same as basal only)
    - Medium: ~0.55 (helps with meal but inflexible)
    - Hard: ~0.45 (multiple meals but no adaptation)
    """
    if state.meal_announced > 0:
        bolus = state.meal_announced / 10.0  # 1:10 carb ratio
        return Action(insulin_bolus=bolus)
    return Action(insulin_bolus=0.0)


def baseline_proportional_controller(state: State) -> Action:
    """
    Proportional controller with meal handling
    
    Strategy:
    - Meal bolus: 1:10 carb ratio
    - Correction bolus: proportional to glucose error
    - IOB adjustment: reduce dose if insulin on board
    
    Expected Performance:
    - Easy: ~0.85 (good fasting control)
    - Medium: ~0.65 (handles single meal well)
    - Hard: ~0.60 (reasonable full day management)
    """
    target_glucose = 100.0
    correction_factor = 50.0  # mg/dL per unit
    carb_ratio = 10.0  # grams per unit
    
    bolus = 0.0
    
    # Meal bolus
    if state.meal_announced > 0:
        meal_bolus = state.meal_announced / carb_ratio
        bolus += meal_bolus
    
    # Correction bolus if glucose is high
    glucose_error = state.glucose - target_glucose
    if glucose_error > 30:  # Only correct if significantly high
        correction_bolus = glucose_error / correction_factor
        # Reduce by insulin on board (avoid stacking)
        correction_bolus = max(0, correction_bolus - state.active_insulin * 0.5)
        bolus += correction_bolus
    
    return Action(insulin_bolus=bolus)


def baseline_pid_controller(state: State) -> Action:
    """
    PID controller (Proportional-Integral-Derivative)
    
    Strategy:
    - P: Proportional to current error
    - I: Integral of past errors (from glucose history)
    - D: Derivative of error (glucose trend)
    - Plus meal handling
    
    Expected Performance:
    - Easy: ~0.88 (excellent fasting)
    - Medium: ~0.70 (good meal management)
    - Hard: ~0.65 (solid full day)
    """
    # PID gains (tuned empirically)
    Kp = 0.02  # Proportional gain
    Ki = 0.001  # Integral gain
    Kd = 0.05  # Derivative gain
    
    target_glucose = 100.0
    carb_ratio = 10.0
    
    bolus = 0.0
    
    # Meal bolus
    if state.meal_announced > 0:
        meal_bolus = state.meal_announced / carb_ratio
        bolus += meal_bolus
    
    # PID correction
    # P: Current error
    error = state.glucose - target_glucose
    p_term = Kp * error
    
    # I: Accumulated error from history
    if len(state.glucose_history) >= 12:
        i_term = Ki * sum(g - target_glucose for g in state.glucose_history)
    else:
        i_term = 0.0
    
    # D: Rate of change (derivative)
    if len(state.glucose_history) >= 2:
        glucose_trend = state.glucose - state.glucose_history[-2]
        d_term = Kd * glucose_trend
    else:
        d_term = 0.0
    
    # Total correction
    correction = p_term + i_term + d_term
    
    # Only give positive correction, account for IOB
    correction = max(0, correction - state.active_insulin * 0.3)
    bolus += correction
    
    return Action(insulin_bolus=bolus)


def baseline_model_predictive(state: State) -> Action:
    """
    Simplified Model Predictive Control (MPC)
    
    Strategy:
    - Predict future glucose based on current state
    - Optimize insulin dose to reach target
    - Account for meals and IOB
    
    Expected Performance:
    - Easy: ~0.90 (excellent)
    - Medium: ~0.75 (very good)
    - Hard: ~0.70 (strong performance)
    """
    target_glucose = 100.0
    prediction_horizon = 6  # 30 minutes ahead
    sensitivity = 50.0  # mg/dL per unit
    carb_effect = 4.0  # mg/dL per gram
    carb_ratio = 10.0
    
    bolus = 0.0
    
    # Meal bolus
    if state.meal_announced > 0:
        meal_bolus = state.meal_announced / carb_ratio
        bolus += meal_bolus
    
    # Predict future glucose
    # Simple linear prediction based on current trajectory
    if len(state.glucose_history) >= 3:
        # Calculate trend from last 3 readings (15 minutes)
        trend = (state.glucose - state.glucose_history[-3]) / 3
        predicted_glucose = state.glucose + trend * prediction_horizon
    else:
        predicted_glucose = state.glucose
    
    # Account for IOB effect
    iob_effect = -state.active_insulin * sensitivity * 0.25  # 25% of total effect in next 30min
    predicted_glucose += iob_effect
    
    # Account for COB effect
    cob_effect = state.active_carbs * carb_effect * 0.3  # 30% absorbed in next 30min
    predicted_glucose += cob_effect
    
    # Calculate correction needed
    predicted_error = predicted_glucose - target_glucose
    
    if predicted_error > 20:
        correction = predicted_error / sensitivity
        # Don't stack insulin too much
        correction = max(0, correction - state.active_insulin * 0.4)
        bolus += correction
    
    return Action(insulin_bolus=bolus)


def baseline_aggressive(state: State) -> Action:
    """
    Aggressive controller - prioritizes tight control over safety
    
    WARNING: Higher risk of hypoglycemia
    
    Strategy:
    - Target 90 mg/dL (tighter than standard)
    - More aggressive corrections
    - Less conservative IOB adjustment
    
    Expected Performance:
    - Easy: ~0.75 (risk of lows)
    - Medium: ~0.65 (tight control but risky)
    - Hard: ~0.55 (too many hypo events)
    """
    target_glucose = 90.0  # Tighter target
    correction_factor = 40.0  # More aggressive (less mg/dL per unit)
    carb_ratio = 8.0  # More aggressive carb ratio
    
    bolus = 0.0
    
    # Meal bolus
    if state.meal_announced > 0:
        meal_bolus = state.meal_announced / carb_ratio
        bolus += meal_bolus
    
    # Aggressive correction
    glucose_error = state.glucose - target_glucose
    if glucose_error > 10:  # Correct even small deviations
        correction_bolus = glucose_error / correction_factor
        # Less conservative IOB adjustment
        correction_bolus = max(0, correction_bolus - state.active_insulin * 0.2)
        bolus += correction_bolus
    
    return Action(insulin_bolus=bolus)


def baseline_conservative(state: State) -> Action:
    """
    Conservative controller - prioritizes safety over tight control
    
    Strategy:
    - Higher target glucose (less risk of lows)
    - Less aggressive corrections
    - More conservative IOB adjustment
    
    Expected Performance:
    - Easy: ~0.80 (safe but higher glucose)
    - Medium: ~0.60 (safe meal handling)
    - Hard: ~0.55 (runs higher but very safe)
    """
    target_glucose = 120.0  # Higher, safer target
    correction_factor = 60.0  # Less aggressive
    carb_ratio = 12.0  # More conservative
    
    bolus = 0.0
    
    # Meal bolus
    if state.meal_announced > 0:
        meal_bolus = state.meal_announced / carb_ratio
        bolus += meal_bolus
    
    # Conservative correction
    glucose_error = state.glucose - target_glucose
    if glucose_error > 50:  # Only correct significant highs
        correction_bolus = glucose_error / correction_factor
        # Very conservative IOB adjustment
        correction_bolus = max(0, correction_bolus - state.active_insulin * 0.7)
        bolus += correction_bolus
    
    # Additional safety: don't dose if trending down
    if len(state.glucose_history) >= 2:
        if state.glucose < state.glucose_history[-2]:
            bolus *= 0.5  # Reduce dose if glucose already falling
    
    return Action(insulin_bolus=bolus)


def baseline_adaptive(state: State) -> Action:
    """
    Adaptive controller - adjusts strategy based on time of day
    
    Strategy:
    - Morning (4-10am): More aggressive (dawn phenomenon)
    - Afternoon (10am-6pm): Moderate
    - Evening (6pm-midnight): Conservative
    - Night (midnight-4am): Very conservative
    
    Expected Performance:
    - Easy: ~0.85 (adapts to circadian patterns)
    - Medium: ~0.68 (time-aware)
    - Hard: ~0.72 (best overall - adapts to full day)
    """
    hour = state.time_of_day
    carb_ratio = 10.0
    
    # Time-of-day dependent parameters
    if 4 <= hour < 10:  # Morning - dawn phenomenon
        target = 95.0
        correction_factor = 45.0
        iob_factor = 0.3
    elif 10 <= hour < 18:  # Afternoon - active period
        target = 100.0
        correction_factor = 50.0
        iob_factor = 0.4
    elif 18 <= hour < 24:  # Evening - conservative
        target = 110.0
        correction_factor = 55.0
        iob_factor = 0.6
    else:  # Night - very conservative
        target = 115.0
        correction_factor = 60.0
        iob_factor = 0.7
    
    bolus = 0.0
    
    # Meal bolus
    if state.meal_announced > 0:
        meal_bolus = state.meal_announced / carb_ratio
        bolus += meal_bolus
    
    # Time-adaptive correction
    glucose_error = state.glucose - target
    if glucose_error > 30:
        correction = glucose_error / correction_factor
        correction = max(0, correction - state.active_insulin * iob_factor)
        bolus += correction
    
    # Exercise adjustment
    if state.exercise_level > 0.3:
        bolus *= (1 - state.exercise_level * 0.3)  # Reduce during exercise
    
    return Action(insulin_bolus=bolus)


# Dictionary for easy access
BASELINE_AGENTS = {
    'basal_only': baseline_basal_only,
    'fixed_bolus': baseline_fixed_bolus,
    'proportional': baseline_proportional_controller,
    'pid': baseline_pid_controller,
    'mpc': baseline_model_predictive,
    'aggressive': baseline_aggressive,
    'conservative': baseline_conservative,
    'adaptive': baseline_adaptive,
}


if __name__ == "__main__":
    """Test all baseline agents"""
    from t1d_env import T1DEnv, TaskDifficulty
    from grading import T1DGrader
    import pandas as pd
    
    print("="*80)
    print("BASELINE AGENTS COMPARISON")
    print("="*80)
    
    grader = T1DGrader()
    results_data = []
    
    for agent_name, agent_fn in BASELINE_AGENTS.items():
        print(f"\nTesting: {agent_name}")
        print("-"*80)
        
        for task in [TaskDifficulty.EASY, TaskDifficulty.MEDIUM, TaskDifficulty.HARD]:
            # Run 3 episodes
            scores = []
            pass_count = 0
            
            for _ in range(3):
                env = T1DEnv(task=task)
                result = grader.grade_episode(env, agent_fn, verbose=False)
                scores.append(result.score)
                if result.passed:
                    pass_count += 1
            
            mean_score = np.mean(scores)
            pass_rate = pass_count / 3
            
            results_data.append({
                'Agent': agent_name,
                'Task': task.value,
                'Mean Score': f"{mean_score:.3f}",
                'Pass Rate': f"{pass_rate:.1%}"
            })
            
            print(f"  {task.value:15s}: Score={mean_score:.3f}, Pass={pass_rate:.1%}")
    
    # Summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    df = pd.DataFrame(results_data)
    print(df.to_string(index=False))
    
    # Find best agent per task
    print("\n" + "="*80)
    print("BEST AGENTS PER TASK")
    print("="*80)
    for task in ['fasting_control', 'single_meal', 'full_day']:
        task_results = [r for r in results_data if r['Task'] == task]
        best = max(task_results, key=lambda x: float(x['Mean Score']))
        print(f"{task:15s}: {best['Agent']:15s} (Score: {best['Mean Score']})")
