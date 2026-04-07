"""
Automated Grading System for T1D Environment
Evaluates agent performance across three difficulty levels
"""

from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass
import numpy as np
from t1d_env import T1DEnv, TaskDifficulty, Action, State


@dataclass
class GradingResult:
    """Results from grading a single episode"""
    task: TaskDifficulty
    score: float  # 0.0 to 1.0
    passed: bool
    metrics: Dict[str, float]
    feedback: str


class T1DGrader:
    """
    Automated grader for T1D environment
    
    Each task has specific success criteria:
    
    EASY (Fasting Control):
        - Target: 80% time in range (70-180 mg/dL)
        - Safety: No severe hypoglycemia (<54 mg/dL)
        - Pass threshold: 0.70 score
    
    MEDIUM (Single Meal):
        - Target: 70% time in range
        - Recovery: Return to baseline ±20 mg/dL
        - Safety: No severe hypoglycemia
        - Pass threshold: 0.60 score
    
    HARD (Full Day):
        - Target: 70% time in range
        - Safety: <5% time in hypoglycemia, no severe events
        - Variability: Glucose CV < 36%
        - Pass threshold: 0.65 score
    """
    
    def __init__(self):
        self.pass_thresholds = {
            TaskDifficulty.EASY: 0.70,
            TaskDifficulty.MEDIUM: 0.60,
            TaskDifficulty.HARD: 0.65,
        }
    
    def grade_episode(self, 
                      env: T1DEnv, 
                      agent_fn: Callable[[State], Action],
                      verbose: bool = True) -> GradingResult:
        """
        Grade a single episode using provided agent function
        
        Args:
            env: Initialized T1D environment
            agent_fn: Function that takes State and returns Action
            verbose: Whether to print progress
        
        Returns:
            GradingResult with score and metrics
        """
        state = env.reset()
        total_reward = 0
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Grading Task: {env.task.value}")
            print(f"Episode length: {env.episode_length} steps ({env.episode_length * 5 / 60:.1f} hours)")
            print(f"Initial glucose: {state.glucose:.1f} mg/dL")
            print(f"{'='*60}\n")
        
        # Run episode
        for step in range(env.episode_length):
            action = agent_fn(state)
            state, reward, done, info = env.step(action)
            total_reward += reward
            
            if verbose and step % 12 == 0:  # Every hour
                print(f"Hour {step // 12:2d}: "
                      f"Glucose={state.glucose:6.1f} mg/dL, "
                      f"IOB={state.active_insulin:4.1f}U, "
                      f"COB={state.active_carbs:5.1f}g, "
                      f"TIR={info['time_in_range']:5.1%}")
        
        # Calculate final metrics
        metrics = self._calculate_metrics(env)
        
        # Get score
        score = env.get_score()
        
        # Determine pass/fail
        passed = score >= self.pass_thresholds[env.task]
        
        # Generate feedback
        feedback = self._generate_feedback(env.task, score, metrics, passed)
        
        if verbose:
            print(f"\n{'='*60}")
            print(feedback)
            print(f"{'='*60}\n")
        
        return GradingResult(
            task=env.task,
            score=score,
            passed=passed,
            metrics=metrics,
            feedback=feedback
        )
    
    def grade_all_tasks(self, 
                        agent_fn: Callable[[State], Action],
                        num_episodes: int = 5,
                        verbose: bool = True) -> Dict[TaskDifficulty, List[GradingResult]]:
        """
        Grade agent on all three tasks with multiple episodes
        
        Returns:
            Dictionary mapping task to list of grading results
        """
        results = {}
        
        for task in [TaskDifficulty.EASY, TaskDifficulty.MEDIUM, TaskDifficulty.HARD]:
            task_results = []
            
            if verbose:
                print(f"\n{'#'*60}")
                print(f"# EVALUATING TASK: {task.value.upper()}")
                print(f"# Running {num_episodes} episodes")
                print(f"{'#'*60}")
            
            for episode in range(num_episodes):
                if verbose:
                    print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
                
                env = T1DEnv(task=task)
                result = self.grade_episode(env, agent_fn, verbose=verbose)
                task_results.append(result)
            
            results[task] = task_results
            
            # Summary statistics
            if verbose:
                scores = [r.score for r in task_results]
                pass_rate = sum(r.passed for r in task_results) / num_episodes
                
                print(f"\n{'='*60}")
                print(f"TASK SUMMARY: {task.value}")
                print(f"{'='*60}")
                print(f"Mean Score:    {np.mean(scores):.3f} ± {np.std(scores):.3f}")
                print(f"Min Score:     {np.min(scores):.3f}")
                print(f"Max Score:     {np.max(scores):.3f}")
                print(f"Pass Rate:     {pass_rate:.1%} ({sum(r.passed for r in task_results)}/{num_episodes})")
                print(f"Pass Threshold: {self.pass_thresholds[task]:.2f}")
                print(f"{'='*60}\n")
        
        return results
    
    def _calculate_metrics(self, env: T1DEnv) -> Dict[str, float]:
        """Calculate comprehensive metrics from completed episode"""
        glucose_log = env.glucose_log
        
        # Time in range metrics
        tir = env._calculate_time_in_range()
        time_above = sum(1 for g in glucose_log if g > 180) / len(glucose_log)
        time_below = sum(1 for g in glucose_log if g < 70) / len(glucose_log)
        time_very_low = sum(1 for g in glucose_log if g < 54) / len(glucose_log)
        
        # Glucose statistics
        mean_glucose = np.mean(glucose_log)
        std_glucose = np.std(glucose_log)
        cv_glucose = (std_glucose / mean_glucose) * 100 if mean_glucose > 0 else 0
        
        # Min/max
        min_glucose = np.min(glucose_log)
        max_glucose = np.max(glucose_log)
        
        # Events
        hypo_events = env.hypo_events
        severe_hypo_events = env.severe_hypo_events
        
        # Glucose variability (MAGE - Mean Amplitude of Glycemic Excursions)
        excursions = []
        for i in range(1, len(glucose_log)):
            change = abs(glucose_log[i] - glucose_log[i-1])
            if change > std_glucose:
                excursions.append(change)
        mage = np.mean(excursions) if excursions else 0.0
        
        return {
            'time_in_range': tir,
            'time_above_range': time_above,
            'time_below_range': time_below,
            'time_very_low': time_very_low,
            'mean_glucose': mean_glucose,
            'std_glucose': std_glucose,
            'cv_glucose': cv_glucose,
            'min_glucose': min_glucose,
            'max_glucose': max_glucose,
            'hypo_events': hypo_events,
            'severe_hypo_events': severe_hypo_events,
            'mage': mage,
        }
    
    def _generate_feedback(self, 
                          task: TaskDifficulty, 
                          score: float, 
                          metrics: Dict[str, float],
                          passed: bool) -> str:
        """Generate detailed feedback for the agent"""
        feedback_lines = []
        
        # Header
        result_emoji = "✅" if passed else "❌"
        feedback_lines.append(f"{result_emoji} {'PASSED' if passed else 'FAILED'} - Score: {score:.3f}")
        feedback_lines.append("")
        
        # Key metrics
        feedback_lines.append("KEY METRICS:")
        feedback_lines.append(f"  Time in Range (70-180):  {metrics['time_in_range']:6.1%}")
        feedback_lines.append(f"  Time Above Range (>180): {metrics['time_above_range']:6.1%}")
        feedback_lines.append(f"  Time Below Range (<70):  {metrics['time_below_range']:6.1%}")
        feedback_lines.append(f"  Mean Glucose:            {metrics['mean_glucose']:6.1f} mg/dL")
        feedback_lines.append(f"  Glucose CV:              {metrics['cv_glucose']:6.1f}%")
        feedback_lines.append(f"  Min Glucose:             {metrics['min_glucose']:6.1f} mg/dL")
        feedback_lines.append(f"  Max Glucose:             {metrics['max_glucose']:6.1f} mg/dL")
        feedback_lines.append(f"  Severe Hypo Events:      {metrics['severe_hypo_events']:6d}")
        feedback_lines.append("")
        
        # Task-specific feedback
        if task == TaskDifficulty.EASY:
            feedback_lines.append("TASK: Fasting Control (8 hours)")
            feedback_lines.append(f"  Target: ≥80% Time in Range")
            feedback_lines.append(f"  Achieved: {metrics['time_in_range']:.1%}")
            if metrics['time_in_range'] >= 0.80:
                feedback_lines.append("  ✅ Target met!")
            else:
                shortfall = 0.80 - metrics['time_in_range']
                feedback_lines.append(f"  ⚠️  {shortfall:.1%} below target")
            
        elif task == TaskDifficulty.MEDIUM:
            feedback_lines.append("TASK: Single Meal Management (6 hours)")
            feedback_lines.append(f"  Target: ≥70% Time in Range")
            feedback_lines.append(f"  Achieved: {metrics['time_in_range']:.1%}")
            if metrics['time_in_range'] >= 0.70:
                feedback_lines.append("  ✅ Target met!")
            
        else:  # HARD
            feedback_lines.append("TASK: Full Day Management (24 hours)")
            feedback_lines.append(f"  Target: ≥70% Time in Range, <5% Time Low")
            feedback_lines.append(f"  TIR: {metrics['time_in_range']:.1%}")
            feedback_lines.append(f"  Time Low: {metrics['time_below_range']:.1%}")
            if metrics['time_in_range'] >= 0.70 and metrics['time_below_range'] < 0.05:
                feedback_lines.append("  ✅ Both targets met!")
        
        # Safety check
        feedback_lines.append("")
        if metrics['severe_hypo_events'] > 0:
            feedback_lines.append(f"🚨 SAFETY VIOLATION: {metrics['severe_hypo_events']} severe hypoglycemia events (<54 mg/dL)")
            feedback_lines.append("   Automatic failure - patient safety is paramount")
        else:
            feedback_lines.append("✅ No severe hypoglycemia events - safe control")
        
        # Recommendations
        feedback_lines.append("")
        feedback_lines.append("RECOMMENDATIONS:")
        
        if metrics['time_above_range'] > 0.20:
            feedback_lines.append("  • Reduce time above range - consider more aggressive bolus dosing")
        
        if metrics['time_below_range'] > 0.10:
            feedback_lines.append("  • Too much time in hypoglycemia - reduce insulin doses")
        
        if metrics['cv_glucose'] > 36:
            feedback_lines.append(f"  • High glucose variability (CV={metrics['cv_glucose']:.1f}%) - aim for smoother control")
        
        if metrics['time_in_range'] < 0.70:
            feedback_lines.append("  • Focus on time in range - balance insulin dosing with carb intake")
        
        return "\n".join(feedback_lines)


# Baseline agents for testing

def baseline_basal_only(state: State) -> Action:
    """Simplest baseline: No bolus insulin, just basal"""
    return Action(insulin_bolus=0.0)


def baseline_fixed_bolus(state: State) -> Action:
    """Fixed bolus on meal announcement"""
    if state.meal_announced > 0:
        # Simple 1:10 carb ratio
        bolus = state.meal_announced / 10.0
        return Action(insulin_bolus=bolus)
    return Action(insulin_bolus=0.0)


def baseline_proportional_controller(state: State) -> Action:
    """
    Proportional controller based on glucose error
    
    Simple control law:
    - If glucose > 180: give correction bolus
    - If meal announced: give meal bolus
    - Account for insulin on board
    """
    target_glucose = 100.0
    correction_factor = 50.0  # mg/dL per unit
    carb_ratio = 10.0  # grams per unit
    
    bolus = 0.0
    
    # Meal bolus
    if state.meal_announced > 0:
        meal_bolus = state.meal_announced / carb_ratio
        bolus += meal_bolus
    
    # Correction bolus
    glucose_error = state.glucose - target_glucose
    if glucose_error > 30:  # Only correct if significantly high
        correction_bolus = glucose_error / correction_factor
        # Reduce by insulin on board
        correction_bolus = max(0, correction_bolus - state.active_insulin * 0.5)
        bolus += correction_bolus
    
    return Action(insulin_bolus=bolus)


if __name__ == "__main__":
    grader = T1DGrader()
    
    print("="*60)
    print("TESTING BASELINE AGENTS")
    print("="*60)
    
    # Test each baseline
    agents = [
        ("Basal Only", baseline_basal_only),
        ("Fixed Bolus", baseline_fixed_bolus),
        ("Proportional Controller", baseline_proportional_controller),
    ]
    
    for agent_name, agent_fn in agents:
        print(f"\n{'#'*60}")
        print(f"# AGENT: {agent_name}")
        print(f"{'#'*60}")
        
        results = grader.grade_all_tasks(agent_fn, num_episodes=3, verbose=True)
        
        # Overall summary
        print(f"\n{'='*60}")
        print(f"OVERALL SUMMARY: {agent_name}")
        print(f"{'='*60}")
        
        for task in [TaskDifficulty.EASY, TaskDifficulty.MEDIUM, TaskDifficulty.HARD]:
            task_results = results[task]
            mean_score = np.mean([r.score for r in task_results])
            pass_rate = sum(r.passed for r in task_results) / len(task_results)
            
            print(f"{task.value:15s}: Score={mean_score:.3f}, Pass Rate={pass_rate:.1%}")
