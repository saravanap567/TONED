"""
Type 1 Diabetes Management Environment
OpenEnv-compliant RL environment for insulin dosing decisions
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from enum import Enum


class TaskDifficulty(Enum):
    EASY = "fasting_control"
    MEDIUM = "single_meal"
    HARD = "full_day"


@dataclass
class PatientParams:
    """Patient-specific physiological parameters"""
    weight: float = 70.0  # kg
    insulin_sensitivity: float = 50.0  # mg/dL per unit insulin
    carb_ratio: float = 10.0  # grams per unit insulin
    basal_rate: float = 1.0  # units per hour
    glucose_baseline: float = 100.0  # mg/dL
    
    # Absorption rates
    insulin_absorption_time: float = 4.0  # hours (rapid-acting)
    carb_absorption_time: float = 2.0  # hours
    
    # Variability factors
    insulin_variability: float = 0.15  # 15% coefficient of variation
    carb_variability: float = 0.20  # 20% for carb counting errors


@dataclass
class State:
    """Environment state at each timestep"""
    glucose: float  # Current blood glucose (mg/dL)
    active_insulin: float  # Insulin on board (units)
    active_carbs: float  # Carbs on board (grams)
    time_of_day: float  # 0-24 hours
    meal_announced: float  # Upcoming meal carbs (0 if none)
    exercise_level: float  # 0-1 intensity
    
    # History (last 12 steps = 1 hour with 5min intervals)
    glucose_history: List[float]
    insulin_history: List[float]
    
    def to_dict(self) -> Dict:
        return {
            'glucose': self.glucose,
            'active_insulin': self.active_insulin,
            'active_carbs': self.active_carbs,
            'time_of_day': self.time_of_day,
            'meal_announced': self.meal_announced,
            'exercise_level': self.exercise_level,
            'glucose_history': self.glucose_history,
            'insulin_history': self.insulin_history
        }


@dataclass
class Action:
    """Agent's action space"""
    insulin_bolus: float  # Units of insulin to deliver (0-20 units)
    
    def to_dict(self) -> Dict:
        return {'insulin_bolus': self.insulin_bolus}


class T1DEnv:
    """
    Type 1 Diabetes Management Environment
    
    Observation Space:
        - glucose: Current blood glucose [40, 400] mg/dL
        - active_insulin: Insulin on board [0, 50] units
        - active_carbs: Carbs being absorbed [0, 200] grams
        - time_of_day: Hour of day [0, 24]
        - meal_announced: Upcoming meal [0, 150] grams
        - exercise_level: Exercise intensity [0, 1]
        - glucose_history: Last 12 readings (1 hour)
        - insulin_history: Last 12 doses
    
    Action Space:
        - insulin_bolus: Continuous [0, 20] units
    
    Timestep: 5 minutes (288 steps per day)
    """
    
    def __init__(self, task: TaskDifficulty = TaskDifficulty.EASY, patient_params: Optional[PatientParams] = None):
        self.task = task
        self.patient = patient_params or PatientParams()
        self.timestep = 5.0 / 60.0  # 5 minutes in hours
        
        # Episode configuration based on task
        self.episode_length = self._get_episode_length()
        self.meal_schedule = self._get_meal_schedule()
        self.exercise_schedule = self._get_exercise_schedule()
        
        # State tracking
        self.current_step = 0
        self.state: Optional[State] = None
        
        # Insulin and carb kinetics (using exponential decay model)
        self.insulin_queue: List[Tuple[float, float]] = []  # (amount, time_remaining)
        self.carb_queue: List[Tuple[float, float]] = []
        
        # Safety limits
        self.glucose_min = 40.0  # Critical low
        self.glucose_max = 400.0  # Critical high
        self.target_min = 70.0
        self.target_max = 180.0
        
        # Metrics for grading
        self.glucose_log: List[float] = []
        self.hypo_events = 0
        self.severe_hypo_events = 0
        
    def _get_episode_length(self) -> int:
        """Returns number of timesteps based on task"""
        if self.task == TaskDifficulty.EASY:
            return 96  # 8 hours (overnight)
        elif self.task == TaskDifficulty.MEDIUM:
            return 72  # 6 hours (meal + recovery)
        else:  # HARD
            return 288  # 24 hours (full day)
    
    def _get_meal_schedule(self) -> List[Tuple[int, float]]:
        """Returns (timestep, carbs) for scheduled meals"""
        if self.task == TaskDifficulty.EASY:
            return []  # No meals during fasting
        elif self.task == TaskDifficulty.MEDIUM:
            return [(12, 60.0)]  # Single 60g meal at 1 hour
        else:  # HARD
            return [
                (48, 45.0),   # Breakfast at 4 hours (8am if start at 4am)
                (108, 60.0),  # Lunch at 9 hours (1pm)
                (180, 75.0),  # Dinner at 15 hours (7pm)
            ]
    
    def _get_exercise_schedule(self) -> List[Tuple[int, int, float]]:
        """Returns (start_step, duration_steps, intensity) for exercise"""
        if self.task == TaskDifficulty.HARD:
            return [(120, 12, 0.6)]  # Moderate exercise for 1 hour after lunch
        return []
    
    def reset(self) -> State:
        """Reset environment to initial state"""
        self.current_step = 0
        self.glucose_log = []
        self.hypo_events = 0
        self.severe_hypo_events = 0
        self.insulin_queue = []
        self.carb_queue = []
        
        # Initialize with slight variability
        initial_glucose = self.patient.glucose_baseline + np.random.normal(0, 10)
        initial_glucose = np.clip(initial_glucose, self.glucose_min, self.glucose_max)
        
        self.state = State(
            glucose=initial_glucose,
            active_insulin=0.0,
            active_carbs=0.0,
            time_of_day=4.0 if self.task == TaskDifficulty.HARD else 0.0,  # Start at 4am for full day
            meal_announced=0.0,
            exercise_level=0.0,
            glucose_history=[initial_glucose] * 12,
            insulin_history=[0.0] * 12
        )
        
        self.glucose_log.append(initial_glucose)
        return self.state
    
    def step(self, action: Action) -> Tuple[State, float, bool, Dict]:
        """
        Execute one timestep
        
        Returns:
            state: New state
            reward: Immediate reward
            done: Whether episode is complete
            info: Additional metrics
        """
        # Validate action
        insulin_dose = np.clip(action.insulin_bolus, 0.0, 20.0)
        
        # Add basal insulin
        basal_dose = self.patient.basal_rate * self.timestep
        total_insulin = insulin_dose + basal_dose
        
        # Add insulin with variability
        actual_insulin = total_insulin * (1 + np.random.normal(0, self.patient.insulin_variability))
        self.insulin_queue.append((actual_insulin, self.patient.insulin_absorption_time))
        
        # Check for scheduled meals
        for meal_step, meal_carbs in self.meal_schedule:
            if self.current_step == meal_step:
                # Announce meal 1 step (5 min) before
                self.state.meal_announced = meal_carbs
            elif self.current_step == meal_step + 1:
                # Deliver meal with carb counting error
                actual_carbs = meal_carbs * (1 + np.random.normal(0, self.patient.carb_variability))
                self.carb_queue.append((actual_carbs, self.patient.carb_absorption_time))
                self.state.meal_announced = 0.0
        
        # Check for exercise
        exercise_intensity = 0.0
        for start, duration, intensity in self.exercise_schedule:
            if start <= self.current_step < start + duration:
                exercise_intensity = intensity
        self.state.exercise_level = exercise_intensity
        
        # Update insulin and carb kinetics
        self._update_kinetics()
        
        # Calculate glucose change
        glucose_delta = self._calculate_glucose_change(exercise_intensity)
        
        # Update glucose
        new_glucose = self.state.glucose + glucose_delta
        new_glucose = np.clip(new_glucose, self.glucose_min, self.glucose_max)
        
        # Track hypoglycemia events
        if new_glucose < self.target_min:
            self.hypo_events += 1
        if new_glucose < 54.0:  # Severe hypoglycemia threshold
            self.severe_hypo_events += 1
        
        # Update state
        self.state.glucose = new_glucose
        self.state.time_of_day = (self.state.time_of_day + self.timestep) % 24
        self.state.glucose_history.append(new_glucose)
        self.state.glucose_history = self.state.glucose_history[-12:]
        self.state.insulin_history.append(insulin_dose)
        self.state.insulin_history = self.state.insulin_history[-12:]
        
        self.glucose_log.append(new_glucose)
        self.current_step += 1
        
        # Calculate reward
        reward = self._calculate_reward(new_glucose, insulin_dose)
        
        # Check if episode is done
        done = self.current_step >= self.episode_length
        
        # Info dict
        info = {
            'time_in_range': self._calculate_time_in_range(),
            'hypo_events': self.hypo_events,
            'severe_hypo_events': self.severe_hypo_events,
            'glucose': new_glucose,
            'active_insulin': self.state.active_insulin,
        }
        
        return self.state, reward, done, info
    
    def _update_kinetics(self):
        """Update insulin and carb queues with exponential decay"""
        # Update insulin on board
        new_insulin_queue = []
        active_insulin = 0.0
        for amount, time_remaining in self.insulin_queue:
            new_time = time_remaining - self.timestep
            if new_time > 0:
                # Exponential decay
                decay_rate = 1.0 / self.patient.insulin_absorption_time
                remaining = amount * np.exp(-decay_rate * self.timestep)
                new_insulin_queue.append((remaining, new_time))
                active_insulin += remaining
        self.insulin_queue = new_insulin_queue
        self.state.active_insulin = active_insulin
        
        # Update carbs on board
        new_carb_queue = []
        active_carbs = 0.0
        for amount, time_remaining in self.carb_queue:
            new_time = time_remaining - self.timestep
            if new_time > 0:
                decay_rate = 1.0 / self.patient.carb_absorption_time
                remaining = amount * np.exp(-decay_rate * self.timestep)
                new_carb_queue.append((remaining, new_time))
                active_carbs += remaining
        self.carb_queue = new_carb_queue
        self.state.active_carbs = active_carbs
    
    def _calculate_glucose_change(self, exercise_intensity: float) -> float:
        """Calculate glucose change based on insulin and carbs"""
        # Glucose decrease from insulin
        insulin_effect = -self.state.active_insulin * self.patient.insulin_sensitivity * (self.timestep / 4.0)
        
        # Glucose increase from carbs
        carb_effect = self.state.active_carbs * 4.0 * (self.timestep / 2.0)  # 4 mg/dL per gram carb
        
        # Exercise effect (increases insulin sensitivity)
        exercise_multiplier = 1.0 + (exercise_intensity * 0.5)
        insulin_effect *= exercise_multiplier
        
        # Hepatic glucose production (reduced by insulin)
        hgp = 2.0 * self.timestep * (1.0 - min(self.state.active_insulin / 10.0, 0.8))
        
        # Random noise
        noise = np.random.normal(0, 2.0)
        
        return insulin_effect + carb_effect + hgp + noise
    
    def _calculate_reward(self, glucose: float, insulin_dose: float) -> float:
        """
        Reward function with partial progress signals
        
        Components:
        1. Time in range reward (primary)
        2. Hypoglycemia penalty (safety)
        3. Variability penalty (smooth control)
        4. Insulin economy (avoid over-dosing)
        """
        # 1. Time in range (0 to 1)
        if self.target_min <= glucose <= self.target_max:
            range_reward = 1.0
        elif glucose < self.target_min:
            # Penalize hypoglycemia more severely
            range_reward = -2.0 * (self.target_min - glucose) / self.target_min
        else:
            # Penalize hyperglycemia less severely
            range_reward = -0.5 * (glucose - self.target_max) / 100.0
        
        # 2. Severe hypoglycemia penalty
        if glucose < 54.0:
            safety_penalty = -10.0
        elif glucose < 70.0:
            safety_penalty = -1.0
        else:
            safety_penalty = 0.0
        
        # 3. Glucose variability penalty
        if len(self.state.glucose_history) >= 2:
            glucose_change = abs(self.state.glucose - self.state.glucose_history[-2])
            variability_penalty = -0.1 * (glucose_change / 50.0)
        else:
            variability_penalty = 0.0
        
        # 4. Insulin economy (discourage excessive dosing)
        insulin_penalty = -0.01 * insulin_dose
        
        # Total reward
        reward = range_reward + safety_penalty + variability_penalty + insulin_penalty
        
        return reward
    
    def _calculate_time_in_range(self) -> float:
        """Calculate percentage of time in target range"""
        if len(self.glucose_log) == 0:
            return 0.0
        in_range = sum(1 for g in self.glucose_log if self.target_min <= g <= self.target_max)
        return in_range / len(self.glucose_log)
    
    def get_score(self) -> float:
        """
        Calculate final score (0.0 to 1.0) based on task
        
        Scoring criteria:
        - EASY: 80% TIR, no severe hypos
        - MEDIUM: 70% TIR, return to baseline, no severe hypos
        - HARD: 70% TIR, <5% time in hypo, no severe hypos
        """
        tir = self._calculate_time_in_range()
        
        if self.task == TaskDifficulty.EASY:
            # Simple TIR-based scoring
            if self.severe_hypo_events > 0:
                return 0.0
            return min(tir / 0.80, 1.0)
        
        elif self.task == TaskDifficulty.MEDIUM:
            # TIR + recovery to baseline
            final_glucose = self.glucose_log[-1]
            recovery_score = 1.0 - abs(final_glucose - self.patient.glucose_baseline) / 100.0
            recovery_score = max(0.0, recovery_score)
            
            if self.severe_hypo_events > 0:
                return 0.0
            
            return 0.7 * (tir / 0.70) + 0.3 * recovery_score
        
        else:  # HARD
            # TIR + time below range + severe hypo events
            time_below = sum(1 for g in self.glucose_log if g < self.target_min) / len(self.glucose_log)
            
            if self.severe_hypo_events > 0:
                return 0.0
            
            tir_score = min(tir / 0.70, 1.0)
            safety_score = max(0.0, 1.0 - time_below / 0.05)
            
            return 0.8 * tir_score + 0.2 * safety_score


# Example usage
if __name__ == "__main__":
    # Test easy task
    env = T1DEnv(task=TaskDifficulty.EASY)
    state = env.reset()
    
    print(f"Task: {env.task.value}")
    print(f"Episode length: {env.episode_length} steps ({env.episode_length * 5 / 60:.1f} hours)")
    print(f"Initial glucose: {state.glucose:.1f} mg/dL")
    print()
    
    # Simple baseline: constant basal only
    total_reward = 0
    for step in range(env.episode_length):
        action = Action(insulin_bolus=0.0)  # No bolus, just basal
        state, reward, done, info = env.step(action)
        total_reward += reward
        
        if step % 12 == 0:  # Every hour
            print(f"Hour {step // 12}: Glucose={state.glucose:.1f}, TIR={info['time_in_range']:.2%}")
    
    final_score = env.get_score()
    print(f"\nFinal Score: {final_score:.3f}")
    print(f"Total Reward: {total_reward:.1f}")
    print(f"Time in Range: {env._calculate_time_in_range():.2%}")
    print(f"Hypo Events: {env.hypo_events}")
    print(f"Severe Hypo Events: {env.severe_hypo_events}")
