from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import numpy as np
import uvicorn

app = FastAPI()


# Pydantic models for request/response
class ResetConfig(BaseModel):
    """Optional configuration for environment reset"""
    seed: Optional[int] = None
    options: Optional[Dict[str, Any]] = None


class ResetResponse(BaseModel):
    """Response from reset endpoint"""
    observation: List[float]
    info: Dict[str, Any]


class StepRequest(BaseModel):
    """Request for step endpoint"""
    action: List[float]


class StepResponse(BaseModel):
    """Response from step endpoint"""
    observation: List[float]
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]


class ObservationSpaceResponse(BaseModel):
    """Observation space specification"""
    type: str
    shape: List[int]
    low: List[float]
    high: List[float]
    dtype: str


class ActionSpaceResponse(BaseModel):
    """Action space specification"""
    type: str
    shape: List[int]
    low: List[float]
    high: List[float]
    dtype: str


# Global environment state
class GlucoseEnvironment:
    """Simple glucose dynamics environment"""
    
    def __init__(self):
        self.state = None
        self.steps = 0
        self.max_steps = 288  # 24 hours * 12 (5-min intervals)
        self.target_glucose = 100.0
        self.current_glucose = 120.0
        self.insulin_effect = 0.0
        self.carb_effect = 0.0
        
    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        
        self.current_glucose = np.random.uniform(80, 180)
        self.insulin_effect = 0.0
        self.carb_effect = 0.0
        self.steps = 0
        
        obs = self._get_observation()
        info = {
            "glucose": float(self.current_glucose),
            "target": float(self.target_glucose),
            "steps": self.steps
        }
        
        return obs, info
    
    def step(self, action: List[float]):
        # Action: [insulin_dose, carb_intake]
        insulin_dose = action[0] if len(action) > 0 else 0.0
        carb_intake = action[1] if len(action) > 1 else 0.0
        
        # Update insulin effect (decreases glucose)
        self.insulin_effect = insulin_dose * 5.0
        
        # Update carb effect (increases glucose)
        self.carb_effect = carb_intake * 3.0
        
        # Glucose dynamics
        glucose_change = -self.insulin_effect + self.carb_effect + np.random.normal(0, 2)
        self.current_glucose += glucose_change
        
        # Decay effects
        self.insulin_effect *= 0.8
        self.carb_effect *= 0.7
        
        # Constrain glucose to realistic range
        self.current_glucose = np.clip(self.current_glucose, 40, 400)
        
        # Calculate reward (closer to target is better)
        glucose_error = abs(self.current_glucose - self.target_glucose)
        reward = -glucose_error / 100.0
        
        # Add penalty for extreme values
        if self.current_glucose < 70 or self.current_glucose > 180:
            reward -= 1.0
        
        self.steps += 1
        
        # Check termination
        terminated = self.current_glucose < 40 or self.current_glucose > 400
        truncated = self.steps >= self.max_steps
        
        obs = self._get_observation()
        info = {
            "glucose": float(self.current_glucose),
            "target": float(self.target_glucose),
            "steps": self.steps,
            "insulin_effect": float(self.insulin_effect),
            "carb_effect": float(self.carb_effect)
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self):
        return [
            float(self.current_glucose),
            float(self.target_glucose),
            float(self.insulin_effect),
            float(self.carb_effect),
            float(self.steps) / self.max_steps
        ]


# Global environment instance
env = GlucoseEnvironment()


@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "environment": "Glucose Dynamics RL Environment",
        "version": "1.0.0"
    }


@app.post("/reset", response_model=ResetResponse)
def reset(config: Optional[ResetConfig] = None):
    """
    Reset the environment to initial state.
    
    This endpoint accepts an OPTIONAL body with seed and options.
    If no body is provided, it uses default values.
    """
    seed = None
    if config is not None and config.seed is not None:
        seed = config.seed
    
    obs, info = env.reset(seed=seed)
    
    return ResetResponse(
        observation=obs,
        info=info
    )


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    """
    Take a step in the environment with the given action.
    
    Action: [insulin_dose, carb_intake]
    - insulin_dose: Units of insulin (0-10)
    - carb_intake: Grams of carbohydrates (0-100)
    """
    obs, reward, terminated, truncated, info = env.step(request.action)
    
    return StepResponse(
        observation=obs,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        info=info
    )


@app.get("/observation_space", response_model=ObservationSpaceResponse)
def observation_space():
    """
    Return the observation space specification.
    
    Observation: [current_glucose, target_glucose, insulin_effect, carb_effect, time_progress]
    """
    return ObservationSpaceResponse(
        type="Box",
        shape=[5],
        low=[0.0, 0.0, 0.0, 0.0, 0.0],
        high=[500.0, 200.0, 100.0, 300.0, 1.0],
        dtype="float32"
    )


@app.get("/action_space", response_model=ActionSpaceResponse)
def action_space():
    """
    Return the action space specification.
    
    Action: [insulin_dose, carb_intake]
    - insulin_dose: 0-10 units
    - carb_intake: 0-100 grams
    """
    return ActionSpaceResponse(
        type="Box",
        shape=[2],
        low=[0.0, 0.0],
        high=[10.0, 100.0],
        dtype="float32"
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
