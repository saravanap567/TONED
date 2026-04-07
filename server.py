"""
OpenEnv Server for T1D Environment
Meta PyTorch Hackathon - Round 1

This wraps the T1D environment in a FastAPI server
to expose it via HTTP API for OpenEnv compliance.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn

from t1d_env import T1DEnv, TaskDifficulty, Action, Observation, Reward

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="T1D Diabetes Environment Server",
    description="OpenEnv-compliant server for Type 1 Diabetes management",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# GLOBAL ENVIRONMENT INSTANCE
# ============================================================================

env_instance: Optional[T1DEnv] = None

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ResetRequest(BaseModel):
    """Request model for reset endpoint"""
    task: str = "easy"  # easy, medium, or hard


class ResetResponse(BaseModel):
    """Response model for reset endpoint"""
    observation: Dict[str, Any]
    info: Dict[str, Any] = {}


class StepRequest(BaseModel):
    """Request model for step endpoint"""
    action: Dict[str, float]  # {"insulin_bolus": 0.0}


class StepResponse(BaseModel):
    """Response model for step endpoint"""
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "name": "T1D Diabetes Environment Server",
        "version": "1.0.0",
        "hackathon": "Meta PyTorch Hackathon - Round 1",
        "endpoints": {
            "POST /reset": "Reset environment",
            "POST /step": "Take action",
            "GET /state": "Get current state",
            "GET /info": "Get environment info"
        }
    }


@app.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest):
    """
    Reset the environment
    
    POST /reset with {"task": "easy|medium|hard"}
    Returns initial observation
    """
    global env_instance
    
    # Map task string to difficulty
    task_map = {
        "easy": TaskDifficulty.EASY,
        "medium": TaskDifficulty.MEDIUM,
        "hard": TaskDifficulty.HARD
    }
    
    task_difficulty = task_map.get(request.task, TaskDifficulty.EASY)
    
    # Create new environment
    env_instance = T1DEnv(task=task_difficulty)
    
    # Reset and get initial observation
    obs = env_instance.reset()
    
    # Convert observation to dict
    obs_dict = {
        "glucose": obs.glucose,
        "active_insulin": obs.active_insulin,
        "active_carbs": obs.active_carbs,
        "time_of_day": obs.time_of_day,
        "meal_announced": obs.meal_announced,
        "exercise_level": obs.exercise_level,
        "glucose_history": obs.glucose_history,
        "insulin_history": obs.insulin_history
    }
    
    return ResetResponse(
        observation=obs_dict,
        info={"task": request.task, "episode_length": env_instance.episode_length}
    )


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    """
    Take a step in the environment
    
    POST /step with {"action": {"insulin_bolus": 0.5}}
    Returns (observation, reward, done, info)
    """
    global env_instance
    
    if env_instance is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    # Parse action
    try:
        action = Action(insulin_bolus=request.action.get("insulin_bolus", 0.0))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid action: {e}")
    
    # Step environment
    try:
        obs, reward, done, info = env_instance.step(action)
        
        # Convert to dict
        obs_dict = {
            "glucose": obs.glucose,
            "active_insulin": obs.active_insulin,
            "active_carbs": obs.active_carbs,
            "time_of_day": obs.time_of_day,
            "meal_announced": obs.meal_announced,
            "exercise_level": obs.exercise_level,
            "glucose_history": obs.glucose_history,
            "insulin_history": obs.insulin_history
        }
        
        # Get reward value (handle both Reward object and float)
        if isinstance(reward, Reward):
            reward_value = reward.total
        else:
            reward_value = float(reward)
        
        return StepResponse(
            observation=obs_dict,
            reward=reward_value,
            done=done,
            info=info
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step failed: {e}")


@app.get("/state")
def get_state():
    """Get current environment state"""
    global env_instance
    
    if env_instance is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    obs = env_instance.state()
    
    return {
        "glucose": obs.glucose,
        "active_insulin": obs.active_insulin,
        "active_carbs": obs.active_carbs,
        "time_of_day": obs.time_of_day,
        "meal_announced": obs.meal_announced,
        "exercise_level": obs.exercise_level,
        "glucose_history": obs.glucose_history,
        "insulin_history": obs.insulin_history
    }


@app.get("/info")
def get_info():
    """Get environment information"""
    global env_instance
    
    if env_instance is None:
        return {
            "status": "not_initialized",
            "message": "Call /reset to initialize environment"
        }
    
    return {
        "status": "ready",
        "task": env_instance.task.value,
        "episode_length": env_instance.episode_length,
        "current_step": env_instance.current_step,
        "time_in_range": env_instance._calculate_time_in_range()
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860,
        log_level="info"
    )
