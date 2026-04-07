"""
OpenEnv Server for T1D Environment
Compatible with original t1d_env.py (non-Pydantic version)

Meta PyTorch Hackathon - Round 1
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uvicorn

from t1d_env import T1DEnv, TaskDifficulty, Action, State

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
current_state: Optional[State] = None

# ============================================================================
# REQUEST/RESPONSE MODELS (Pydantic)
# ============================================================================

class ResetRequest(BaseModel):
    """Request model for reset endpoint"""
    task: str = "easy"


class ResetResponse(BaseModel):
    """Response model for reset endpoint"""
    observation: Dict[str, Any]
    info: Dict[str, Any] = {}


class StepRequest(BaseModel):
    """Request model for step endpoint"""
    action: Dict[str, float]


class StepResponse(BaseModel):
    """Response model for step endpoint"""
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def state_to_dict(state: State) -> Dict[str, Any]:
    """Convert State to dictionary"""
    return {
        "glucose": state.glucose,
        "active_insulin": state.active_insulin,
        "active_carbs": state.active_carbs,
        "time_of_day": state.time_of_day,
        "meal_announced": state.meal_announced,
        "exercise_level": state.exercise_level,
        "glucose_history": state.glucose_history,
        "insulin_history": state.insulin_history
    }


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
        "status": "running",
        "endpoints": {
            "POST /reset": "Reset environment with task parameter",
            "POST /step": "Take action in environment",
            "GET /state": "Get current state",
            "GET /info": "Get environment information"
        }
    }


@app.post("/reset")
def reset(request: ResetRequest):
    """
    Reset the environment
    
    POST /reset with {"task": "easy|medium|hard"}
    Returns initial observation
    """
    global env_instance, current_state
    
    # Map task string to difficulty
    task_map = {
        "easy": TaskDifficulty.EASY,
        "medium": TaskDifficulty.MEDIUM,
        "hard": TaskDifficulty.HARD
    }
    
    task_difficulty = task_map.get(request.task, TaskDifficulty.EASY)
    
    try:
        # Create new environment
        env_instance = T1DEnv(task=task_difficulty)
        
        # Reset and get initial observation
        current_state = env_instance.reset()
        
        # Convert state to dict
        obs_dict = state_to_dict(current_state)
        
        return {
            "observation": obs_dict,
            "info": {
                "task": request.task,
                "episode_length": env_instance.episode_length,
                "target_range": [70, 180]
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@app.post("/step")
def step(request: StepRequest):
    """
    Take a step in the environment
    
    POST /step with {"action": {"insulin_bolus": 2.5}}
    Returns (observation, reward, done, info)
    """
    global env_instance, current_state
    
    if env_instance is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first."
        )
    
    try:
        # Parse action
        insulin_bolus = request.action.get("insulin_bolus", 0.0)
        action = Action(insulin_bolus=insulin_bolus)
        
        # Step environment
        current_state, reward, done, info = env_instance.step(action)
        
        # Convert state to dict
        obs_dict = state_to_dict(current_state)
        
        # Handle reward (could be float or Reward object)
        if hasattr(reward, 'total'):
            reward_value = reward.total
        else:
            reward_value = float(reward)
        
        return {
            "observation": obs_dict,
            "reward": reward_value,
            "done": done,
            "info": info
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}")


@app.get("/state")
def get_state():
    """Get current environment state"""
    global env_instance, current_state
    
    if env_instance is None or current_state is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first."
        )
    
    return {"state": state_to_dict(current_state)}


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


@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "t1d-env-server"}


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("Starting T1D Environment Server...")
    print("OpenEnv API endpoints: /reset, /step, /state, /info")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860,
        log_level="info"
    )
