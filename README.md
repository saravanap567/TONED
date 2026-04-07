
---
title: TONED
emoji: 📈
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: 6.11.0
app_file: app.py
pinned: false
license: mit
---
sdk: docker

# 🏥 Type 1 Diabetes RL Environment Server

**Meta PyTorch Hackathon - Round 1 Submission**

OpenEnv-compliant server for Type 1 Diabetes blood glucose management through intelligent insulin dosing.

## 🚀 OpenEnv API Endpoints

This server exposes the T1D environment via HTTP API:

- **POST /reset** - Initialize environment
- **POST /step** - Take action  
- **GET /state** - Get current state
- **GET /info** - Environment information

## 🎮 Three Tasks

- **Easy**: 8-hour fasting control (≥80% time in range)
- **Medium**: Single meal management (≥70% time in range)
- **Hard**: Full day with meals & exercise (≥70% time in range)

## 🤖 Features

- 6 baseline agents (basal only → adaptive)
- Realistic physiological model
- Automated grading (0.0-1.0 scores)
- Safety-critical (prevents hypoglycemia)
- Clinically validated (ADA guidelines)

## 📊 Environment

**Observation**: Glucose, insulin on board, carbs, time, meals, exercise, history

**Action**: Insulin bolus dose (0-20 units)

**Reward**: Time in range + safety + stability

## 🏥 Impact

Affects 1.6M people with Type 1 Diabetes in the US

## 🔗 API Usage

```python
import requests

# Reset environment
response = requests.post("http://localhost:7860/reset", json={"task": "easy"})
obs = response.json()

# Take step
response = requests.post("http://localhost:7860/step", json={"action": {"insulin_bolus": 2.5}})
result = response.json()
```

---

**Built with**: Python, FastAPI, OpenEnv, Pydantic