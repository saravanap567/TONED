"""
Interactive Gradio App for T1D Environment
Meta PyTorch Hackathon - Round 1
"""

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
from typing import List, Tuple

# Import environment components
from t1d_env import T1DEnv, TaskDifficulty, Action, State, PatientParams
from grading import T1DGrader
from baseline_agents import (
    baseline_basal_only,
    baseline_fixed_bolus,
    baseline_proportional_controller,
    baseline_pid_controller,
    baseline_model_predictive,
    baseline_adaptive
)


def create_glucose_plot(glucose_log: List[float], timestep: float = 5.0) -> Image.Image:
    """Create glucose timeline visualization"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Time axis in hours
    time_hours = np.arange(len(glucose_log)) * (timestep / 60.0)
    
    # Plot glucose
    ax.plot(time_hours, glucose_log, 'b-', linewidth=2, label='Blood Glucose')
    
    # Target range
    ax.axhline(y=70, color='green', linestyle='--', alpha=0.5, label='Target Range')
    ax.axhline(y=180, color='green', linestyle='--', alpha=0.5)
    ax.fill_between(time_hours, 70, 180, alpha=0.2, color='green')
    
    # Hypoglycemia zones
    ax.fill_between(time_hours, 40, 70, alpha=0.2, color='yellow', label='Hypoglycemia')
    ax.fill_between(time_hours, 40, 54, alpha=0.3, color='red', label='Severe Hypoglycemia')
    
    # Hyperglycemia
    ax.fill_between(time_hours, 180, 400, alpha=0.1, color='orange', label='Hyperglycemia')
    
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('Blood Glucose (mg/dL)', fontsize=12)
    ax.set_title('Blood Glucose Timeline', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(40, 400)
    
    # Convert to image
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    
    return img


def create_metrics_plot(metrics: dict) -> Image.Image:
    """Create metrics visualization"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Time distribution pie chart
    ax = axes[0]
    sizes = [
        metrics['time_in_range'] * 100,
        metrics['time_above_range'] * 100,
        metrics['time_below_range'] * 100
    ]
    labels = ['In Range\n(70-180)', 'Above Range\n(>180)', 'Below Range\n(<70)']
    colors = ['#4CAF50', '#FF9800', '#F44336']
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title('Time Distribution', fontweight='bold')
    
    # Glucose statistics
    ax = axes[1]
    stats = ['Mean', 'Std Dev', 'Min', 'Max']
    values = [
        metrics['mean_glucose'],
        metrics['std_glucose'],
        metrics['min_glucose'],
        metrics['max_glucose']
    ]
    bars = ax.bar(stats, values, color=['#2196F3', '#9C27B0', '#F44336', '#FF9800'])
    ax.set_ylabel('mg/dL', fontsize=10)
    ax.set_title('Glucose Statistics', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}',
                ha='center', va='bottom', fontsize=9)
    
    # Safety metrics
    ax = axes[2]
    safety_metrics = ['CV\n(%)', 'Hypo\nEvents', 'Severe\nHypos']
    safety_values = [
        metrics['cv_glucose'],
        metrics['hypo_events'],
        metrics['severe_hypo_events']
    ]
    colors_safety = ['#2196F3', '#FF9800', '#F44336']
    bars = ax.bar(safety_metrics, safety_values, color=colors_safety)
    ax.set_title('Safety Metrics', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=9)
    
    # Convert to image
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    
    return img


def run_simulation(task_name: str, agent_name: str, num_episodes: int = 3) -> Tuple[str, Image.Image, Image.Image]:
    """Run simulation and return results"""
    
    # Map task name to difficulty
    task_map = {
        "Easy: Fasting Control (8 hours)": TaskDifficulty.EASY,
        "Medium: Single Meal (6 hours)": TaskDifficulty.MEDIUM,
        "Hard: Full Day (24 hours)": TaskDifficulty.HARD,
    }
    task = task_map[task_name]
    
    # Map agent name to function
    agent_map = {
        "Basal Only (Baseline)": baseline_basal_only,
        "Fixed Bolus": baseline_fixed_bolus,
        "Proportional Controller": baseline_proportional_controller,
        "PID Controller": baseline_pid_controller,
        "Model Predictive": baseline_model_predictive,
        "Adaptive (Recommended)": baseline_adaptive,
    }
    agent_fn = agent_map[agent_name]
    
    # Create grader
    grader = T1DGrader()
    
    # Run episodes
    results = []
    for _ in range(num_episodes):
        env = T1DEnv(task=task)
        result = grader.grade_episode(env, agent_fn, verbose=False)
        results.append(result)
    
    # Get last episode for visualization
    env = T1DEnv(task=task)
    state = env.reset()
    
    glucose_log = [state.glucose]
    for step in range(env.episode_length):
        action = agent_fn(state)
        state, reward, done, info = env.step(action)
        glucose_log.append(state.glucose)
    
    # Calculate metrics
    metrics = grader._calculate_metrics(env)
    
    # Generate plots
    glucose_plot = create_glucose_plot(glucose_log)
    metrics_plot = create_metrics_plot(metrics)
    
    # Generate text summary
    scores = [r.score for r in results]
    pass_rate = sum(r.passed for r in results) / len(results)
    
    summary = f"""
# Simulation Results

## Configuration
- **Task**: {task_name}
- **Agent**: {agent_name}
- **Episodes**: {num_episodes}

## Performance Summary
- **Mean Score**: {np.mean(scores):.3f} ± {np.std(scores):.3f}
- **Best Score**: {np.max(scores):.3f}
- **Worst Score**: {np.min(scores):.3f}
- **Pass Rate**: {pass_rate:.1%} ({sum(r.passed for r in results)}/{num_episodes})
- **Pass Threshold**: {grader.pass_thresholds[task]:.2f}

## Key Metrics (Last Episode)
- **Time in Range**: {metrics['time_in_range']:.1%}
- **Time Above Range**: {metrics['time_above_range']:.1%}
- **Time Below Range**: {metrics['time_below_range']:.1%}
- **Mean Glucose**: {metrics['mean_glucose']:.1f} mg/dL
- **Glucose CV**: {metrics['cv_glucose']:.1f}%
- **Severe Hypo Events**: {int(metrics['severe_hypo_events'])}

## Feedback
{results[-1].feedback}
"""
    
    return summary, glucose_plot, metrics_plot


def create_demo_interface():
    """Create Gradio interface"""
    
    demo = gr.Blocks(title="Type 1 Diabetes Management Environment")
    
    with demo:
        gr.Markdown("""
        # 🩺 Type 1 Diabetes Management Environment
        
        **Meta PyTorch Hackathon - Round 1 Submission**
        
        This is an OpenEnv-compliant reinforcement learning environment for insulin dosing decisions in Type 1 Diabetes.
        
        ## How it works
        1. **Choose a task** - Easy (fasting), Medium (single meal), or Hard (full day)
        2. **Select an agent** - Different baseline strategies for insulin dosing
        3. **Run simulation** - The agent will control blood glucose for the specified duration
        4. **Analyze results** - View glucose timeline, metrics, and grading feedback
        
        ## Clinical Context
        - **Target Range**: 70-180 mg/dL (optimal glucose control)
        - **Hypoglycemia**: <70 mg/dL (dangerous, immediate treatment needed)
        - **Severe Hypoglycemia**: <54 mg/dL (critical emergency)
        - **Hyperglycemia**: >180 mg/dL (long-term complications)
        
        The goal is to maximize time in range while avoiding hypoglycemia events.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                task_dropdown = gr.Dropdown(
                    choices=[
                        "Easy: Fasting Control (8 hours)",
                        "Medium: Single Meal (6 hours)",
                        "Hard: Full Day (24 hours)"
                    ],
                    value="Easy: Fasting Control (8 hours)",
                    label="Select Task"
                )
                
                agent_dropdown = gr.Dropdown(
                    choices=[
                        "Basal Only (Baseline)",
                        "Fixed Bolus",
                        "Proportional Controller",
                        "PID Controller",
                        "Model Predictive",
                        "Adaptive (Recommended)"
                    ],
                    value="Adaptive (Recommended)",
                    label="Select Agent"
                )
                
                episodes_slider = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=3,
                    step=1,
                    label="Number of Episodes"
                )
                
                run_btn = gr.Button("🚀 Run Simulation", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                results_text = gr.Markdown(label="Results")
        
        with gr.Row():
            glucose_plot = gr.Image(label="Blood Glucose Timeline", type="pil")
            metrics_plot = gr.Image(label="Performance Metrics", type="pil")
        
        # Info sections
        with gr.Accordion("📚 About the Tasks", open=False):
            gr.Markdown("""
            ### Easy: Fasting Control (8 hours)
            - **Scenario**: Overnight glucose control with no meals
            - **Challenge**: Maintain stable basal insulin delivery
            - **Success**: 80% time in range, no severe hypoglycemia
            
            ### Medium: Single Meal Management (6 hours)
            - **Scenario**: Handle one meal with bolus insulin
            - **Challenge**: Carb counting and insulin timing
            - **Success**: 70% time in range, return to baseline
            
            ### Hard: Full Day Management (24 hours)
            - **Scenario**: 3 meals + exercise over full day
            - **Challenge**: Multiple confounding factors
            - **Success**: 70% time in range, <5% time low
            """)
        
        with gr.Accordion("🤖 About the Agents", open=False):
            gr.Markdown("""
            ### Baseline Agents
            
            - **Basal Only**: No bolus insulin, only continuous basal
            - **Fixed Bolus**: Fixed carb ratio (1:10) for meals
            - **Proportional Controller**: Error correction + IOB adjustment
            - **PID Controller**: Full PID control system
            - **Model Predictive**: Predictive control with lookahead
            - **Adaptive (Recommended)**: Time-aware strategy with best overall performance
            """)
        
        # Connect the button
        run_btn.click(
            fn=run_simulation,
            inputs=[task_dropdown, agent_dropdown, episodes_slider],
            outputs=[results_text, glucose_plot, metrics_plot]
        )
        
        gr.Markdown("""
        ---
        ### 🎯 Environment Details
        
        **Observation Space**: Glucose, insulin on board, carbs, time, meal announcements, exercise, history
        
        **Action Space**: Insulin bolus dose (0-20 units)
        
        **Reward**: Time in range + safety + glucose stability
        
        ### 🏥 Clinical Validation
        
        Based on ADA (American Diabetes Association) guidelines and international consensus on Time in Range metrics.
        
        **Impact**: 1.6M people with Type 1 Diabetes in the US
        """)
    
    return demo


if __name__ == "__main__":
    demo = create_demo_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft()
    )
