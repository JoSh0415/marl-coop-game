import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Function: Extract the three main behavioural rates per 100 steps
def get_behaviour_rates(model, level):
    filepath = f"eval_results/ppo_{model}_level_{level}/eval_ppo_{model}_level_{level}_best.summary.json"
    
    if not os.path.exists(filepath):
        print(f"Warning: Missing data for {filepath}")
        return [0.0, 0.0, 0.0]
        
    with open(filepath, "r") as f:
        data = json.load(f)
        
    steps_sum = float(data["steps"]["sum"])
    if steps_sum == 0:
        return [0.0, 0.0, 0.0]

    events = data["events"]
    
    # Calculate rates per 100 steps
    collision_rate = (float(events["collision_attempts"]["sum"]) / steps_sum) * 100
    wrong_pot_rate = (float(events["wrong_pot_adds"]["sum"]) / steps_sum) * 100
    both_idle_rate = (float(events["both_idle_steps"]["sum"]) / steps_sum) * 100
    
    return [collision_rate, wrong_pot_rate, both_idle_rate]

if __name__ == "__main__":
    models = ["centralised", "decentralised", "decentralised_comms"]
    model_labels = ["Centralised PPO", "Decentralised PPO", "Decentralised PPO + Comms"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    levels = [1, 2, 3]
    level_labels = ["Level 1\n(Bottleneck)", "Level 2\n(Partition)", "Level 3\n(Open Layout)"]
    
    metrics = ["Collision Attempts", "Wrong Pot Adds", "Both-Idle Steps"]

    # Collect data: data[metric_idx][level_idx][model_idx]
    data = np.zeros((len(metrics), len(levels), len(models)))
    
    for i, level in enumerate(levels):
        for j, model in enumerate(models):
            rates = get_behaviour_rates(model, level)
            for m in range(len(metrics)):
                data[m, i, j] = rates[m]

    # Create the figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8))
    bar_width = 0.24
    x = np.arange(len(levels))

    for m, ax in enumerate(axes):
        # Plot bars for each model
        for j, model in enumerate(models):
            x_pos = x + (j - 1) * bar_width 
            values = data[m, :, j]
            
            bars = ax.bar(x_pos, values, width=bar_width, label=model_labels[j], color=colors[j], zorder=3)
            
            # Annotate bars
            ymax = ax.get_ylim()[1]
            offset = ymax * 0.005
            for bar, value in zip(bars, values):
                # Skip the label if the value is 0
                if value > 0.005:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + offset,
                        f"{value:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                    )

        ax.set_title(f"{metrics[m]}", fontsize=15, pad=12)
        
        # Format X-axis
        ax.set_xticks(x)
        ax.set_xticklabels(level_labels, fontsize=11)
        
        # Format Y-axis
        ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
        ax.set_axisbelow(True)
        
        if m == 0:
            ax.set_ylabel("Events per 100 steps", fontsize=12)

    # Single shared legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02), fontsize=12, frameon=True)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.16, top=0.94)
    
    os.makedirs("analytics", exist_ok=True)
    plt.savefig("analytics/behaviour_rates_barchart.png", dpi=300, bbox_inches='tight')
    print("Saved analytics/behaviour_rates_barchart.png")