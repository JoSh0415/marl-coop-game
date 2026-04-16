import os
import json
import matplotlib.pyplot as plt

# Function: Load metric data from JSON sweeps
def load_sweep_data(model, level, steps, metric):
    values = []
    for step in steps:
        filepath = f"eval_sweeps/ppo_{model}_level_{level}/eval_checkpoint_{step}_level_{level}.summary.json"
        
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                data = json.load(f)
                values.append(data.get(metric, 0.0))
        else:
            print(f"Warning: Missing data for {filepath}")
            values.append(0.0)
            
    return values

if __name__ == "__main__":
    models = ["centralised", "decentralised", "decentralised_comms"]
    model_labels = ["Centralised PPO", "Decentralised PPO", "Decentralised PPO + Task-State Cue"]
    levels = [1, 2, 3]
    level_names = ["(Bottleneck)", "(Partition)", "(Open Layout)"]
    
    # Checkpoints every 500k up to 10M
    steps = list(range(500000, 10500000, 500000))
    x_vals = [s / 1000000.0 for s in steps]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)

    for i, level in enumerate(levels):
        ax = axes[i]
        
        for j, model in enumerate(models):
            y_values = load_sweep_data(model, level, steps, "perfect_rate")
            ax.plot(x_vals, y_values, marker='o', linewidth=2, markersize=6, label=model_labels[j])

        ax.set_title(f"Level {level} {level_names[i]}", fontsize=14, pad=10)
        ax.set_xlabel("Training Steps (Millions)", fontsize=12)
        
        # Only set the Y-label on the first graph
        if i == 0:
            ax.set_ylabel("Perfect Episode Rate", fontsize=12)

        # Add a buffer below 0.0 and above 1.0 so points don't sit on the border
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle='--', alpha=0.6)

    # Create a single shared legend at the bottom of the entire figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.0), fontsize=12, frameon=True)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)

    plt.savefig("analytics/perfect_rate_over_checkpoints.png", dpi=300, bbox_inches='tight')
    print("Saved perfect_rate_over_checkpoints.png")