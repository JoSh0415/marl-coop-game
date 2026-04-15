import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Function: Load best summary json
def load_best_summary(model, level):
    filepath = f"eval_results/ppo_{model}_level_{level}/eval_best_checkpoint_level_{level}.summary.json"

    if not os.path.exists(filepath):
        print(f"Warning: Missing data for {filepath}")
        return None

    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

# Function: Extract Agent 1's share of each action
def extract_a1_shares(summary):
    actions = [
        ("agent_1_ingredient_pickups", "agent_2_ingredient_pickups"),
        ("agent_1_valid_pot_adds", "agent_2_valid_pot_adds"),
        ("agent_1_bowl_pickups", "agent_2_bowl_pickups"),
        ("agent_1_done_soup_pickups", "agent_2_done_soup_pickups"),
        ("agent_1_serves", "agent_2_serves"),
    ]
    
    if summary is None:
        return [np.nan] * len(actions)
        
    events = summary["events"]
    a1_shares = []

    for a1_key, a2_key in actions:
        a1 = float(events[a1_key]["sum"])
        a2 = float(events[a2_key]["sum"])
        total = a1 + a2

        if total > 0:
            a1_share = (a1 / total) * 100.0
        else:
            a1_share = np.nan

        a1_shares.append(a1_share)

    return np.array(a1_shares)

if __name__ == "__main__":
    models = ["centralised", "decentralised", "decentralised_comms"]
    model_labels = ["Centralised PPO", "Decentralised PPO", "Decentralised PPO + Comms"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    levels = [1, 2, 3]
    level_titles = ["Level 1 (Bottleneck)", "Level 2 (Partition)", "Level 3 (Open Layout)"]

    action_labels = [
        "Ingredient Pickups",
        "Valid Pot Adds",
        "Bowl Pickups",
        "Done-Soup Pickups",
        "Serves",
    ]

    os.makedirs("analytics", exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 6.8), sharey=True)
    
    y_base = np.arange(len(action_labels))
    
    # Tighter offsets to clearly group algorithms under one metric
    y_offsets = [0.16, 0.0, -0.16]

    for level_idx, level in enumerate(levels):
        ax = axes[level_idx]

        # Light horizontal guides so each lollipop row maps clearly to an action
        for y in y_base:
            ax.axhline(y, color='#c5c5c5', linestyle='--', linewidth=0.85, alpha=0.7, zorder=0)

        ax.axvspan(40, 60, color='#e0e0e0', alpha=0.5, zorder=0, label="Near-Even Split" if level_idx == 0 else "")
        
        # Draw the exact 50/50 center line clearly
        ax.axvline(50, color='#777777', linestyle='--', linewidth=1.5, zorder=1)

        # Plot data
        for model_idx, model in enumerate(models):
            summary = load_best_summary(model, level)
            a1_shares = extract_a1_shares(summary)
            
            y_positions = y_base + y_offsets[model_idx]

            for y, share in zip(y_positions, a1_shares):
                if not np.isnan(share):
                    # Solid line from 50 to the dot
                    ax.plot([50, share], [y, y], color=colors[model_idx], linewidth=2.1, zorder=2)

            # Scatter dots with a white edge to make them pop cleanly
            ax.scatter(a1_shares, y_positions, color=colors[model_idx], s=115, 
                       edgecolor='white', linewidth=1.2, 
                       label=model_labels[model_idx] if level_idx == 0 else "", zorder=3)

        # Formatting
        ax.set_title(level_titles[level_idx], fontsize=15, pad=32)
        
        # X-axis formatting
        ax.set_xlim(-5, 105)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=11)
        ax.set_ylim(-0.55, len(action_labels) - 0.45)
        
        if level_idx == 1:
            ax.set_xlabel("Share of actions performed by Agent 1 (%)", fontsize=13, labelpad=15)
            
        ax.text(25, 1.02, '← Agent 2 dominates', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=10, color='#555555', style='italic')
        ax.text(75, 1.02, 'Agent 1 dominates →', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=10, color='#555555', style='italic')

        # Clean Y-axis
        ax.set_yticks(y_base)
        ax.set_yticklabels(action_labels, fontsize=12)
        ax.tick_params(axis='y', which='both', length=0) 
        
        # Invert Y axis so the first action is at the top
        if level_idx == 0:
            ax.invert_yaxis()
            
        # Clean up borders
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", linestyle=":", linewidth=0.9, alpha=0.45, zorder=0)

    # Shared legend
    fig.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=4, 
        frameon=True,
        fontsize=12,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.14, top=1.0, wspace=0.1)

    output_path = "analytics/role_split_lollipop.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved {output_path}")
