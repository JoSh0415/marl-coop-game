import os
import pandas as pd

# Function: Calculate the % of episodes containing at least one occurrence of the metric
def get_sparse_error_percentage(model, level, metric):
    filepath = f"eval_results/ppo_{model}_level_{level}/eval_ppo_{model}_level_{level}_best.csv"
    
    if not os.path.exists(filepath):
        print(f"Warning: Missing data for {filepath}")
        return 0.0
        
    try:
        df = pd.read_csv(filepath)
        if metric not in df.columns:
            print(f"Warning: Missing column '{metric}' in {filepath}")
            return 0.0
            
        # Calculate % of episodes where this error happened > 0 times
        pct = (df[metric] > 0).mean() * 100.0
        return float(pct)
        
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return 0.0

if __name__ == "__main__":
    models = ["centralised", "decentralised", "decentralised_comms"]
    model_labels = ["Centralised PPO", "Decentralised PPO", "Decentralised PPO + Task-State Cue"]
    
    levels = [1, 2, 3]
    level_names = ["Level 1 (Bottleneck)", "Level 2 (Partition)", "Level 3 (Open Layout)"]
    
    metrics = {
        "wrong_serve_attempts": "Wrong Serve",
        "wrong_done_soup_pickups": "Wrong Done-Soup Pickup",
        "burnt_soup_pickups": "Burnt Soup Pickup",
    }
    
    rows = []
    
    # Collect data for the table
    for i, level in enumerate(levels):
        for j, model in enumerate(models):
            row = {
                "Level": level_names[i],
                "Algorithm": model_labels[j],
            }
            
            for raw_metric, pretty_name in metrics.items():
                val = get_sparse_error_percentage(model, level, raw_metric)
                row[pretty_name] = val
                    
            rows.append(row)

    # Create the base DataFrame for the CSV (pure numbers)
    table_df = pd.DataFrame(rows)
    
    # Create a formatted copy for Markdown display (adds the % symbol)
    formatted_df = table_df.copy()
    for pretty_name in metrics.values():
        formatted_df[pretty_name] = formatted_df[pretty_name].apply(lambda x: f"{x:.1f}%")

    os.makedirs("analytics", exist_ok=True)
    
    # Save the raw numbers to CSV
    csv_path = os.path.join("analytics", "sparse_error_incidence_table.csv")
    table_df.to_csv(csv_path, index=False)
    
    # Save the formatted table to Markdown
    md_path = os.path.join("analytics", "sparse_error_incidence_table.md")
    markdown_text = formatted_df.to_markdown(index=False)
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown_text)

    # Print to console
    print("\n=== SPARSE ERROR INCIDENCE TABLE (Best Checkpoints) ===\n")
    print(markdown_text)
    print(f"\nSaved {csv_path}")
    print(f"Saved {md_path}")