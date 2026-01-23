import os
import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
CSV_DIR = os.path.join("dataset", "csv_files")
MASTER_CSV = os.path.join(CSV_DIR, "master_data.csv")

def plot_distribution():
    if not os.path.exists(MASTER_CSV):
        print("Error: Master CSV not found!")
        return

    # Load Data
    df = pd.read_csv(MASTER_CSV)
    
    # Check total
    total_count = len(df)
    print(f"--- Dataset Audit ---")
    print(f"Total Artifacts: {total_count}")
    
    # count by Classification
    class_counts = df['Classification'].value_counts()
    print("\nClass Breakdown:")
    print(class_counts)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    bars = class_counts.plot(kind='bar', color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    
    plt.title(f"Artifact Distribution (Total: {total_count})", fontsize=16)
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add numbers on top of bars
    for i, v in enumerate(class_counts):
        plt.text(i, v + 10, str(v), ha='center', fontweight='bold')
        
    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join("analysis", "class_distribution.png")
    plt.savefig(save_path)
    print(f"\nGraph saved to: {save_path}")
    print("Check the 'analysis' folder to see the image!")

if __name__ == "__main__":
    plot_distribution()