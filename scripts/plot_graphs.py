#!/usr/bin/env python3
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    scripts_dir = Path(__file__).resolve().parent
    base_dir = scripts_dir.parent
    
    csv_file = base_dir / "beam_search_results.csv"
    graphs_dir = base_dir / "graphs"

    if not csv_file.exists():
        print(f"Error: Could not find {csv_file}")
        print("Make sure you ran the evaluate_beam.sh script first!")
        return
    graphs_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(csv_file)
    sns.set_theme(style="whitegrid")


    plt.figure(figsize=(8, 5))
    ax1 = sns.lineplot(data=df, x="beam_size", y="bleu", marker="o", color="blue", linewidth=2)
    plt.title("Impact of Beam Size on BLEU Score (bpe_8k)", fontsize=14, pad=15)
    plt.xlabel("Beam Size (k)", fontsize=12)
    plt.ylabel("BLEU Score", fontsize=12)
    plt.xticks(df["beam_size"]) # Ensure all beam sizes show on the x-axis
    
    bleu_graph_path = graphs_dir / "beam_vs_bleu.png"
    plt.savefig(bleu_graph_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {bleu_graph_path}")
    
    plt.figure(figsize=(8, 5))
    ax2 = sns.lineplot(data=df, x="beam_size", y="time_s", marker="s", color="orange", linewidth=2)
    plt.title("Impact of Beam Size on Translation Time", fontsize=14, pad=15)
    plt.xlabel("Beam Size (k)", fontsize=12)
    plt.ylabel("Time Taken (seconds)", fontsize=12)
    plt.xticks(df["beam_size"])
    
    # Save Graph 2
    time_graph_path = graphs_dir / "beam_vs_time.png"
    plt.savefig(time_graph_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {time_graph_path}")

if __name__ == "__main__":
    main()