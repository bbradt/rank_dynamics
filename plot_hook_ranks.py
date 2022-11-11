import os
import glob
import torch
import tqdm
import pickle
import argparse
import pandas as pd
import pylab
import matplotlib.pyplot as plt
import seaborn as sb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plots", default="all")
    parser.add_argument("--output_root", default="results")
    parser.add_argument("--experiment_name", default="test")
    parser.add_argument("--output_folder", default="plots")
    parser.add_argument("--stats_folder", default="stats")
    parser.add_argument("--epochs", default=10)
    args = parser.parse_args()

    outputs_path = os.path.join(
        args.output_root, args.experiment_name, args.output_folder)
    os.makedirs(outputs_path, exist_ok=True)

    stats_path = os.path.join(
        args.output_root, args.experiment_name, args.stats_folder
    )

    hook_rank_stats = os.path.join(stats_path, "hook_stats.csv")
    
    df = pd.read_csv(hook_rank_stats)
    modules = df['module'].unique()
    print("Starting")
    sb.set(font_scale=2)
    fig, ax = plt.subplots(len(modules), 3, figsize=(3*10, len(modules)*10))
    for i, module in enumerate(modules):
        grad_ax = ax[i, 0]
        act_ax = ax[i, 1]
        delta_ax = ax[i, 2]
        mod_df = df[df['module'] == module]
        sb.lineplot(x="epoch", y="act_rank", hue="group", data=mod_df, ax=act_ax, palette=sb.color_palette("tab10"))
        sb.lineplot(x="epoch", y="grad_rank", hue="group", data=mod_df, ax=grad_ax, palette=sb.color_palette("tab10"))
        sb.lineplot(x="epoch", y="delta_rank", hue="group", data=mod_df, ax=delta_ax, palette=sb.color_palette("tab10"))
        grad_ax.set_ylabel("Average Rank")
        grad_ax.get_legend().remove()
        act_ax.set_ylabel("Average Rank")
        act_ax.get_legend().remove()
        delta_ax.set_ylabel("Average Rank")
        delta_ax.get_legend().remove()
        grad_ax.set_title(module)        
        print("\t Done %s" % module)
    plt.savefig(os.path.join(outputs_path, "rank_dynamics.png"), bbox_inches="tight")
    ax = pylab.gca()
    figLegend = pylab.figure(figsize = (1.5,1.3))

    # produce a legend for the objects in the other figure
    pylab.figlegend(*ax.get_legend_handles_labels(), loc = 'upper left')

    # save the two figures to files
    figLegend.savefig(os.path.join(outputs_path, "rank_dynamics_legend.png"))
    print("Done")