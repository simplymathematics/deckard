import argparse
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from deckard.layers.plots import set_matplotlib_vars

set_matplotlib_vars()


sns.set_theme(style="whitegrid", font_scale=2.5, font="times new roman")


def plot_trash_score(input_file: str, output_file: str, title:str):
    big_df = pd.read_csv(input_file, index_col=0, low_memory=False)
    # Capitalize all letters in the dataset
    big_df["dataset"] = big_df["dataset"].str.upper()
    # Split the device_id on the last word and only keep the last word
    big_df["Device"] = big_df["device_id"].str.split("-").str[-1].str.upper()
    # Accuracy Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    c_bar = sns.boxenplot(
        data=big_df,
        x="dataset",
        y="c_adv",
        hue="Device",
        ax=ax,
    )
    title = title.replace("_", "  ").title()
    title = title.replace("  ", "-")
    c_bar.set_title(f"{title}")
    c_bar.set_ylabel("TRASH Score")
    c_bar.set_xlabel("")
    c_bar.tick_params(axis="x", rotation=0)
    c_bar.set_yscale("log")
    c_bar.legend(
        title="Device",
        loc="upper left",
        bbox_to_anchor=(1, 1),
    )
    # Add red line a y=1
    c_bar.axhline(1, color="red", linestyle="solid")
    # Save the plot
    fig.tight_layout()
    fig.savefig(output_file)



parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--title", type=str, required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    plot_trash_score(**vars(args))

