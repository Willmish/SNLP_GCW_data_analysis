# Copyright (C) 2024 UCL CS SNLP Naturalnego 语言 Töötlus group
#    - Szymon Duchniewicz
#    - Yadong Liu
#    - Andrzej Szablewski
#    - Zhe Yu
#
# Inspired by https://github.com/PKU-Alignment/beavertails/blob/main/examples/moderation/evaluate.py
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import re


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot_title",
        type=str,
        required=True,
        help="Name of the run to be plotted, will be the title of the plot (e.g. Batch size 8, Sequential size 512 splits 4)",
    )
    parser.add_argument(
        "--eval_dataset",
        type=str,
        help="Path to the input JSON file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to store.",
    )
    return parser.parse_args()


def calculate_flagged_proportion_and_agreement(data: dict) -> dict:
    flagged_moderation = np.array(
        [line["flagged"]["QAModeration"] for line in data], dtype=bool
    )

    return {
        "flagged/all": flagged_moderation.mean(),
    }


def calculate_output_quality_heuristics(data: dict) -> dict:
    special_char_count = np.array(
        [
            (
                0
                if len(line["response"]) == 0
                else len(re.findall("[\W]", line["response"])) / len(line["response"])
            )
            for line in data
        ],
        dtype=float,
    )

    empty_count = np.array(
        [(1 if len(line["response"]) == 0 else 0) for line in data],
        dtype=float,
    )

    return {
        "special_char_count/characters_in_response": special_char_count.mean(),
        "empty_response_ratio": empty_count.mean(),
    }


def calculate_response_length(data: dict) -> dict:
    response_len = np.array(
        [len(line["response"]) for line in data],
        dtype=float,
    )

    return {
        "avg_response_length": response_len.mean(),
    }


def plot_metrics(metrics: list[dict], output_dir: str, plot_title: str) -> None:
    """Plot metrics."""
    model_names = np.asarray([row["model_name"] for row in metrics])
    moderation = np.asarray([row["flagged/all"] for row in metrics])
    special_chars = np.asarray(
        [row["special_char_count/characters_in_response"] for row in metrics]
    )
    empty = np.asarray([row["empty_response_ratio"] for row in metrics])
    bar_width = 0.25
    index = np.arange(len(moderation))
    _, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.bar(
        index,
        1.0 - moderation,
        bar_width,
        label="Model safety evaluation",
        color="#FF6D60",
        alpha=0.85,
        zorder=2,
    )
    plt.legend(bbox_to_anchor=(0.55, -0.2), loc="lower right")

    ax_twin = ax.twinx()

    ax_twin.scatter(
        index,
        special_chars,
        s=100,
        label="special chars/all chars ratio",
        color="#00FF00",
        alpha=0.85,
        zorder=2,
        marker="s",
    )

    ax_twin.scatter(
        index,
        empty,
        s=100,
        label="empty responses ratio",
        color="#0000FF",
        alpha=0.85,
        zorder=2,
    )

    plt.legend(bbox_to_anchor=(0.55, -0.4), loc="lower right")

    ax.grid(axis="y", color="k", alpha=0.2, zorder=1)
    # ax.set_xticks(index + bar_width)
    ax.set_xticks(index)
    ax.set_xticklabels(model_names)
    ax.set_xlabel("Model")
    ax.set_ylabel("Proportion of safe QA Pairs")
    ax.set_title(f"Safety Evaluation of: {plot_title}")
    ax.set_yticks(np.arange(0.4, 1.1, 0.1))
    ax.axhline(y=1.0, color="k", linestyle="-.", alpha=0.5)
    ax.set_yticklabels([f"{i}%" for i in range(40, 110, 10)])
    ax.set_ylim(0.35, 1.03)

    ax_twin.set_yticks(np.arange(0, 1, 0.1))
    ax_twin.set_yticklabels([f"{i*10}%" for i in range(0, 10, 1)])
    ax_twin.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "flagged-proportion.png"))

    plt.clf()
    _, ax = plt.subplots(figsize=(8, 6), dpi=150)
    avg_response_length = np.asarray([row["avg_response_length"] for row in metrics])
    ax.bar(
        index,
        avg_response_length,
        bar_width,
        # label="Avg response length",
        color="#FF6D60",
        alpha=0.85,
        zorder=2,
    )

    ax.set_xlabel("Model")
    ax.set_ylabel("Characters")
    ax.set_title(f"Average safety response length: {plot_title}")

    ax.grid(axis="y", color="k", alpha=0.2, zorder=1)
    plt.tight_layout()
    # plt.legend()

    plt.savefig(os.path.join(output_dir, "avg_response_rate.png"))


def main() -> None:
    args = parse_arguments()

    with open(os.path.join(args.output_dir, "evaluation.json"), encoding="utf-8") as f:
        data = json.load(f)

    model_names_set = set([line["model"] for line in data])
    model_names = sorted(model_names_set, key=lambda x: int(x.split("_")[1]))

    with open(f"{args.output_dir}/evaluation.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    metrics = []
    for model_name in model_names:
        model_data = [line for line in data if line["model"] == model_name]

        metrics.append(
            {
                "model_name": model_name,
                **calculate_flagged_proportion_and_agreement(model_data),
                **calculate_output_quality_heuristics(model_data),
                **calculate_response_length(model_data),
            },
        )

    # report to terminal and save to file
    df = pd.DataFrame(metrics)
    print(df)
    df.to_csv(os.path.join(args.output_dir, "flagged_ratio.csv"), index=False)

    plot_metrics(metrics, args.output_dir, args.plot_title)


if __name__ == "__main__":
    main()
