# Copyright (C) 2024 UCL CS SNLP Naturalnego 语言 Töötlus group
#    - Szymon Duchniewicz
#    - Yadong Liu
#    - Andrzej Szablewski
#    - Zhe Yu
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import json
import argparse
import os

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Perform analysis of lm-eval-harness eval runs."
    )
    parser.add_argument(
        "--plot_title",
        type=str,
        required=True,
        help="Name of the run to be plotted, will be the title of the plot (e.g. Batch size 8, Sequential size 512 splits 4)",
    )
    parser.add_argument(
        "--eval_csv_framework",
        type=str,
        required=True,
        help="Path to the input CSV file of the framework evaluation.",
    )
    parser.add_argument(
        "--eval_csv_harmfulness",
        type=str,
        required=True,
        help="Path to the input CSV file of the harmfulness evaluation.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="Path to save the target plots and CSV files combined.",
    )
    args = parser.parse_args()
    return args


def fetch_log_data(log_dir: str) -> dict[int, dict]:
    log_file_names = os.listdir(log_dir)
    assert (
        log_file_names
    ), f"Beep boop, no files in a directory provided ({log_dir}). Maybe you forgot to copy them?"

    log_files_contents = {}
    for file_name in log_file_names:
        with open(os.path.join(log_dir, file_name), "r") as f:
            log_files_contents[int(file_name.split("_")[-1].split(".")[0])] = json.load(
                f
            )

    # return sorted dictionary
    return {
        k: v
        for k, v in sorted(
            log_files_contents.items(),
            key=lambda x: x[0],
        )
    }


def filter_json_logs(log_data: dict[int, dict]) -> dict[int, dict]:
    results = {}
    for model_iter_num, log in log_data.items():
        results[model_iter_num] = {
            "winogrande_acc": log["results"]["winogrande"]["acc,none"],
            "truthfulqa_mc2_acc": log["results"]["truthfulqa_mc2"]["acc,none"],
            "hellaswag_acc_norm": log["results"]["hellaswag"]["acc_norm,none"],
            #"gsm8k_exact_match_flexible": log["results"]["gsm8k"][
            #    "exact_match,flexible-extract"
            #],
            "arc_challenge_acc_norm": log["results"]["arc_challenge"]["acc_norm,none"],
            "mmlu_acc": log["results"]["mmlu"]["acc,none"],
            "toxigen_acc_norm": log["results"]["toxigen"]["acc_norm,none"],
        }

    return results


def create_plot(df: pd.DataFrame, log_dir: str, plot_title: str) -> None:
    ax = df.plot(style=["o-"] * 7)
    fig = ax.get_figure()
    ax.set_title(f"Task eval of: {plot_title}")
    ax.grid()
    # Limit y axis to the same size for all
    ax.set_ylim(0.2, 1.0)
    fig.savefig(os.path.join(log_dir, "figure.png"))


def main(eval_csv_framework: str, eval_csv_harmfulness: str, plot_title: str, log_dir: str):
    df_framework = pd.read_csv(os.path.join(eval_csv_framework, "results.csv"),index_col=False)
    df_harmfulness = pd.read_csv(os.path.join(eval_csv_harmfulness, "flagged_ratio.csv"), index_col=False)

    # XXX: At the moment, ratios in flagged_ratio csv are: flagged/all . We are intersted in the trend of safe_responses/all == 1 - flagged/all.
    df_harmfulness_transformed = 1 - df_harmfulness["flagged/all"]
    df_harmfulness = df_harmfulness_transformed.to_frame(name="safety_eval_beaverdam-7b")

    df = pd.concat([df_framework, df_harmfulness], axis=1)

    create_plot(df, log_dir, plot_title)
    df.to_csv(os.path.join(log_dir, "results.csv"), index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args.eval_csv_framework, args.eval_csv_harmfulness, args.plot_title, args.log_dir)
