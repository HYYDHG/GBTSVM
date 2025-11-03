"""Command-line interface for running GBTSVM experiments.

This script mirrors the behaviour of the original repository while adding a
simple internationalisation layer so that the console output can be displayed
in either English or Chinese.  The computational pipeline remains unchanged:

* load every CSV file from the ``Data`` directory,
* split the dataset into training and test partitions,
* apply the feature-attention preprocessing,
* generate granular balls and run the GBTSVM classifier.

The only new behaviour is language-aware status messaging and basic
error-reporting when the expected data directory is missing or empty.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

from GBTSVM import GBTSVM
from attention import compute_feature_attention
from gen_ball import gen_balls


SUPPORTED_LANGUAGES = ("en", "zh")


MESSAGES = {
    "missing_data_dir": {
        "en": "Dataset directory '{path}' not found. Please add CSV files before running the experiment.",
        "zh": "未找到数据集目录“{path}”。运行实验前请先添加 CSV 文件。",
    },
    "no_csv_files": {
        "en": "No CSV files were found in '{path}'.",
        "zh": "在“{path}”中未找到任何 CSV 文件。",
    },
    "processing_dataset": {
        "en": "Processing dataset: {name}",
        "zh": "正在处理数据集：{name}",
    },
    "test_summary": {
        "en": "Test accuracy: {accuracy} | Test time: {time}",
        "zh": "测试准确率：{accuracy} | 测试耗时：{time}",
    },
}


def get_message(key: str, language: str, **params: object) -> str:
    """Return a translated status message."""

    template = MESSAGES[key]
    text = template.get(language, template["en"])
    return text.format(**params)


def load_datasets(data_dir: Path) -> Iterable[Tuple[str, np.ndarray]]:
    """Yield dataset name and content for every CSV file in ``data_dir``."""

    for file_path in sorted(data_dir.glob("*.csv")):
        yield file_path.name, np.loadtxt(file_path, delimiter=",", dtype=float)


def prepare_dataset(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split the dataset, apply attention weights and return train/test arrays."""

    m, n = data.shape

    labels = data[:, n - 1]
    labels[labels == 0] = -1
    data[:, n - 1] = labels

    rng = np.random.default_rng(seed=0)
    indices = rng.permutation(m)
    shuffled = data[indices]

    split_point = int(m * (1 - 0.30))
    train = shuffled[:split_point].copy()
    test = shuffled[split_point:].copy()

    attention_weights = compute_feature_attention(train)
    train[:, :-1] *= attention_weights
    test[:, :-1] *= attention_weights

    return train, test


def build_training_balls(train: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Generate the granular balls and reshape them for GBTSVM input."""

    pur = 1 - (0.015 * 5)
    num = 4
    c1 = 0.00001
    c2 = 0.00001

    balls = gen_balls(train, pur=pur, delbals=num)

    radius = [item[1] for item in balls]
    center = [item[0] for item in balls]
    label = [item[-1] for item in balls]

    radius_array = np.asarray(radius)
    center_array = np.asarray(center)
    label_array = np.asarray(label)

    z_train = np.hstack((center_array, radius_array.reshape(radius_array.shape[0], 1)))
    lab = label_array.reshape(label_array.shape[0], 1)
    return np.hstack((z_train, lab)), c1, c2


def run_pipeline(data_dir: Path, language: str) -> int:
    """Execute the GBTSVM pipeline and print translated status messages."""

    if not data_dir.exists():
        print(get_message("missing_data_dir", language, path=data_dir))
        return 1

    datasets = list(load_datasets(data_dir))
    if not datasets:
        print(get_message("no_csv_files", language, path=data_dir))
        return 1

    for name, data in datasets:
        print(get_message("processing_dataset", language, name=name))
        train, test = prepare_dataset(data)
        train_balls, c1, c2 = build_training_balls(train)
        test_accuracy, test_time = GBTSVM(train_balls, test, c1, c2)
        print(
            get_message(
                "test_summary",
                language,
                accuracy=f"{test_accuracy:.4f}",
                time=f"{test_time:.4f}",
            )
        )

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run GBTSVM experiments with optional Chinese output."
    )
    parser.add_argument(
        "--language",
        choices=SUPPORTED_LANGUAGES,
        default="en",
        help="Language for console messages (default: en).",
    )
    args = parser.parse_args()

    language = args.language
    data_dir = Path(__file__).resolve().parent / "Data"
    return run_pipeline(data_dir, language)


if __name__ == "__main__":
    raise SystemExit(main())
