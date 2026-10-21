"""Do the models get the same questions wrong?

Lines up every model's 0/1 score on each question in the Aftermath matched sample and asks
whether misses cluster on the same questions more than they would if each model erred
independently at its own overall error rate.

Usage (from the repo root):
    uv run python code/error_overlap.py --run luna6_none
"""

import argparse

import numpy as np
import pandas as pd
from compare_runs import OUTPUT_DIR, load_comparison

QUESTION_KEY = ["Image Name", "QA Type", "Question", "Reference Answer"]


def expected_wrong_count_distribution(error_rates: list[float]) -> np.ndarray:
    """P(exactly k models wrong) if each model misses independently at its own rate."""
    distribution = np.array([1.0])
    for rate in error_rates:
        distribution = np.convolve(distribution, [1 - rate, rate])
    return distribution


def build_wrong_matrix(combined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Questions x models matrix of 1 = wrong, plus per-question attributes."""
    scores = combined.pivot_table(index=QUESTION_KEY, columns="Model Name", values="Score")
    scored = scores.notna().all(axis=1) & (scores != -1).all(axis=1)
    wrong = (1 - scores[scored]).astype(int)
    attributes = combined.drop_duplicates(QUESTION_KEY).set_index(QUESTION_KEY)
    attributes = attributes.loc[wrong.index, ["Problem ID", "Has Error", "Question Cat"]]
    return wrong, attributes


def _share_table(attributes: pd.DataFrame, mask: pd.Series, column: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "all questions": attributes[column].value_counts(normalize=True),
            "selected": attributes.loc[mask, column].value_counts(normalize=True),
        }
    ).fillna(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="append", required=True, help="Run name(s) in output/")
    args = parser.parse_args()

    combined = load_comparison(args.run, drop_image_creation=False)
    wrong, attributes = build_wrong_matrix(combined)
    models = list(wrong.columns)
    wrong_count = wrong.sum(axis=1)
    print(f"{len(wrong)} questions scored by all {len(models)} models\n")

    observed = wrong_count.value_counts().reindex(range(len(models) + 1), fill_value=0)
    expected = expected_wrong_count_distribution(wrong.mean().tolist()) * len(wrong)
    bands = {"0": [0], "1-3": [1, 2, 3], "4-8": list(range(4, 9)), "9-11": [9, 10, 11]}
    bands[str(len(models))] = [len(models)]
    print("Questions by number of models that got them wrong (observed vs. if independent):")
    for label, counts in bands.items():
        print(f"  {label:>5}: {observed[counts].sum():6d}  vs  {expected[counts].sum():8.1f}")

    hard = wrong_count >= 9
    errors_on_hard = wrong[hard].values.sum() / wrong.values.sum()
    print(f"\nShare of all misses that fall on questions >=9 models miss: {errors_on_hard:.1%}")

    correlation = wrong.corr()
    off_diagonal = correlation.where(~np.eye(len(models), dtype=bool))
    print(f"Mean pairwise correlation of misses (phi): {np.nanmean(off_diagonal.values):.2f}")
    print("\nMost similar miss patterns:")
    pairs = off_diagonal.stack().sort_values(ascending=False)
    for (first, second), value in pairs.iloc[::2].head(5).items():
        print(f"  {first} ~ {second}: {value:.2f}")

    run_models = {
        name
        for run in args.run
        for name in pd.read_csv(OUTPUT_DIR / run / "judged.csv", usecols=["Model Name"])[
            "Model Name"
        ].unique()
    }
    for model in sorted(run_models):
        others = wrong.drop(columns=model)
        unique = (wrong[model] == 1) & (others.sum(axis=1) <= 2)
        print(f"\n{model}: closest miss patterns:")
        print(off_diagonal[model].dropna().sort_values(ascending=False).head(3).round(2))
        print(
            f"{model} misses {int(wrong[model].sum())} questions; "
            f"{int(unique.sum())} of them are missed by at most 2 other models."
        )
        print(_share_table(attributes, unique, "Has Error").round(2).to_string())
        print(_share_table(attributes, unique, "Question Cat").round(2).head(8).to_string())

    print("\nQuestions >=11 of 12 models miss, compared with all questions:")
    very_hard = wrong_count >= len(models) - 1
    print(f"  count: {int(very_hard.sum())}")
    print(_share_table(attributes, very_hard, "Has Error").round(2).to_string())
    print(_share_table(attributes, very_hard, "Question Cat").round(2).to_string())

    summary = attributes.assign(**{"Models wrong": wrong_count})
    summary.join(wrong.add_prefix("Wrong: ")).to_csv(OUTPUT_DIR / args.run[0] / "error_overlap.csv")


if __name__ == "__main__":
    main()
