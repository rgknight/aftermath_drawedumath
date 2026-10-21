import pandas as pd
from build_subset import collect_answers, select_hard_teacher_questions


def _questions() -> pd.DataFrame:
    rows = [
        # (QA Type, Question Cat, Models correct, Models scored)
        ("teacher", "Counting content", 2, 11),  # keep
        ("teacher", "Counting content", 5, 11),  # keep: 5 of 11 is not a majority
        ("teacher", "Counting content", 6, 11),  # drop: most models correct
        ("claude", "Counting content", 0, 11),  # drop: AI-generated
        ("gpt4o", "Writing and labels", 0, 11),  # drop: AI-generated
        ("teacher", "Image creation and medium", 0, 11),  # drop: image/paper question
        ("teacher", "Writing and labels", 1, 10),  # drop: a model has no usable score
    ]
    return pd.DataFrame(
        rows, columns=["QA Type", "Question Cat", "Models correct", "Models scored"]
    )


def test_keeps_only_hard_teacher_questions_scored_by_every_model():
    subset = select_hard_teacher_questions(_questions(), model_count=11)

    assert subset["Models correct"].tolist() == [2, 5]
    assert set(subset["QA Type"]) == {"teacher"}


def test_collect_answers_groups_by_problem_in_model_order_with_luna_judges():
    key = {"Image Name": "a.png", "QA Type": "teacher", "Reference Answer": "Ref"}
    subset = pd.DataFrame(
        [
            {"Problem ID": 7, "Question": "Q1", **key},
            {"Problem ID": 7, "Question": "Q2", **key},
        ]
    )
    predictions = pd.DataFrame(
        [
            {
                "Problem ID": 7,
                "Question": "Q1",
                **key,
                "Model Name": "B",
                "Model Answer": "b1",
                "Ensemble_Judge_Rating": 2,
            },
            {
                "Problem ID": 7,
                "Question": "Q1",
                **key,
                "Model Name": "A",
                "Model Answer": "a1",
                "Ensemble_Judge_Rating": 4,
            },
            {
                "Problem ID": 7,
                "Question": "Q2",
                **key,
                "Model Name": "A",
                "Model Answer": "a2",
                "Ensemble_Judge_Rating": 1,
            },
        ]
    )
    luna = pd.DataFrame(
        [
            {
                "Problem ID": 7,
                "Question": "Q2",
                **key,
                "Model Answer": "luna2",
                "Ensemble_Judge_Rating": 1,
                "Claude_Judge_Rating": 2,
                "Gemini_Judge_Rating": 1,
                "Openai_Judge_Rating": 1,
                "Claude_Judge_Reason": "c",
                "Gemini_Judge_Reason": "g",
                "Openai_Judge_Reason": "o",
            }
        ]
    )

    files = collect_answers(subset, predictions, luna, model_order=["A", "B"])

    assert list(files) == [7]
    assert files[7]["models"] == {"0": [["a1", 4], ["b1", 2]], "1": [["a2", 1], ["", -1]]}
    assert files[7]["luna"] == {"1": ["luna2", 1, [2, 1, 1], ["c", "g", "o"]]}
