import json

import pandas as pd

from review_app.data import load_subset_items


def test_load_subset_items_attaches_model_answers_and_luna(tmp_path):
    pd.DataFrame(
        [
            {
                "Problem ID": 7,
                "Image Name": "a.png",
                "Image URL": "https://example.com/a.png",
                "Question": " How many? ",
                "Reference Answer": "Three ",
                "Question Cat": "Counting content",
                "Models correct": 1,
            }
        ]
    ).to_csv(tmp_path / "hard_teacher_questions.csv", index=False)
    (tmp_path / "explorer_data.json").write_text(
        json.dumps({"models": ["Model A", "Model B"], "extraModel": "GPT-6 Luna"})
    )
    (tmp_path / "answers").mkdir()
    (tmp_path / "answers" / "7.json").write_text(
        json.dumps(
            {
                "models": {"0": [["3", 4], ["two", 2]]},
                "luna": {"0": ["three", 3, [3, 4, 3], ["r1", "r2", "r3"]]},
            }
        )
    )

    items = load_subset_items(tmp_path)

    assert list(items) == ["q00000"]
    item = items["q00000"]
    assert (item["question"], item["answer"], item["status"], item["order"]) == (
        "How many?",
        "Three",
        "pending",
        0,
    )
    assert item["models"] == [
        {"name": "Model A", "answer": "3", "rating": 4},
        {"name": "Model B", "answer": "two", "rating": 2},
        {"name": "GPT-6 Luna", "answer": "three", "rating": 3},
    ]
