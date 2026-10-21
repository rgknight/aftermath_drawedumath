import pandas as pd
from batch_generate import (
    GENERATE_ANSWER_PROMPT,
    build_request,
    make_custom_id,
    parse_result_line,
    select_sample_rows,
)


def _predictions() -> pd.DataFrame:
    rows = [
        # (model, problem, image, qa type, question, reference)
        ("GPT-5", 1, "a.jpeg", "teacher", " Did they label it?", "Yes."),
        ("GPT-5", 1, "a.jpeg", "claude", "What color?", "Blue."),
        ("GPT-5", 1, "b.png", "teacher", " Did they label it?", "No."),
        ("GPT-5", 2, "c.jpeg", "gpt4o", "How many?", "3"),
        ("GPT-4.1", 1, "a.jpeg", "teacher", " Did they label it?", "Yes."),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "Model Name",
            "Problem ID",
            "Image Name",
            "QA Type",
            "Question",
            "Reference Answer",
        ],
    )


def test_select_sample_rows_matches_on_stem_and_uses_one_model():
    # The sample file lists .png names for images stored as .jpeg in predictions.csv.
    sampled = ["a.png", "b.png"]

    rows = select_sample_rows(_predictions(), sampled)

    assert sorted(rows["Image Name"]) == ["a.jpeg", "a.jpeg", "b.png"]
    assert "Model Name" not in rows.columns


def test_make_custom_id_is_stable_and_distinguishes_rows():
    rows = select_sample_rows(_predictions(), ["a", "b"])
    ids = [make_custom_id(row) for _, row in rows.iterrows()]

    assert len(set(ids)) == len(ids)
    assert ids == [make_custom_id(row) for _, row in rows.iterrows()]
    assert all(len(custom_id) <= 64 for custom_id in ids)


def test_build_request_mirrors_upstream_prompt_with_no_reasoning():
    row = pd.Series({"Question": " Did they label it?", "Image Name": "a.jpeg"})

    request = build_request(
        row,
        custom_id="abc",
        image_url="https://example.com/a.jpeg",
        model="gpt-6-luna",
        reasoning_effort="none",
        max_output_tokens=4096,
    )

    assert request["custom_id"] == "abc"
    assert request["method"] == "POST"
    assert request["url"] == "/v1/responses"
    body = request["body"]
    assert body["model"] == "gpt-6-luna"
    assert body["instructions"] == GENERATE_ANSWER_PROMPT
    assert body["reasoning"] == {"effort": "none"}
    assert body["max_output_tokens"] == 4096
    assert body["store"] is False
    (message,) = body["input"]
    assert message["role"] == "user"
    text, image = message["content"]
    # Upstream concatenates the raw (unstripped) question.
    assert text == {
        "type": "input_text",
        "text": "Answer the following question:  Did they label it?",
    }
    assert image == {"type": "input_image", "image_url": "https://example.com/a.jpeg"}


def _response_line(status="completed", content=None, incomplete_reason=None):
    if content is None:
        content = [{"type": "output_text", "text": "Yes, it is labeled."}]
    return {
        "id": "batch_req_1",
        "custom_id": "abc",
        "response": {
            "status_code": 200,
            "body": {
                "status": status,
                "incomplete_details": (
                    {"reason": incomplete_reason} if incomplete_reason else None
                ),
                "output": [
                    {"type": "reasoning", "summary": []},
                    {"type": "message", "role": "assistant", "content": content},
                ],
                "usage": {
                    "input_tokens": 1400,
                    "output_tokens": 12,
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            },
        },
        "error": None,
    }


def test_parse_result_line_extracts_answer_text_and_usage():
    result = parse_result_line(_response_line())

    assert result == {
        "custom_id": "abc",
        "Model Answer": "Yes, it is labeled.",
        "Response Status": "completed",
        "Incomplete Reason": "",
        "Error": "",
        "Input Tokens": 1400,
        "Output Tokens": 12,
        "Reasoning Tokens": 0,
    }


def test_parse_result_line_keeps_partial_text_from_incomplete_response():
    result = parse_result_line(
        _response_line(status="incomplete", incomplete_reason="max_output_tokens")
    )

    assert result["Response Status"] == "incomplete"
    assert result["Incomplete Reason"] == "max_output_tokens"
    assert result["Model Answer"] == "Yes, it is labeled."


def test_parse_result_line_records_refusal_text():
    result = parse_result_line(
        _response_line(content=[{"type": "refusal", "refusal": "I can't help with that."}])
    )

    assert result["Model Answer"] == "I can't help with that."
    assert result["Response Status"] == "refused"


def test_parse_result_line_records_request_errors():
    line = {
        "id": "batch_req_2",
        "custom_id": "xyz",
        "response": {
            "status_code": 400,
            "body": {"error": {"message": "Invalid image URL", "type": "invalid_request_error"}},
        },
        "error": None,
    }

    result = parse_result_line(line)

    assert result["custom_id"] == "xyz"
    assert result["Model Answer"] == ""
    assert result["Response Status"] == "error"
    assert result["Error"] == "Invalid image URL"
