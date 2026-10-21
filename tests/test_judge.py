import json

import pytest
from judge import (
    JUDGE_PROMPT_TEMPLATE,
    build_claude_request,
    build_gemini_line,
    build_judge_prompt,
    build_openai_line,
    ensemble_rating,
    parse_claude_result,
    parse_gemini_line,
    parse_judge_text,
    parse_openai_line,
    score_from_rating,
)


def test_build_judge_prompt_fills_template():
    prompt = build_judge_prompt("Q?", "ref", "model", escape_backslashes=False)

    assert prompt == JUDGE_PROMPT_TEMPLATE.format(question="Q?", teacher_a="ref", model_a="model")


def test_build_judge_prompt_doubles_backslashes_only_when_asked():
    # Upstream's Claude judge escapes backslashes in both answers; the other judges don't.
    escaped = build_judge_prompt("Q \\x?", "\\frac{1}{2}", "\\(x\\)", escape_backslashes=True)
    raw = build_judge_prompt("Q \\x?", "\\frac{1}{2}", "\\(x\\)", escape_backslashes=False)

    assert "Answer 1 (Ground Truth): \\\\frac{1}{2}" in escaped
    assert "Answer 2 (Model Output): \\\\(x\\\\)" in escaped
    assert "Question: Q \\x?" in escaped
    assert "Answer 1 (Ground Truth): \\frac{1}{2}" in raw


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"rating": 4, "reason": "Same."}', (4, "Same.")),
        ('```json\n{"rating": 3, "reason": "Partial."}\n```', (3, "Partial.")),
        ('Here you go: {"rating": 2, "reason": "Wrong."} Thanks', (2, "Wrong.")),
        ('{"rating": "3", "reason": "String rating."}', (3, "String rating.")),
        # Invalid JSON escapes such as \a are doubled before parsing, as upstream does.
        ('{"rating": 4, "reason": "Uses \\alpha notation."}', (4, "Uses \\alpha notation.")),
    ],
)
def test_parse_judge_text_extracts_rating_and_reason(text, expected):
    assert parse_judge_text(text) == expected


@pytest.mark.parametrize("text", ["", "no json here", '{"rating": 7, "reason": "x"}', "{bad"])
def test_parse_judge_text_returns_minus_one_when_unusable(text):
    rating, _ = parse_judge_text(text)

    assert rating == -1


@pytest.mark.parametrize(
    ("ratings", "expected"),
    [
        ([4, 4, 2], 4),
        ([3, 2, 3], 3),
        ([2, 3, 4], 2),  # three-way tie goes to the lowest, as in the published ensemble
        ([4, -1, 3], 3),  # two-judge split also goes to the lowest
        ([-1, -1, 4], 4),
        ([-1, -1, -1], -1),
    ],
)
def test_ensemble_rating_matches_published_tie_breaking(ratings, expected):
    assert ensemble_rating(ratings) == expected


@pytest.mark.parametrize(("rating", "score"), [(1, 0), (2, 0), (3, 1), (4, 1), (-1, -1)])
def test_score_from_rating_binarizes(rating, score):
    assert score_from_rating(rating) == score


def test_build_openai_line_matches_upstream_gpt4o_request():
    line = build_openai_line("id1", "PROMPT", model="gpt-4o-2024-08-06")

    assert line == {
        "custom_id": "id1",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-2024-08-06",
            "messages": [{"role": "user", "content": "PROMPT"}],
            "response_format": {"type": "json_object"},
        },
    }


def test_build_claude_request_matches_upstream_claude_request():
    request = build_claude_request("id1", "PROMPT", model="claude-sonnet-4-5-20250929")

    assert request == {
        "custom_id": "id1",
        "params": {
            "model": "claude-sonnet-4-5-20250929",
            "max_tokens": 10000,
            "messages": [{"role": "user", "content": "PROMPT"}],
        },
    }


def test_build_gemini_line_matches_upstream_gemini_request():
    assert build_gemini_line("id1", "PROMPT") == {
        "key": "id1",
        "request": {"contents": [{"parts": [{"text": "PROMPT"}]}]},
    }


def _judgement(rating=4):
    return json.dumps({"rating": rating, "reason": "ok"})


def test_parse_openai_line_reads_message_content():
    line = {
        "custom_id": "id1",
        "response": {
            "status_code": 200,
            "body": {"choices": [{"message": {"content": _judgement(3)}}]},
        },
        "error": None,
    }

    assert parse_openai_line(line) == ("id1", 3, "ok")


def test_parse_openai_line_records_api_errors():
    line = {
        "custom_id": "id1",
        "response": {"status_code": 500, "body": {"error": {"message": "boom"}}},
        "error": None,
    }

    custom_id, rating, reason = parse_openai_line(line)

    assert (custom_id, rating) == ("id1", -1)
    assert "boom" in reason


def test_parse_gemini_line_reads_first_candidate_text():
    line = {
        "key": "id1",
        "response": {"candidates": [{"content": {"parts": [{"text": _judgement(2)}]}}]},
    }

    assert parse_gemini_line(line) == ("id1", 2, "ok")


def test_parse_gemini_line_records_errors_and_empty_candidates():
    error_line = {"key": "id1", "error": {"message": "quota"}}
    empty_line = {"key": "id2", "response": {"candidates": []}}

    assert parse_gemini_line(error_line)[:2] == ("id1", -1)
    assert parse_gemini_line(empty_line)[:2] == ("id2", -1)


def test_parse_claude_result_reads_text_block():
    result = {
        "custom_id": "id1",
        "result": {
            "type": "succeeded",
            "message": {"content": [{"type": "text", "text": _judgement(4)}]},
        },
    }

    assert parse_claude_result(result) == ("id1", 4, "ok")


def test_parse_claude_result_records_errored_and_expired():
    errored = {
        "custom_id": "id1",
        "result": {"type": "errored", "error": {"error": {"message": "overloaded"}}},
    }
    expired = {"custom_id": "id2", "result": {"type": "expired"}}

    assert parse_claude_result(errored)[:2] == ("id1", -1)
    assert parse_claude_result(expired)[:2] == ("id2", -1)
