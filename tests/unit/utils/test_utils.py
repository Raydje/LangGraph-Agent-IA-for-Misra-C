import json
import pytest
from app.utils import parse_json_response


def test_plain_json_string():
    result = parse_json_response('{"key": "value"}')
    assert result == {"key": "value"}


def test_json_with_json_fence():
    text = '```json\n{"key": "value"}\n```'
    assert parse_json_response(text) == {"key": "value"}


def test_json_with_plain_fence():
    text = '```\n{"key": "value"}\n```'
    assert parse_json_response(text) == {"key": "value"}


def test_leading_trailing_whitespace_stripped():
    text = '  \n  {"key": "value"}  \n  '
    assert parse_json_response(text) == {"key": "value"}


def test_invalid_json_raises_decode_error():
    with pytest.raises(json.JSONDecodeError):
        parse_json_response("not valid json")


def test_nested_types_parsed_correctly():
    text = '{"is_compliant": true, "cited_rules": ["Rule 1.1", "Rule 2.3"], "confidence_score": 0.95}'
    result = parse_json_response(text)
    assert result["is_compliant"] is True
    assert result["cited_rules"] == ["Rule 1.1", "Rule 2.3"]
    assert result["confidence_score"] == 0.95


def test_fenced_json_with_extra_blank_lines():
    text = '```json\n\n  {"a": 1}  \n\n```'
    assert parse_json_response(text) == {"a": 1}


def test_empty_json_object():
    assert parse_json_response("{}") == {}


def test_bool_false_value():
    result = parse_json_response('{"approved": false, "feedback": "fail"}')
    assert result["approved"] is False
