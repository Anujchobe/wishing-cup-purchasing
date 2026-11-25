from decimal import Decimal

import pytest

from app.llm.parsing import describe_error, extract_json


@pytest.mark.parametrize("reply", [
    '{"a": 1}',
    'Sure! Here it is:\n{"a": 1}\nHope that helps.',
    '```json\n{"a": 1}\n```',
    '<think>let me think {not json}</think>{"a": 1}',
    '```\n{"a": 1}\n```',
])
def test_finds_the_json(reply):
    assert extract_json(reply) == {"a": 1}


def test_braces_inside_strings_are_handled():
    assert extract_json('ok {"note": "use } and { carefully", "n": 2} done') == {"note": "use } and { carefully", "n": 2}


def test_decimals_never_become_floats():
    value = extract_json('{"size": 0.1, "confidence": 0.95}')
    assert value == {"size": Decimal("0.1"), "confidence": Decimal("0.95")}
    assert isinstance(value["size"], Decimal)


@pytest.mark.parametrize("reply, message", [
    ("", "empty"),
    ("I can't help with that.", "no JSON object"),
    ('{"a": 1,, }', "broken"),
    ('{"a": "unterminated}', "cut off"),
    ('{"sku": "A1", "confidence": ', "broken or cut off"),
])
def test_bad_replies_say_why(reply, message):
    with pytest.raises(ValueError, match=message):
        extract_json(reply)


def test_describe_error_lists_each_problem():
    class FakeValidationError(ValueError):
        def errors(self):
            return [{"loc": ("lines", 0, "product_code"), "msg": "not on the list"}, {"loc": (), "msg": "bad"}]

    assert describe_error(FakeValidationError()) == "lines.0.product_code: not on the list; (whole answer): bad"
    assert describe_error(ValueError("plain")) == "plain"
