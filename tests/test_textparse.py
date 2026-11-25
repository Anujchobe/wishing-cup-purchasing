from datetime import date
from decimal import Decimal

import pytest

from app import textparse as t

TUE = date(2026, 11, 24)


@pytest.mark.parametrize("text, value", [
    ("12", "12"), ("1.5", "1.5"), (".5", "0.5"), ("1 1/2", "1.5"), ("1-1/2", "1.5"), ("1/2", "0.5"),
    ("½", "0.5"), ("1½", "1.5"), ("two", "2"), ("a", "1"), ("half", "0.5"), ("3,000", "3000"),
])
def test_quantities(text, value):
    assert t.parse_quantity(text) == Decimal(value)


@pytest.mark.parametrize("text", ["a few", "some", "1/0", "lots", "", "2-3"])
def test_unreadable_quantities(text):
    assert t.parse_quantity(text) is None


def test_quantity_must_be_in_the_text():
    msg = "50 lb flour, 2 dozen eggs and two gallons milk"
    assert t.quantity_in_text(msg, "50") and t.quantity_in_text(msg, "2") and t.quantity_in_text(msg, "two")
    assert not t.quantity_in_text(msg, "5")    # '5' is not '50'
    assert not t.quantity_in_text(msg, "40")
    assert t.quantity_in_text("1.5kg chocolate", "1.5")
    assert not t.mentions("1.5 kg", "5") and not t.mentions("1/2 lb", "2") and not t.mentions("1/2 lb", "1")


def test_a_number_at_the_end_of_a_sentence_still_counts():
    """Found by the AI test cases: 'Need it by Dec 3.' lost its date because of the final period."""
    assert t.mentions("Need it by Dec 3.", "by Dec 3") and t.mentions("We need 30.", "30")
    assert t.mentions("deliver by 12/3.", "12/3")


def test_unit_must_be_in_the_text():
    msg = "30 pounds butter, 3 cases cream"
    assert t.unit_in_text(msg, "lb") and t.unit_in_text(msg, "pounds") and t.unit_in_text(msg, "case")
    assert not t.unit_in_text(msg, "kg") and not t.unit_in_text(msg, "gallons")


@pytest.mark.parametrize("text, expected", [
    ("Friday", date(2026, 11, 27)), ("by thu", date(2026, 11, 26)), ("tomorrow", date(2026, 11, 25)),
    ("Tuesday", date(2026, 12, 1)),            # same weekday as today = next week
    ("12/3", date(2026, 12, 3)), ("Dec 3", date(2026, 12, 3)), ("3rd December", date(2026, 12, 3)),
    ("2026-12-03", date(2026, 12, 3)), ("Jan 5", date(2027, 1, 5)), ("needed Monday", date(2026, 11, 30)),
    ("Friday please", date(2026, 11, 27)),
])
def test_dates(text, expected):
    assert t.parse_needed_by(text, TUE) == expected


@pytest.mark.parametrize("text", ["next friday", "soon", "2/30", "for the", "asap"])
def test_unsure_dates_return_none(text):
    assert t.parse_needed_by(text, TUE) is None


def test_no_ai_line_reader():
    msg = ("Hi! For the factory we need 50 lb AP flour, 2 dozen eggs, 3 gallons of heavy cream, "
           "1.5 kg dark chocolate and some saffron. Also vanilla 32 oz. By Friday please.")
    got = [(l.product_text, l.quantity, l.unit) for l in t.split_lines(msg, TUE)]
    assert got == [("AP flour", "50", "lb"), ("eggs", "2", "dozen"), ("heavy cream", "3", "gallons"),
                   ("dark chocolate", "1.5", "kg"), ("some saffron", None, None), ("vanilla", "32", "oz")]
    assert [(m.text, m.value) for m in t.find_needed_by(msg, TUE)] == [("By Friday", date(2026, 11, 27))]


def test_line_reader_word_orders():
    got = [(l.product_text, l.quantity, l.unit) for l in t.split_lines(
        "need 30 lemons\n2 gal whole milk\noat milk x 6\n3 cases butter thanks", TUE)]
    assert got == [("lemons", "30", None), ("whole milk", "2", "gal"), ("oat milk", "6", None), ("butter", "3", "cases")]


def test_normalize_for_matching():
    assert t.normalize("Oat milk, barista") == "oat milk barista"
    assert t.normalize("STRAWBERRIES") == "strawberry"
    assert t.normalize("3 cases & boxes") == "3 case and box"
