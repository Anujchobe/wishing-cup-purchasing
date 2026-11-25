from decimal import Decimal
from fractions import Fraction

import pytest
from hypothesis import given
from hypothesis import strategies as st

from app import units
from app.suppliers import fake_world
from app.units import Pack, pack_base_qty, parse_pack, to_base

D = Decimal


@pytest.mark.parametrize(
    "text, count, size, unit, base, qty",
    [
        ("36/1 LB", 36, "1", "lb", "lb", "36"),
        ("4/8#", 4, "8", "lb", "lb", "32"),
        ("6/0.5 GAL", 6, "0.5", "gal", "fl_oz", "384"),
        ("15 DZ", 1, "15", "dz", "each", "180"),
        ("1000 CT", 1, "1000", "each", "each", "1000"),
        ("50# BAG", 1, "50", "lb", "lb", "50"),
        ("CS 36 / 1#", 36, "1", "lb", "lb", "36"),
        ("2x25lb", 2, "25", "lb", "lb", "50"),
        ("12/26 OZ", 12, "26", "oz", "lb", "19.5"),
        ("6x64 fl oz", 6, "64", "fl_oz", "fl_oz", "384"),
        ("Case of 12 x 1 quart", 12, "1", "qt", "fl_oz", "384"),
        ("Case of 6 x 1 fl. ounce", 6, "1", "fl_oz", "fl_oz", "6"),
        ("4x2.5 kg", 4, "2.5", "kg", "lb", "22.046226218"),
        ("1,000 CT", 1, "1000", "each", "each", "1000"),
        ("2/5 LB PAIL", 2, "5", "lb", "lb", "10"),
    ],
)
def test_reads_pack_text(text, count, size, unit, base, qty):
    assert parse_pack(text) == Pack(count, D(size), unit)
    assert pack_base_qty(text, base) == D(qty)


@pytest.mark.parametrize(
    "text",
    ["6/#10 CAN", "4-5 LB", "5 LB AVG", "4/6/8 OZ", "APPROX 5 LB", "LB", "", "0/5 LB", "3/0 LB",
     "12 PK", "1 SACK", "5 LB 2 OZ", "5LB X 4"],
)
def test_unsure_pack_text_returns_none(text):
    assert parse_pack(text) is None


def test_plain_oz_on_a_liquid_is_ambiguous():
    assert parse_pack("4/32 OZ") == Pack(4, D(32), "oz")
    assert pack_base_qty("4/32 OZ", "lb") == D(8)   # a weight product: OZ is weight
    assert pack_base_qty("4/32 OZ", "fl_oz") is None  # a liquid: weight or volume? don't guess
    assert pack_base_qty("4/32 FL OZ", "fl_oz") == D(128)


@pytest.mark.parametrize(
    "qty, unit, base, expected",
    [
        ("1", "oz", "lb", "0.0625"),
        ("0.125", "oz", "lb", "0.0078125"),  # exact, even past 6 places
        ("1", "gal", "fl_oz", "128"),
        ("1", "qt", "fl_oz", "32"),
        ("1", "pt", "fl_oz", "16"),
        ("1", "cup", "fl_oz", "8"),
        ("2", "dz", "each", "24"),
        ("1", "kg", "lb", "2.204622622"),
        ("453.59237", "g", "lb", "1"),        # exactly one pound
        ("1", "l", "fl_oz", "33.814022702"),
        ("29.5735295625", "ml", "fl_oz", "1"),  # exactly one fluid ounce
    ],
)
def test_conversions(qty, unit, base, expected):
    assert to_base(D(qty), unit, base) == D(expected)


@pytest.mark.parametrize("unit, base", [("lb", "fl_oz"), ("gal", "lb"), ("dz", "lb"), ("each", "fl_oz"),
                                        ("furlong", "lb"), ("oz", "fl_oz")])
def test_units_that_dont_fit_return_none(unit, base):
    assert to_base(D(1), unit, base) is None


def test_bad_quantities_are_refused():
    with pytest.raises(TypeError):
        to_base(1.5, "lb", "lb")
    with pytest.raises(ValueError):
        to_base(D(-1), "lb", "lb")
    with pytest.raises(ValueError):
        to_base(D(1), "lb", "kg")  # kg isn't a base unit


def test_aliases():
    for word in ("LB", "lbs", "#", "Pounds"):
        assert units.normalize_unit(word) == "lb"
    assert units.normalize_unit("fl. oz") == "fl_oz"
    assert units.normalize_unit("DOZ") == "dz"
    assert units.normalize_unit("bushel") is None


EXACT = [u for u in units.UNITS.values() if u.code not in ("kg", "g", "ml", "l")]
METRIC = [units.UNITS[c] for c in ("kg", "g", "ml", "l")]


@given(qty=st.decimals(min_value=0, max_value=10**6, places=4), unit=st.sampled_from(EXACT))
def test_us_units_convert_exactly(qty, unit):
    assert Fraction(to_base(qty, unit.code, unit.base)) == Fraction(qty) * unit.to_base


@given(qty=st.decimals(min_value=0, max_value=10**6, places=4), unit=st.sampled_from(METRIC))
def test_metric_units_are_within_half_a_billionth(qty, unit):
    exact = Fraction(qty) * unit.to_base
    assert abs(Fraction(to_base(qty, unit.code, unit.base)) - exact) <= Fraction(1, 2 * 10**units.INEXACT_PLACES)


@given(
    count=st.integers(1, 500),
    size=st.decimals(min_value=D("0.01"), max_value=1000, places=2),
    word=st.sampled_from(["LB", "lb", "#", "OZ", "GAL", "QT", "fl oz", "KG", "DZ", "CT", "ml"]),
    sep=st.sampled_from(["/", " / ", "x", "X", " x "]),
    spaces=st.sampled_from(["", " ", "  "]),
)
def test_reader_handles_any_count_size_and_spacing(count, size, word, sep, spaces):
    text = f"{spaces}{count}{sep}{size}{' ' if word != '#' else ''}{word}{spaces}"
    assert parse_pack(text) == Pack(count, size, units.normalize_unit(word))


def test_reader_agrees_with_the_fake_suppliers_on_every_item():
    """Every fake item's raw text reads back to its true pack. The only Nones are the
    planted ambiguous Sysco liquids written as plain 'OZ'."""
    unsure = set()
    for offer in fake_world.all_offers():
        base = fake_world.product(offer.product_code).base_unit
        parsed = parse_pack(offer.raw_pack_text)
        assert parsed is not None, offer.raw_pack_text
        assert (parsed.count, parsed.size) == (offer.count, offer.size), offer.raw_pack_text
        qty = pack_base_qty(offer.raw_pack_text, base)
        if qty is None:
            unsure.add((offer.supplier_code, offer.product_code))
        else:
            assert qty == offer.pack_base_qty, offer.raw_pack_text
    assert unsure == {("sysco", "oat_milk"), ("sysco", "vanilla_extract"), ("sysco", "caramel_sauce")}
