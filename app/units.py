"""Units: convert quantities to base units and read supplier pack text.

Base units: lb for weight, fl oz for volume, each for counts.

Conversion uses exact fractions, then returns a Decimal:
  * US units (oz, gal, qt, pt, cup, dozen) convert exactly: 1 oz = 0.0625 lb.
  * Metric units can't be exact in lb or fl oz (1 kg = 2.20462262184877... lb).
    They are rounded once, to 9 decimal places (about half a microgram), always
    the same way. The same input therefore always gives the same Decimal, which
    is what matters when the app compares a need with a case size.
  * Weight never converts to volume or count (no densities, no guessing).

Pack reading is deliberately strict. It returns None, never a guess, when the
text is unusual or ambiguous, and the item goes to the AI (step 3) or a person.
The rules:
  * "N/S UNIT" and "NxS UNIT" mean N packs of size S ("4/8#" = 4 x 8 lb).
  * A single "S UNIT" means one pack ("15 DZ", "1000 CT", "50# BAG").
  * Prefixes "CASE OF", "CASE", "CS", "PACK OF" and container words at the end
    ("BAG", "CAN", "JUG"...) are ignored.
  * Plain "OZ" is a weight. For a product measured in fl oz, plain "OZ" is
    ambiguous (weight or volume?), so converting it returns None.
  * Ranges, averages, can sizes and nested packs ("4-5 LB", "5 LB AVG",
    "6/#10 CAN", "4/6/8 OZ") return None.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction

INEXACT_PLACES = 9
BASE_UNITS = ("lb", "fl_oz", "each")
_BASE_FOR_DIMENSION = {"weight": "lb", "volume": "fl_oz", "count": "each"}

_LB_IN_KG = Fraction("0.45359237")       # exact by definition
_FL_OZ_IN_ML = Fraction("29.5735295625")  # exact by definition (US fluid ounce)


@dataclass(frozen=True)
class Unit:
    code: str
    dimension: str      # weight | volume | count
    to_base: Fraction   # how many base units one of this unit is

    @property
    def base(self) -> str:
        return _BASE_FOR_DIMENSION[self.dimension]


UNITS: dict[str, Unit] = {u.code: u for u in (
    Unit("lb", "weight", Fraction(1)),
    Unit("oz", "weight", Fraction(1, 16)),
    Unit("kg", "weight", 1 / _LB_IN_KG),
    Unit("g", "weight", 1 / (_LB_IN_KG * 1000)),
    Unit("fl_oz", "volume", Fraction(1)),
    Unit("cup", "volume", Fraction(8)),
    Unit("pt", "volume", Fraction(16)),
    Unit("qt", "volume", Fraction(32)),
    Unit("gal", "volume", Fraction(128)),
    Unit("ml", "volume", 1 / _FL_OZ_IN_ML),
    Unit("l", "volume", 1000 / _FL_OZ_IN_ML),
    Unit("each", "count", Fraction(1)),
    Unit("dz", "count", Fraction(12)),
)}

_ALIASES: dict[str, str] = {}
for _code, _words in {
    "lb": "LB LBS POUND POUNDS #",
    "oz": "OZ OUNCE OUNCES",
    "kg": "KG KGS KILO KILOS KILOGRAM KILOGRAMS",
    "g": "G GM GMS GR GRAM GRAMS",
    "fl_oz": "FLOZ|FL OZ|FL OUNCE|FL OUNCES|FLUID OUNCE|FLUID OUNCES",
    "cup": "CUP CUPS",
    "pt": "PT PTS PINT PINTS",
    "qt": "QT QTS QUART QUARTS",
    "gal": "GAL GALS GALLON GALLONS",
    "ml": "ML MLS MILLILITER MILLILITERS MILLILITRE MILLILITRES",
    "l": "L LT LTR LTRS LITER LITERS LITRE LITRES",
    "each": "EA EACH CT COUNT PC PCS PIECE PIECES",
    "dz": "DZ DOZ DOZEN DOZENS",
}.items():
    for _word in (_words.split("|") if "|" in _words else _words.split()):
        _ALIASES[_word] = _code

_PREFIXES = re.compile(r"^(?:CASE OF|PACK OF|CASE|CS)\s+")
_CONTAINERS = {"BAG", "BAGS", "BOX", "BOXES", "CASE", "CS", "CAN", "CANS", "JUG", "JUGS", "TUB", "TUBS",
               "PAIL", "JAR", "JARS", "BTL", "BOTTLE", "BOTTLES", "CTN", "CARTON", "CARTONS", "BLOCK",
               "BLOCKS", "PKG", "PACK", "PK", "SLEEVE", "SLEEVES", "BRICK", "BRICKS"}
_PACK = re.compile(r"^(?:(?P<count>\d+)\s*[/X]\s*)?(?P<size>\d+(?:\.\d+)?)\s*(?P<unit>#|[A-Z][A-Z ]*)$")


@dataclass(frozen=True)
class Pack:
    """What a case holds, as written: `count` packs of `size` `unit`."""

    count: int
    size: Decimal
    unit: str

    @property
    def total(self) -> Decimal:
        return self.count * self.size


def normalize_unit(word: str) -> str | None:
    """'LBS', 'lb', '#' -> 'lb'. Unknown words -> None."""
    if word in UNITS:
        return word
    key = re.sub(r"\s+", " ", word.replace(".", " ")).strip().upper()
    return _ALIASES.get(key)


def to_base(qty: Decimal | int, unit: str, base_unit: str) -> Decimal | None:
    """Convert `qty` `unit` to `base_unit`. None if the units don't fit or are ambiguous."""
    if isinstance(qty, (bool, float)) or not isinstance(qty, (Decimal, int)):
        raise TypeError(f"quantity must be Decimal or int, got {type(qty).__name__}")
    qty = Decimal(qty)
    if not qty.is_finite() or qty < 0:
        raise ValueError(f"quantity must be a finite number >= 0, got {qty}")
    if base_unit not in BASE_UNITS:
        raise ValueError(f"unknown base unit {base_unit!r}; expected one of {BASE_UNITS}")
    code = normalize_unit(unit)
    if code is None:
        return None
    if code == "oz" and base_unit == "fl_oz":
        return None  # "OZ" on a liquid: weight or volume? Don't guess.
    u = UNITS[code]
    if u.base != base_unit:
        return None
    return fraction_to_decimal(Fraction(qty) * u.to_base)


def fraction_to_decimal(value: Fraction) -> Decimal:
    """Exact if the fraction has a finite decimal form, else rounded to INEXACT_PLACES (half-even)."""
    den = value.denominator
    twos = fives = 0
    while den % 2 == 0:
        den //= 2
        twos += 1
    while den % 5 == 0:
        den //= 5
        fives += 1
    if den == 1:  # terminates: scale to an integer, no rounding at all
        places = max(twos, fives)
        return _scaled(value.numerator * (10**places // value.denominator), places)
    return _scaled(round(value * 10**INEXACT_PLACES), INEXACT_PLACES)


def _scaled(n: int, places: int) -> Decimal:
    # Built from a string so the context precision can never round it.
    return Decimal(f"{n}E-{places}")


def parse_pack(text: str) -> Pack | None:
    """Read pack text like '36/1 LB', '4/8#', '6/0.5 GAL', '15 DZ', '1000 CT'. None if unsure."""
    if not isinstance(text, str):
        return None
    s = text.upper().replace("×", "X")
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)          # 1,000 -> 1000
    s = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", s)           # drop periods that aren't decimal points
    s = re.sub(r"\s+", " ", s).strip()
    s = _PREFIXES.sub("", s)
    words = s.split(" ")
    while len(words) > 1 and words[-1] in _CONTAINERS:
        words.pop()
    match = _PACK.match(" ".join(words))
    if not match:
        return None
    unit = normalize_unit(match["unit"].strip())
    if unit is None:
        return None
    count = int(match["count"]) if match["count"] else 1
    size = Decimal(match["size"])
    if count <= 0 or size <= 0:
        return None
    return Pack(count=count, size=size, unit=unit)


def pack_base_qty(text: str, base_unit: str) -> Decimal | None:
    """Whole case in base units, read from pack text. None if unsure."""
    pack = parse_pack(text)
    if pack is None:
        return None
    return to_base(pack.total, pack.unit, base_unit)


def display_unit(unit: str) -> str:
    return {"fl_oz": "fl oz"}.get(unit, unit)
