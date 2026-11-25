"""Reading a manager's message with plain code: quantities, dates, and checks that
an AI answer only uses words and numbers that are really in the message.

Nothing here guesses. Anything unusual returns None and ends up in "unclear".
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from fractions import Fraction

from app import clock, units

NUMBER_WORDS: dict[str, Fraction] = {
    "a": Fraction(1), "an": Fraction(1), "one": Fraction(1), "two": Fraction(2), "three": Fraction(3),
    "four": Fraction(4), "five": Fraction(5), "six": Fraction(6), "seven": Fraction(7), "eight": Fraction(8),
    "nine": Fraction(9), "ten": Fraction(10), "eleven": Fraction(11), "twelve": Fraction(12),
    "fifteen": Fraction(15), "twenty": Fraction(20), "thirty": Fraction(30), "forty": Fraction(40),
    "fifty": Fraction(50), "hundred": Fraction(100), "half": Fraction(1, 2),
}
_UNICODE_FRACTIONS = {"½": Fraction(1, 2), "¼": Fraction(1, 4), "¾": Fraction(3, 4)}

# Supplier-specific pack words. "3 cases" can't become lb without knowing whose case.
CASE_WORDS = {"case", "cases", "cs", "bag", "bags", "box", "boxes", "pack", "packs", "pk", "carton", "cartons",
              "tub", "tubs", "jug", "jugs", "bottle", "bottles", "can", "cans", "sleeve", "sleeves"}

_NUM = r"\d+(?:\.\d+)?|\.\d+"
_QTY_PATTERNS = [
    (re.compile(rf"^(\d+)\s*[-\s]\s*(\d+)/(\d+)$"), "mixed"),        # 1 1/2, 1-1/2
    (re.compile(r"^(\d+)/(\d+)$"), "fraction"),                       # 1/2
    (re.compile(rf"^(\d+)?([½¼¾])$"), "unicode"),                     # ½, 1½
    (re.compile(rf"^({_NUM})$"), "number"),                           # 12, 1.5, .5
]
QTY_TOKEN = r"(?:\d+\s*-\s*\d+/\d+|\d+\s+\d+/\d+|\d+/\d+|\d*[½¼¾]|\d+(?:\.\d+)?|\.\d+)"


def parse_quantity(text: str) -> Decimal | None:
    """'12' -> 12, '1 1/2' -> 1.5, '½' -> 0.5, 'two' -> 2, 'a' -> 1. Anything else -> None."""
    if not isinstance(text, str):
        return None
    s = re.sub(r"\s+", " ", text.strip().lower().replace(",", ""))
    if s in NUMBER_WORDS:
        return units.fraction_to_decimal(NUMBER_WORDS[s])
    for pattern, kind in _QTY_PATTERNS:
        m = pattern.match(s)
        if not m:
            continue
        if kind == "mixed":
            den = int(m[3])
            return units.fraction_to_decimal(int(m[1]) + Fraction(int(m[2]), den)) if den else None
        if kind == "fraction":
            den = int(m[2])
            return units.fraction_to_decimal(Fraction(int(m[1]), den)) if den else None
        if kind == "unicode":
            return units.fraction_to_decimal(int(m[1] or 0) + _UNICODE_FRACTIONS[m[2]])
        return Decimal(m[1])
    return None


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def mentions(text: str, fragment: str) -> bool:
    """Is `fragment` in `text` as a whole word or number (case and spacing ignored)?

    '5' is not in '50 lb' or '1.5 kg'; '1.5' is in '1.5kg'; 'lb' is in '50lb'; 'Dec 3' is in 'by Dec 3.'
    (a period or slash only joins numbers when there is a digit on its other side).
    """
    t, f = _norm_text(text), _norm_text(fragment)
    if not f:
        return False
    numeric_start = f[0].isdigit() or f[0] in ".½¼¾"
    numeric_end = f[-1].isdigit() or f[-1] in "½¼¾"

    def char(i: int) -> str:
        return t[i] if 0 <= i < len(t) else " "

    for m in re.finditer(re.escape(f), t):
        before, after = char(m.start() - 1), char(m.end())
        if numeric_start:
            ok_before = not (before.isdigit() or (before in "./" and char(m.start() - 2).isdigit()))
        else:
            ok_before = not before.isalpha()
        if numeric_end:
            ok_after = not (after.isdigit() or (after in "./" and char(m.end() + 1).isdigit()))
        else:
            ok_after = not after.isalpha()
        if ok_before and ok_after:
            return True
    return False


def quantities_in(text: str) -> set[Decimal]:
    """Every quantity written in the message, as numbers or number words."""
    found = set()
    for m in re.finditer(QTY_TOKEN, text):
        value = parse_quantity(m.group(0))
        if value is not None:
            found.add(value)
    for word in re.findall(r"[a-z]+", text.lower()):
        if word in NUMBER_WORDS:
            found.add(units.fraction_to_decimal(NUMBER_WORDS[word]))
    return found


def quantity_in_text(text: str, quantity: str) -> bool:
    """True if this quantity was really written in the message ('2' also matches 'two')."""
    if mentions(text, quantity):
        return True
    value = parse_quantity(quantity)
    return value is not None and value in quantities_in(text)


def _singular(word: str) -> str:
    """Rough singular, applied the same way to both sides of a comparison."""
    if len(word) > 3 and word.endswith("ies"):
        return word[:-3] + "y"                         # berries -> berry
    if word.endswith(("sses", "xes", "zes", "ches", "shes")):
        return word[:-2]                               # glasses -> glass, boxes -> box
    if len(word) > 2 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]                               # cases -> case, lbs -> lb
    return word


def unit_in_text(text: str, unit: str) -> bool:
    """True if the message uses this unit, in any spelling ('lb' matches 'lbs' and 'pounds')."""
    if mentions(text, unit):
        return True
    code = units.normalize_unit(unit)
    words = re.findall(r"[a-z]+|#", text.lower())
    pairs = [f"{a} {b}" for a, b in zip(words, words[1:])]
    if code is not None:
        return any(units.normalize_unit(w) == code for w in words + pairs)
    target = _singular(unit.lower().strip())
    return any(_singular(w) == target for w in words)


def normalize(s: str) -> str:
    """For fuzzy matching: lowercase, punctuation to spaces, plurals to singular."""
    s = s.lower().replace("&", " and ")
    words = re.sub(r"[^a-z0-9]+", " ", s).split()
    return " ".join(_singular(w) for w in words)


# ------------------------------------------------------------------ dates

_WEEKDAY_WORDS = {
    "mon": 0, "monday": 0, "tue": 1, "tues": 1, "tuesday": 1, "wed": 2, "weds": 2, "wednesday": 2,
    "thu": 3, "thur": 3, "thurs": 3, "thursday": 3, "fri": 4, "friday": 4, "sat": 5, "saturday": 5,
    "sun": 6, "sunday": 6,
}
_MONTHS = {m: n for n, names in enumerate(
    [("jan", "january"), ("feb", "february"), ("mar", "march"), ("apr", "april"), ("may",), ("jun", "june"),
     ("jul", "july"), ("aug", "august"), ("sep", "sept", "september"), ("oct", "october"),
     ("nov", "november"), ("dec", "december")], start=1) for m in names}
_DATE_LEAD = re.compile(r"^(?:needed\s+|need\s+it\s+|due\s+)?(?:by|for|on|before|due|until)?\s*(?:this\s+)?")


def parse_needed_by(text: str, today: date) -> date | None:
    """'Friday', 'by thu', 'tomorrow', '12/3', 'Dec 3', '2026-12-03' -> a date. Unsure -> None.

    A bare weekday means the next one after today ('Friday' said on a Friday is a week out).
    'next Friday' is ambiguous (this coming one, or the week after?), so it returns None.
    """
    s = re.sub(r"[,.!?]+$", "", _norm_text(text))
    s = _DATE_LEAD.sub("", s).strip()
    s = re.sub(r"\s+please$", "", s)
    if not s or s.startswith("next "):
        return None
    if s == "today":
        return today
    if s == "tomorrow":
        return today + timedelta(days=1)
    if s in _WEEKDAY_WORDS:
        ahead = (_WEEKDAY_WORDS[s] - today.weekday()) % 7 or 7
        return today + timedelta(days=ahead)
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if m:
        return _safe_date(int(m[1]), int(m[2]), int(m[3]))
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})(?:/(\d{2}|\d{4}))?", s)  # US month/day
    if m:
        return _with_year(int(m[1]), int(m[2]), m[3], today)
    m = re.fullmatch(r"([a-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?", s) or None
    if m and m[1] in _MONTHS:
        return _with_year(_MONTHS[m[1]], int(m[2]), None, today)
    m = re.fullmatch(r"(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]+)", s)
    if m and m[2] in _MONTHS:
        return _with_year(_MONTHS[m[2]], int(m[1]), None, today)
    return None


def _safe_date(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def _with_year(month: int, day: int, year: str | None, today: date) -> date | None:
    if year is not None:
        y = int(year)
        return _safe_date(y + 2000 if y < 100 else y, month, day)
    found = _safe_date(today.year, month, day)
    if found is not None and found < today:  # "Jan 5" said in November means next January
        found = _safe_date(today.year + 1, month, day)
    return found


_DATE_PHRASE = re.compile(
    r"\b(?:needed\s+|due\s+)?(?:by|for|on|before|due)\s+(?:this\s+)?"
    r"(today|tomorrow|[a-z]{3,9}(?:\s+\d{1,2}(?:st|nd|rd|th)?)?|\d{1,2}/\d{1,2}(?:/\d{2,4})?|\d{4}-\d{2}-\d{2})\b"
    r"|\b(today|tomorrow)\b",
    re.I,
)


@dataclass(frozen=True)
class DateMention:
    text: str        # the phrase as written, e.g. "by Friday"
    value: date | None


def find_needed_by(text: str, today: date) -> list[DateMention]:
    """Date phrases in a message, for the no-AI reader. Only phrases that read as dates are returned."""
    out = []
    for m in _DATE_PHRASE.finditer(text):
        value = parse_needed_by(m.group(0), today)
        if value is not None:
            out.append(DateMention(m.group(0).strip(), value))
    return out


def day_label(d: date) -> str:
    return f"{clock.weekday_name(d)} {d:%b} {d.day}"


# ------------------------------------------------------------------ no-AI line reader

_FILLER = re.compile(
    r"^(?:hi|hey|hello|thanks|thank you|please|pls|we need|we'll need|i need|need|also|plus|and|"
    r"can you|could you|send|send us|get|order|we want|add)\b[\s,:!-]*", re.I)
_TRAILING_FILLER = re.compile(r"[\s,:!-]*\b(?:please|pls|thanks|thank you|asap)$", re.I)
# Split on line breaks, ; , ! ?, sentence periods (not decimal points) and joining words.
_SPLIT = re.compile(r"[\n;,!?]|(?<!\d)\.|\.(?!\d)|\band\b|\balso\b|\bplus\b", re.I)


def _unit_words() -> list[str]:
    words = {w.lower() for w in units._ALIASES} | CASE_WORDS  # noqa: SLF001  (shared vocabulary)
    return sorted(words, key=len, reverse=True)


_UNIT_RE = "|".join(re.escape(w) for w in _unit_words() if w != "#") + "|#"
_WORD_QTY = "|".join(sorted(NUMBER_WORDS, key=len, reverse=True))
_LINE_QTY_FIRST = re.compile(
    rf"^(?P<qty>{QTY_TOKEN}|(?:{_WORD_QTY})\b)\s*(?:(?P<unit>{_UNIT_RE})(?![a-z]))?\.?\s*(?:of\s+)?(?P<product>.+)$",
    re.I)
_LINE_QTY_LAST = re.compile(
    rf"^(?P<product>.+?)\s*[:x×-]?\s*(?P<qty>{QTY_TOKEN}|(?:{_WORD_QTY})\b)\s*(?:(?P<unit>{_UNIT_RE})(?![a-z]))?\.?$",
    re.I)


@dataclass(frozen=True)
class RawLine:
    product_text: str
    quantity: str | None
    unit: str | None


def _match_line(seg: str) -> re.Match | None:
    m = _LINE_QTY_FIRST.match(seg) or _LINE_QTY_LAST.match(seg)
    if m:
        return m
    # "For the factory we need 50 lb flour": start reading at the first number
    for start in re.finditer(r"\d|[½¼¾]", seg):
        m = _LINE_QTY_FIRST.match(seg[start.start():])
        if m:
            return m
    return None


def split_lines(text: str, today: date) -> list[RawLine]:
    """The no-AI reader: split a message into (product words, quantity, unit) as written."""
    def drop_dates(m: re.Match) -> str:
        return " " if parse_needed_by(m.group(0), today) is not None else m.group(0)

    cleaned = _DATE_PHRASE.sub(drop_dates, text)
    out = []
    for part in _SPLIT.split(cleaned):
        seg = part.strip(" .!?-")
        while True:
            stripped = _TRAILING_FILLER.sub("", _FILLER.sub("", seg)).strip(" .!?-")
            if stripped == seg:
                break
            seg = stripped
        if not seg:
            continue
        m = _match_line(seg)
        if m and m["product"].strip(" .,-"):
            out.append(RawLine(m["product"].strip(" .,-"), m["qty"].strip(), m["unit"]))
        else:
            out.append(RawLine(seg, None, None))
    return out
