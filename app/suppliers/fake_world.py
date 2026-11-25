"""The fake supplier world: invented products, pack sizes and prices.

FAKE TEST DATA. Nothing here comes from a real supplier and nothing is scraped.
Every number comes from a fixed seed, so the same seed gives the same prices on
every run and every machine. Randomness is keyed with SHA-256, not Python's
hash(), because hash() of a string changes between runs.

Prices are built with Decimal and integer basis points, then rounded once to
whole cents. No floats touch money.

Quantities are in base units: lb, fl oz, each (see app/units.py, which this
module uses for conversion; both are standard-library only). This module is the
ground truth for tests: each item's pack is stored as structured data (count,
size, unit) and the raw pack text is rendered from it, so tests can check that
the app's pack reader recovers the same structure from the text.

Planted problems, so the checks and reports have something real to find:
  - Butter went up 15% at every supplier 3 weeks ago (a market change).
  - Eggs went up 18% at US Foods only, 2 weeks ago (one supplier raised it).
  - PFG's heavy cream shows $0.00 this week (bad data).
  - Supplier 4's vanilla jumped 70% this week (above the 50% limit).
  - Sysco's matcha and PFG's almond flour are out of stock this week.
  - Sysco writes fluid ounces as plain "OZ", so "4/32 OZ" vanilla is ambiguous
    unless you know vanilla is measured by volume.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from functools import lru_cache

from app import units

HISTORY_WEEKS = 8  # weeks of past prices and orders; week index 0 = this week
SUPPLIER_CODES = ("sysco", "usfoods", "pfg", "supplier4")

@dataclass(frozen=True)
class FakeProduct:
    code: str
    name: str
    base_unit: str          # lb, fl_oz or each
    category: str
    cents_per_base: Decimal  # rough wholesale price per lb / fl oz / piece


@dataclass(frozen=True)
class FakeOffer:
    supplier_code: str
    product_code: str
    sku: str
    description: str
    count: int        # packs per case
    size: Decimal     # size of one pack, in `unit`
    unit: str         # app.units code
    raw_pack_text: str

    @property
    def pack_base_qty(self) -> Decimal:
        """Whole case in the product's base unit (lb, fl oz or each). Ground truth."""
        qty = units.to_base(self.count * self.size, self.unit, product(self.product_code).base_unit)
        assert qty is not None, f"{self.sku}: {self.unit} can't convert"  # checked in offers_for
        return qty


def _p(code: str, name: str, unit: str, category: str, cents: str) -> FakeProduct:
    return FakeProduct(code, name, unit, category, Decimal(cents))


PRODUCTS: tuple[FakeProduct, ...] = (
    _p("ap_flour", "All-purpose flour", "lb", "baking", "55"),
    _p("cake_flour", "Cake flour", "lb", "baking", "90"),
    _p("almond_flour", "Almond flour, blanched", "lb", "baking", "650"),
    _p("granulated_sugar", "Granulated sugar", "lb", "baking", "80"),
    _p("powdered_sugar", "Powdered sugar", "lb", "baking", "110"),
    _p("brown_sugar", "Light brown sugar", "lb", "baking", "100"),
    _p("butter_unsalted", "Unsalted butter", "lb", "dairy", "380"),
    _p("heavy_cream", "Heavy cream", "fl_oz", "dairy", "12.5"),
    _p("whole_milk", "Whole milk", "fl_oz", "dairy", "2.75"),
    _p("oat_milk", "Oat milk, barista", "fl_oz", "dairy alternative", "11.9"),
    _p("eggs", "Large eggs", "each", "dairy", "25"),
    _p("cream_cheese", "Cream cheese", "lb", "dairy", "320"),
    _p("dark_chocolate", "Dark chocolate callets", "lb", "chocolate", "700"),
    _p("cocoa_powder", "Cocoa powder", "lb", "chocolate", "600"),
    _p("vanilla_extract", "Pure vanilla extract", "fl_oz", "flavoring", "177"),
    _p("espresso_beans", "Espresso beans", "lb", "coffee", "900"),
    _p("matcha_powder", "Matcha powder", "lb", "coffee", "2720"),
    _p("strawberries", "Strawberries", "lb", "produce", "300"),
    _p("lemons", "Lemons", "each", "produce", "40"),
    _p("sea_salt", "Fine sea salt", "lb", "baking", "150"),
    _p("baking_powder", "Baking powder", "lb", "baking", "300"),
    _p("hot_cup_12oz", "Hot cup 12 oz paper", "each", "packaging", "8"),
    _p("cup_lid_12oz", "Lid for 12 oz hot cup", "each", "packaging", "4"),
    _p("caramel_sauce", "Caramel sauce", "fl_oz", "flavoring", "18.75"),
)
_PRODUCTS_BY_CODE = {p.code: p for p in PRODUCTS}

# (packs per case, pack size, unit). Missing supplier = doesn't carry it.
_PACKS: dict[str, dict[str, tuple[int, str, str]]] = {
    "ap_flour":         {"sysco": (1, "50", "lb"), "usfoods": (1, "25", "lb"), "pfg": (1, "50", "lb"), "supplier4": (4, "5", "lb")},
    "cake_flour":       {"sysco": (1, "50", "lb"), "usfoods": (1, "25", "lb"), "pfg": (1, "50", "lb")},
    "almond_flour":     {"sysco": (1, "25", "lb"), "usfoods": (3, "2", "lb"), "pfg": (1, "25", "lb"), "supplier4": (6, "1", "lb")},
    "granulated_sugar": {"sysco": (1, "50", "lb"), "usfoods": (1, "25", "lb"), "pfg": (1, "50", "lb"), "supplier4": (4, "4", "lb")},
    "powdered_sugar":   {"sysco": (1, "50", "lb"), "usfoods": (12, "2", "lb"), "pfg": (1, "50", "lb")},
    "brown_sugar":      {"sysco": (1, "25", "lb"), "usfoods": (12, "2", "lb"), "pfg": (1, "50", "lb"), "supplier4": (4, "4", "lb")},
    "butter_unsalted":  {"sysco": (36, "1", "lb"), "usfoods": (30, "1", "lb"), "pfg": (36, "1", "lb"), "supplier4": (12, "1", "lb")},
    "heavy_cream":      {"sysco": (12, "1", "qt"), "usfoods": (6, "64", "fl_oz"), "pfg": (12, "1", "qt"), "supplier4": (12, "1", "qt")},
    "whole_milk":       {"sysco": (4, "1", "gal"), "usfoods": (4, "1", "gal"), "pfg": (4, "1", "gal"), "supplier4": (2, "1", "gal")},
    "oat_milk":         {"sysco": (12, "32", "fl_oz"), "usfoods": (12, "32", "fl_oz"), "supplier4": (6, "946", "ml")},
    "eggs":             {"sysco": (1, "15", "dz"), "usfoods": (1, "30", "dz"), "pfg": (1, "15", "dz"), "supplier4": (1, "5", "dz")},
    "cream_cheese":     {"sysco": (10, "3", "lb"), "usfoods": (6, "3", "lb"), "pfg": (1, "30", "lb"), "supplier4": (12, "8", "oz")},
    "dark_chocolate":   {"sysco": (2, "5", "kg"), "usfoods": (4, "2.5", "kg"), "pfg": (1, "22", "lb"), "supplier4": (6, "1", "kg")},
    "cocoa_powder":     {"sysco": (1, "25", "lb"), "usfoods": (6, "1", "kg"), "pfg": (1, "50", "lb")},
    "vanilla_extract":  {"sysco": (4, "32", "fl_oz"), "usfoods": (1, "1", "gal"), "pfg": (12, "16", "fl_oz"), "supplier4": (6, "16", "fl_oz")},
    "espresso_beans":   {"sysco": (6, "2", "lb"), "usfoods": (5, "5", "lb"), "pfg": (6, "2", "lb"), "supplier4": (12, "12", "oz")},
    "matcha_powder":    {"sysco": (1, "1", "kg"), "usfoods": (10, "100", "g"), "supplier4": (1, "500", "g")},
    "strawberries":     {"sysco": (8, "1", "lb"), "usfoods": (1, "8", "lb"), "pfg": (8, "1", "lb"), "supplier4": (4, "2", "lb")},
    "lemons":           {"sysco": (1, "115", "each"), "usfoods": (1, "140", "each"), "pfg": (1, "165", "each"), "supplier4": (1, "40", "each")},
    "sea_salt":         {"sysco": (12, "26", "oz"), "usfoods": (1, "25", "lb"), "pfg": (12, "26", "oz")},
    "baking_powder":    {"sysco": (6, "5", "lb"), "usfoods": (1, "10", "lb"), "pfg": (6, "5", "lb"), "supplier4": (6, "1", "lb")},
    "hot_cup_12oz":     {"sysco": (1, "1000", "each"), "usfoods": (20, "50", "each"), "pfg": (1, "1000", "each"), "supplier4": (1, "500", "each")},
    "cup_lid_12oz":     {"sysco": (1, "1000", "each"), "usfoods": (10, "100", "each"), "pfg": (1, "1000", "each"), "supplier4": (1, "500", "each")},
    "caramel_sauce":    {"sysco": (6, "64", "fl_oz"), "usfoods": (4, "1.89", "l"), "pfg": (6, "64", "fl_oz")},
}

# Overall price level per supplier, in basis points of the base price.
_SUPPLIER_LEVEL_BP = {"sysco": 10000, "usfoods": 9600, "pfg": 9300, "supplier4": 10600}
_NOISE_BP = (9200, 10800)   # fixed per supplier+item: some items are cheap at one supplier, dear at another
_DRIFT_BP = (9950, 10050)   # weekly market drift per product, shared by all suppliers
_WIGGLE_BP = (9980, 10020)  # small weekly noise per supplier+item


@dataclass(frozen=True)
class _PriceEvent:
    product_code: str
    supplier_codes: tuple[str, ...] | None  # None = every supplier
    since_weeks_ago: int                    # applies to this week index and all newer weeks
    factor: Decimal


_EVENTS = (
    _PriceEvent("butter_unsalted", None, 3, Decimal("1.15")),
    _PriceEvent("eggs", ("usfoods",), 2, Decimal("1.18")),
    _PriceEvent("vanilla_extract", ("supplier4",), 0, Decimal("1.70")),
)
_ZERO_PRICE_NOW = {("pfg", "heavy_cream")}
_OUT_OF_STOCK_NOW = {("sysco", "matcha_powder"), ("pfg", "almond_flour")}

# What each location usually needs per week, in base units. Drives fake order history.
USUAL_WEEKLY_NEED: dict[str, dict[str, Decimal]] = {
    "court_st": {
        "whole_milk": Decimal(2000), "oat_milk": Decimal(680), "espresso_beans": Decimal(20),
        "matcha_powder": Decimal("1.75"), "caramel_sauce": Decimal(135), "hot_cup_12oz": Decimal(1500),
        "cup_lid_12oz": Decimal(1500), "lemons": Decimal(30), "heavy_cream": Decimal(200),
        "vanilla_extract": Decimal(16),
    },
    "factory": {
        "ap_flour": Decimal(100), "cake_flour": Decimal(45), "almond_flour": Decimal(18),
        "granulated_sugar": Decimal(90), "powdered_sugar": Decimal(35), "brown_sugar": Decimal(18),
        "butter_unsalted": Decimal(65), "heavy_cream": Decimal(1000), "eggs": Decimal(720),
        "cream_cheese": Decimal(33), "dark_chocolate": Decimal(22), "cocoa_powder": Decimal(9),
        "vanilla_extract": Decimal(50), "strawberries": Decimal(26), "lemons": Decimal(120),
        "sea_salt": Decimal("3.5"), "baking_powder": Decimal("4.5"),
    },
}


# ------------------------------------------------------------------ helpers

def stable_seed(*parts: object) -> int:
    """Deterministic 64-bit seed from any parts. Same on every machine and run."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _bp(rng: random.Random, lo: int, hi: int) -> Decimal:
    return Decimal(rng.randint(lo, hi)) / Decimal(10000)


def _num(value: Decimal) -> str:
    return format(value.normalize(), "f")  # Decimal("50") -> "50", Decimal("2.5") -> "2.5"


def _check_week(week_index: int) -> None:
    if not 0 <= week_index <= HISTORY_WEEKS:
        raise ValueError(f"week_index must be 0..{HISTORY_WEEKS}, got {week_index}")


# Each supplier writes pack sizes its own way. This is the "messy text".
_SYSCO_UNIT = {"lb": "LB", "oz": "OZ", "fl_oz": "OZ", "kg": "KG", "g": "GM", "qt": "QT",
               "gal": "GAL", "l": "LTR", "ml": "ML", "dz": "DZ", "each": "CT"}
_USF_UNIT = {"lb": "lb", "oz": "oz", "fl_oz": "fl oz", "kg": "kg", "g": "g", "qt": "qt",
             "gal": "gal", "l": "L", "ml": "ml", "dz": "dz", "each": "ct"}
_PFG_SUFFIX = {"lb": "#", "oz": " OZ", "fl_oz": " FL OZ", "kg": " KG", "g": " G", "qt": " QT",
               "gal": " GAL", "l": " L", "ml": " ML", "dz": " DOZ", "each": " CT"}
_S4_WORDS = {"lb": ("pound", "pounds"), "oz": ("ounce", "ounces"), "fl_oz": ("fl. ounce", "fl. ounces"),
             "kg": ("kilogram", "kilograms"), "g": ("gram", "grams"), "qt": ("quart", "quarts"),
             "gal": ("gallon", "gallons"), "l": ("liter", "liters"), "ml": ("ml", "ml"),
             "dz": ("dozen", "dozen"), "each": ("count", "count")}


def _render_pack(supplier_code: str, count: int, size: Decimal, unit: str) -> str:
    n = _num(size)
    if supplier_code == "sysco":
        return f"{count}/{n} {_SYSCO_UNIT[unit]}"
    if supplier_code == "pfg":
        if count == 1 and unit == "lb":
            return f"{n}# BAG"
        if count == 1 and unit in ("each", "dz"):
            return f"{n}{_PFG_SUFFIX[unit]}"
        return f"CS {count} / {n}{_PFG_SUFFIX[unit]}"
    if supplier_code == "supplier4":
        singular, plural = _S4_WORDS[unit]
        return f"Case of {count} x {n} {singular if size == 1 else plural}"
    return f"{count}x{n} {_USF_UNIT[unit]}"  # US Foods style, also the default


_PFG_ABBREV = {"FLOUR": "FLR", "SUGAR": "SUG", "CHOCOLATE": "CHOC", "CREAM": "CRM", "POWDER": "PWD",
               "BUTTER": "BTR", "UNSALTED": "UNSLTD", "EXTRACT": "EXT", "STRAWBERRIES": "STRAWB",
               "ESPRESSO": "ESP", "VANILLA": "VAN", "CHEESE": "CHS"}


def _describe(supplier_code: str, name: str, count: int) -> str:
    words = name.replace(",", "").upper().split()
    if supplier_code == "sysco":
        return f"{words[-1]}, {' '.join(words[:-1])}" if len(words) > 1 else words[0]
    if supplier_code == "pfg":
        return " ".join(_PFG_ABBREV.get(w, w) for w in words)
    if supplier_code == "supplier4":
        return name.lower()
    return f"{name} (bulk)" if count == 1 else f"{name}, {count} pk"


def _sku(supplier_code: str, product_code: str) -> str:
    n = stable_seed("sku", supplier_code, product_code)
    if supplier_code == "sysco":
        return str(1_000_000 + n % 9_000_000)
    if supplier_code == "usfoods":
        return f"US{100_000 + n % 900_000}"
    if supplier_code == "pfg":
        return f"P-{10_000 + n % 90_000}"
    if supplier_code == "supplier4":
        return f"S4-{product_code.upper().replace('_', '-')}"
    return f"{supplier_code.upper()}-{n % 1_000_000:06d}"


# ------------------------------------------------------------------ public API

def product(code: str) -> FakeProduct:
    return _PRODUCTS_BY_CODE[code]


@lru_cache(maxsize=None)
def offers_for(supplier_code: str) -> tuple[FakeOffer, ...]:
    """Everything a supplier sells. Unknown supplier codes sell nothing."""
    out = []
    for prod in PRODUCTS:
        pack = _PACKS[prod.code].get(supplier_code)
        if pack is None:
            continue
        count, size_text, unit = pack
        size = Decimal(size_text)
        if units.UNITS[unit].base != prod.base_unit:
            raise ValueError(f"{supplier_code}/{prod.code}: unit {unit} doesn't fit base unit {prod.base_unit}")
        out.append(FakeOffer(
            supplier_code=supplier_code,
            product_code=prod.code,
            sku=_sku(supplier_code, prod.code),
            description=_describe(supplier_code, prod.name, count),
            count=count,
            size=size,
            unit=unit,
            raw_pack_text=_render_pack(supplier_code, count, size, unit),
        ))
    skus = [o.sku for o in out]
    if len(set(skus)) != len(skus):
        raise RuntimeError(f"fake SKU collision at {supplier_code}")
    return tuple(out)


def all_offers() -> tuple[FakeOffer, ...]:
    return tuple(o for code in SUPPLIER_CODES for o in offers_for(code))


def offer(supplier_code: str, product_code: str) -> FakeOffer:
    for o in offers_for(supplier_code):
        if o.product_code == product_code:
            return o
    raise KeyError(f"{supplier_code} does not sell {product_code}")


@lru_cache(maxsize=None)
def _product_drift(product_code: str, seed: int) -> tuple[Decimal, ...]:
    """Market level for a product; index = weeks ago. Walks forward from the oldest week."""
    rng = random.Random(stable_seed(seed, "drift", product_code))
    levels = [Decimal(1)]
    for _ in range(HISTORY_WEEKS):
        levels.append(levels[-1] * _bp(rng, *_DRIFT_BP))
    return tuple(reversed(levels))


def _event_factor(o: FakeOffer, week_index: int) -> Decimal:
    factor = Decimal(1)
    for ev in _EVENTS:
        if ev.product_code != o.product_code or week_index > ev.since_weeks_ago:
            continue
        if ev.supplier_codes is None or o.supplier_code in ev.supplier_codes:
            factor *= ev.factor
    return factor


def case_price_cents(o: FakeOffer, week_index: int, seed: int) -> int:
    """Price of one full case, in whole cents, `week_index` weeks ago (0 = this week)."""
    _check_week(week_index)
    if week_index == 0 and (o.supplier_code, o.product_code) in _ZERO_PRICE_NOW:
        return 0
    noise = random.Random(stable_seed(seed, "noise", o.supplier_code, o.product_code))
    wiggle = random.Random(stable_seed(seed, "wiggle", o.supplier_code, o.product_code, week_index))
    level = Decimal(_SUPPLIER_LEVEL_BP.get(o.supplier_code, 10000)) / Decimal(10000)
    price = (product(o.product_code).cents_per_base * o.pack_base_qty * level
             * _bp(noise, *_NOISE_BP)
             * _product_drift(o.product_code, seed)[week_index]
             * _bp(wiggle, *_WIGGLE_BP)
             * _event_factor(o, week_index))
    return int(price.quantize(Decimal(1), rounding=ROUND_HALF_UP))


def in_stock(o: FakeOffer, week_index: int) -> bool:
    _check_week(week_index)
    return not (week_index == 0 and (o.supplier_code, o.product_code) in _OUT_OF_STOCK_NOW)


def history_need(location_code: str, product_code: str, week_index: int, seed: int) -> Decimal:
    """How much a location needed of a product in a past week, in base units (to 0.01)."""
    base = USUAL_WEEKLY_NEED[location_code][product_code]
    pct = random.Random(stable_seed(seed, "need", location_code, product_code, week_index)).randint(85, 115)
    return (base * pct / Decimal(100)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


_CONF_PREFIX = {"sysco": "SYS", "usfoods": "USF", "pfg": "PFG", "supplier4": "S4"}


def confirmation_for(supplier_code: str, order_id: str) -> str:
    """The confirmation a fake supplier gives for an order id. Same id -> same answer."""
    digest = hashlib.sha256(f"{supplier_code}|{order_id}".encode("utf-8")).hexdigest()[:10].upper()
    return f"FAKE-{_CONF_PREFIX.get(supplier_code, supplier_code.upper())}-{digest}"


def world_digest(seed: int) -> str:
    """One hash over every fake price, stock flag and pack text. Used to prove determinism."""
    h = hashlib.sha256()
    for o in all_offers():
        for w in range(HISTORY_WEEKS + 1):
            h.update(f"{o.supplier_code}|{o.sku}|{o.raw_pack_text}|{o.pack_base_qty}|{w}|"
                     f"{case_price_cents(o, w, seed)}|{in_stock(o, w)}\n".encode("utf-8"))
    return h.hexdigest()
