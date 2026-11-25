import os
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest

from app.suppliers import fake_world as fw

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
SPECIAL = {("usfoods", "eggs"), ("pfg", "heavy_cream"), ("supplier4", "vanilla_extract")}


def ratio(offer, newer: int, older: int) -> Decimal:
    return Decimal(fw.case_price_cents(offer, newer, SEED)) / Decimal(fw.case_price_cents(offer, older, SEED))


def test_same_numbers_on_every_run_even_with_a_different_hash_seed():
    code = "from app.suppliers import fake_world as f; print(f.world_digest(42))"
    digests = {
        subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True,
                       env={**os.environ, "PYTHONHASHSEED": str(h)}).stdout.strip()
        for h in (0, 1, 999)
    }
    assert digests == {fw.world_digest(42)}
    assert fw.world_digest(42) != fw.world_digest(43)


def test_each_supplier_has_unique_skus_and_main_supplier_sells_everything():
    for code in fw.SUPPLIER_CODES:
        skus = [o.sku for o in fw.offers_for(code)]
        assert len(skus) == len(set(skus)) > 0
    assert {o.product_code for o in fw.offers_for("sysco")} == {p.code for p in fw.PRODUCTS}


def test_unknown_supplier_sells_nothing():
    assert fw.offers_for("supplier_99") == ()


@pytest.mark.parametrize(
    "supplier, product, text, base_qty",
    [
        ("sysco", "ap_flour", "1/50 LB", Decimal("50")),
        ("sysco", "vanilla_extract", "4/32 OZ", Decimal("128")),        # fluid oz written as OZ
        ("usfoods", "dark_chocolate", "4x2.5 kg", Decimal("22.046226218")),
        ("pfg", "ap_flour", "50# BAG", Decimal("50")),
        ("pfg", "butter_unsalted", "CS 36 / 1#", Decimal("36")),
        ("pfg", "eggs", "15 DOZ", Decimal("180")),
        ("supplier4", "heavy_cream", "Case of 12 x 1 quart", Decimal("384")),
        ("supplier4", "cream_cheese", "Case of 12 x 8 ounces", Decimal("6")),
    ],
)
def test_pack_text_and_true_size(supplier, product, text, base_qty):
    offer = fw.offer(supplier, product)
    assert offer.raw_pack_text == text
    assert offer.pack_base_qty == base_qty


def test_prices_are_whole_cents_and_positive_except_the_planted_zero():
    for offer in fw.all_offers():
        for week in range(fw.HISTORY_WEEKS + 1):
            price = fw.case_price_cents(offer, week, SEED)
            assert type(price) is int
            if (offer.supplier_code, offer.product_code) == ("pfg", "heavy_cream") and week == 0:
                assert price == 0
            else:
                assert price > 0


def test_butter_is_a_market_change():
    for offer in (o for o in fw.all_offers() if o.product_code == "butter_unsalted"):
        assert ratio(offer, 0, 4) > Decimal("1.10"), offer.supplier_code


def test_eggs_went_up_at_one_supplier_only():
    eggs = {o.supplier_code: o for o in fw.all_offers() if o.product_code == "eggs"}
    assert ratio(eggs.pop("usfoods"), 0, 4) > Decimal("1.10")
    assert all(ratio(o, 0, 4) < Decimal("1.10") for o in eggs.values())


def test_vanilla_jump_at_supplier4():
    assert ratio(fw.offer("supplier4", "vanilla_extract"), 0, 1) > Decimal("1.50")


def test_no_accidental_anomalies():
    """Only the planted events should trip the 10% watch or the 50% jump limit."""
    for offer in fw.all_offers():
        key = (offer.supplier_code, offer.product_code)
        if key in SPECIAL or offer.product_code == "butter_unsalted":
            continue
        assert ratio(offer, 0, 4) < Decimal("1.10"), key
        for week in range(fw.HISTORY_WEEKS):
            assert ratio(offer, week, week + 1) < Decimal("1.50"), (key, week)


def test_planted_out_of_stock():
    out = {(o.supplier_code, o.product_code) for o in fw.all_offers() if not fw.in_stock(o, 0)}
    assert out == {("sysco", "matcha_powder"), ("pfg", "almond_flour")}
    assert all(fw.in_stock(o, 1) for o in fw.all_offers())


def test_week_index_is_bounded():
    offer = fw.offer("sysco", "eggs")
    with pytest.raises(ValueError):
        fw.case_price_cents(offer, fw.HISTORY_WEEKS + 1, SEED)


def test_confirmations_are_labelled_fake_and_stable():
    assert fw.confirmation_for("sysco", "R-1-sysco") == fw.confirmation_for("sysco", "R-1-sysco")
    assert fw.confirmation_for("sysco", "R-1-sysco").startswith("FAKE-SYS-")
    assert fw.confirmation_for("sysco", "R-1-sysco") != fw.confirmation_for("sysco", "R-2-sysco")
