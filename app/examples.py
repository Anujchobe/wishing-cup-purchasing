"""Example requests for the "Load example" button and the walkthrough script."""
from __future__ import annotations

from datetime import date, timedelta

EXAMPLES = {
    "factory": ("Hi! For the factory we need 200 lb all-purpose flour, 90 lb cake flour, 180 lb granulated sugar, "
                "130 lb unsalted butter, 2000 fl oz heavy cream, 1440 eggs, 66 lb cream cheese and 44 lb dark "
                "chocolate. Also some saffron if anyone has it. By {when}."),
    "cafe": ("Need 2000 fl oz whole milk, 680 fl oz oat milk, 20 lb espresso beans, 1500 paper hot cups, "
             "1500 lids and 2 lb matcha powder. By {when}."),
}


def example_text(kind: str, today: date) -> str:
    when = today + timedelta(days=8)
    return EXAMPLES[kind].format(when=f"{when:%b} {when.day}")
