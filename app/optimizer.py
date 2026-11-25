"""Plan picker: three ways to split a shopping list across suppliers.

  simple        cheapest supplier for each item, ignoring minimum orders
  one_supplier  the cheapest single supplier that has every item (and meets its minimum)
  best          lowest total including delivery fees, found with PuLP (an integer program)

Which suppliers can take an item ("eligible"): it sells the item, it's in
stock with a price above zero, the supplier answered (is working), serves this
location, and can deliver by the needed-by date. If a supplier has several
SKUs for an item, the one with the lowest real cost is used. Items with no
eligible supplier go on the can't-fill list in every plan.

The best plan, as an integer program:

    minimize   sum cost[i,s] * x[i,s]  +  sum fee[s] * y[s]
    subject to sum_s x[i,s] = 1                     each item goes to one supplier
               x[i,s] <= y[s]                       using a supplier means paying its fee
               sum_i cost[i,s] * x[i,s] >= min[s] * y[s]   a used supplier reaches its minimum
               y[s] <= sum_i x[i,s]                 no fee for a supplier we don't use
               x, y binary

cost[i,s] is the integer cost in cents of the whole cases bought, so the solver
works on exact whole numbers. The model is built as plain data first and only
then handed to PuLP, so it can be checked on its own. The solver's answer is
re-checked in Python and every total is recomputed from integer cents; the
solver's own objective value is never used.

If the solver fails, finds no solution, or returns something that doesn't check
out, the best plan is the better of the one-supplier plan and a simple fallback
(start from the cheapest-per-item plan and drop suppliers that miss their
minimum, smallest first), with a warning.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal

from app import clock
from app.pricing import ItemCost, format_cents, item_cost

SOLVER_TIME_LIMIT_SECONDS = 10


# ------------------------------------------------------------------ inputs

@dataclass(frozen=True)
class Need:
    item: str          # our product code
    qty: Decimal       # in the product's base unit
    name: str = ""

    @property
    def label(self) -> str:
        return self.name or self.item


@dataclass(frozen=True)
class Offer:
    supplier: str
    item: str
    sku: str
    case_size: Decimal  # whole case, in the item's base unit
    case_price_cents: int
    in_stock: bool = True


@dataclass(frozen=True)
class SupplierTerms:
    code: str
    name: str
    min_order_cents: int = 0
    delivery_fee_cents: int = 0
    delivery_weekdays: tuple[str, ...] = clock.WEEKDAYS
    lead_time_days: int = 0
    serves: frozenset[str] | None = None  # location codes; None = everywhere
    working: bool = True                   # False if it didn't answer for quotes

    @classmethod
    def from_config(cls, sup, working: bool = True) -> SupplierTerms:
        return cls(code=sup.code, name=sup.name, min_order_cents=sup.min_order_cents,
                   delivery_fee_cents=sup.delivery_fee_cents, delivery_weekdays=tuple(sup.delivery_weekdays),
                   lead_time_days=sup.lead_time_days, serves=frozenset(sup.serves_locations), working=working)

    def earliest_delivery(self, order_date: date) -> date | None:
        start = order_date + timedelta(days=self.lead_time_days)
        for offset in range(7):
            day = start + timedelta(days=offset)
            if clock.weekday_name(day) in self.delivery_weekdays:
                return day
        return None


@dataclass(frozen=True)
class Problem:
    needs: tuple[Need, ...]
    offers: tuple[Offer, ...]
    suppliers: tuple[SupplierTerms, ...]
    needed_by: date | None = None   # None = no deadline
    order_date: date | None = None  # None = today, Boston time
    location: str | None = None     # None = skip the "serves this location" check

    def __post_init__(self) -> None:
        items = [n.item for n in self.needs]
        if len(set(items)) != len(items):
            raise ValueError("each item may appear only once in a problem; merge duplicate lines first")
        for n in self.needs:
            if isinstance(n.qty, float) or not isinstance(n.qty, Decimal):
                raise TypeError(f"need for {n.item} must be a Decimal")
            if not n.qty.is_finite() or n.qty <= 0:
                raise ValueError(f"need for {n.item} must be more than zero")
        codes = [s.code for s in self.suppliers]
        if len(set(codes)) != len(codes):
            raise ValueError("supplier codes must be unique")
        for o in self.offers:
            if o.supplier not in codes:
                raise ValueError(f"offer {o.sku} is from unknown supplier {o.supplier!r}")
            if isinstance(o.case_size, float) or not isinstance(o.case_size, Decimal) or o.case_size <= 0:
                raise ValueError(f"offer {o.sku}: case size must be a Decimal above zero")
            if isinstance(o.case_price_cents, bool) or not isinstance(o.case_price_cents, int):
                raise TypeError(f"offer {o.sku}: price must be whole cents")


# ------------------------------------------------------------------ outputs

@dataclass(frozen=True)
class PlanLine:
    item: str
    name: str
    supplier: str
    sku: str
    cost: ItemCost

    @property
    def cost_cents(self) -> int:
        return self.cost.cost_cents


@dataclass(frozen=True)
class SupplierList:
    supplier: str
    name: str
    lines: tuple[PlanLine, ...]
    subtotal_cents: int
    delivery_fee_cents: int
    total_cents: int
    min_order_cents: int
    deliver_on: date | None

    @property
    def meets_minimum(self) -> bool:
        return self.subtotal_cents >= self.min_order_cents

    @property
    def short_by_cents(self) -> int:
        return max(0, self.min_order_cents - self.subtotal_cents)


@dataclass(frozen=True)
class CantFill:
    item: str
    name: str
    reasons: tuple[str, ...]

    @property
    def reason(self) -> str:
        return "; ".join(self.reasons)


@dataclass(frozen=True)
class Plan:
    kind: str           # simple | one_supplier | best
    title: str
    lists: tuple[SupplierList, ...]
    cant_fill: tuple[CantFill, ...]
    solved_by: str      # rule | solver | fallback
    warnings: tuple[str, ...] = ()

    @property
    def items_cents(self) -> int:
        return sum(sl.subtotal_cents for sl in self.lists)

    @property
    def delivery_fees_cents(self) -> int:
        return sum(sl.delivery_fee_cents for sl in self.lists)

    @property
    def total_cents(self) -> int:
        return self.items_cents + self.delivery_fees_cents

    @property
    def meets_minimums(self) -> bool:
        return all(sl.meets_minimum for sl in self.lists)

    @property
    def assignment(self) -> dict[str, str]:
        return {line.item: sl.supplier for sl in self.lists for line in sl.lines}


@dataclass(frozen=True)
class Difference:
    item: str
    name: str
    simple_supplier: str
    best_supplier: str
    reason: str


@dataclass(frozen=True)
class PlanSet:
    simple: Plan
    one_supplier: Plan | None
    one_supplier_note: str
    best: Plan
    savings_vs_one_supplier_cents: int | None
    differences: tuple[Difference, ...]
    warnings: tuple[str, ...] = ()


# ------------------------------------------------------------------ solver interface

@dataclass(frozen=True)
class MilpModel:
    """A solver-neutral 0/1 integer program: minimize objective subject to constraints."""

    variables: tuple[str, ...]
    objective: dict[str, int]
    constraints: tuple[tuple[str, dict[str, int], str, int], ...]  # (name, coefficients, sense, rhs)


@dataclass(frozen=True)
class SolveResult:
    values: dict[str, int] | None   # None = no solution
    status: str
    proven_optimal: bool


Solver = Callable[[MilpModel, float], SolveResult]


def solve_with_pulp(model: MilpModel, time_limit: float) -> SolveResult:
    import pulp  # imported here so a broken PuLP install only disables this plan, not the app

    prob = pulp.LpProblem("wishing_cup_best_plan", pulp.LpMinimize)
    var = {name: pulp.LpVariable(name, cat=pulp.LpBinary) for name in model.variables}
    prob += pulp.lpSum(coef * var[name] for name, coef in model.objective.items())
    for name, coeffs, sense, rhs in model.constraints:
        expr = pulp.lpSum(coef * var[v] for v, coef in coeffs.items())
        if sense == "==":
            prob += (expr == rhs, name)
        elif sense == "<=":
            prob += (expr <= rhs, name)
        elif sense == ">=":
            prob += (expr >= rhs, name)
        else:
            raise ValueError(f"unknown constraint sense {sense!r}")
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)
    if not solver.available():  # e.g. PuLP's bundled CBC can't run on this Mac: use an installed one
        installed = pulp.COIN_CMD(msg=False, timeLimit=time_limit)
        if installed.available():
            solver = installed
    status = prob.solve(solver)
    status_text = pulp.LpStatus.get(status, str(status))
    sol_status = getattr(prob, "sol_status", None)
    integer_feasible = getattr(pulp, "LpSolutionIntegerFeasible", 2)
    if status != pulp.LpStatusOptimal and sol_status != integer_feasible:
        return SolveResult(None, status_text, False)
    values = {name: int(round(v.value() or 0)) for name, v in var.items()}
    return SolveResult(values, status_text, proven_optimal=sol_status != integer_feasible)


# ------------------------------------------------------------------ internals

@dataclass(frozen=True)
class _Option:
    offer: Offer
    cost: ItemCost


@dataclass
class _Context:
    problem: Problem
    order_date: date
    terms: dict[str, SupplierTerms]
    rank: dict[str, int]
    needs: dict[str, Need]
    options: dict[str, dict[str, _Option]]   # item -> supplier -> best option
    fillable: list[str]
    cant_fill: tuple[CantFill, ...]
    warnings: list[str] = field(default_factory=list)


def _day(d: date) -> str:
    return f"{d:%a %b} {d.day}"  # %-d isn't available on Windows


def _prepare(problem: Problem) -> _Context:
    order_date = problem.order_date or clock.today()
    terms = {s.code: s for s in problem.suppliers}
    rank = {s.code: n for n, s in enumerate(problem.suppliers)}
    needs = {n.item: n for n in problem.needs}
    options: dict[str, dict[str, _Option]] = {}
    cant_fill = []
    warnings = [f"{s.name} didn't answer, so the plans use the other suppliers."
                for s in problem.suppliers if not s.working]

    for need in problem.needs:
        options[need.item] = {}
        reasons = []
        for s in problem.suppliers:
            offers = [o for o in problem.offers if o.supplier == s.code and o.item == need.item]
            if not offers:
                reasons.append(f"{s.name} doesn't sell it")
                continue
            if not s.working:
                reasons.append(f"{s.name} didn't answer")
                continue
            if problem.location is not None and s.serves is not None and problem.location not in s.serves:
                reasons.append(f"{s.name} doesn't deliver to this location")
                continue
            arrives = s.earliest_delivery(order_date)
            if arrives is None or (problem.needed_by is not None and arrives > problem.needed_by):
                when = _day(arrives) if arrives else "never (no delivery days)"
                reasons.append(f"{s.name} can't deliver until {when}")
                continue
            usable = [o for o in offers if o.in_stock and o.case_price_cents > 0]
            if not usable:
                stocked = [o for o in offers if o.in_stock]
                reasons.append(f"{s.name} has it at an invalid price" if stocked else f"out of stock at {s.name}")
                continue
            costs = [_Option(o, item_cost(need.qty, o.case_size, o.case_price_cents)) for o in usable]
            options[need.item][s.code] = min(costs, key=lambda c: (c.cost.cost_cents, c.offer.sku))
        if not options[need.item]:
            cant_fill.append(CantFill(need.item, need.label, tuple(reasons) or ("no supplier offers it",)))

    fillable = [n.item for n in problem.needs if options[n.item]]
    return _Context(problem, order_date, terms, rank, needs, options, fillable, tuple(cant_fill), warnings)


def _build(ctx: _Context, kind: str, title: str, assignment: dict[str, str], solved_by: str,
           warnings: list[str] | None = None) -> Plan:
    lists = []
    for code in sorted(set(assignment.values()), key=ctx.rank.__getitem__):
        t = ctx.terms[code]
        lines = tuple(
            PlanLine(item, ctx.needs[item].label, code, ctx.options[item][code].offer.sku, ctx.options[item][code].cost)
            for item in ctx.fillable if assignment[item] == code
        )
        subtotal = sum(line.cost_cents for line in lines)
        lists.append(SupplierList(code, t.name, lines, subtotal, t.delivery_fee_cents,
                                  subtotal + t.delivery_fee_cents, t.min_order_cents,
                                  t.earliest_delivery(ctx.order_date)))
    warnings = list(warnings or [])
    if kind == "simple":
        warnings += [f"{sl.name} is {format_cents(sl.short_by_cents)} short of its "
                     f"{format_cents(sl.min_order_cents)} minimum. This plan ignores minimums."
                     for sl in lists if not sl.meets_minimum]
    return Plan(kind, title, tuple(lists), ctx.cant_fill, solved_by, tuple(warnings))


def _cheapest(ctx: _Context, item: str, allowed: set[str] | None = None) -> str | None:
    choices = [s for s in ctx.options[item] if allowed is None or s in allowed]
    if not choices:
        return None
    return min(choices, key=lambda s: (ctx.options[item][s].cost.cost_cents, ctx.rank[s]))


def _subtotals(ctx: _Context, assignment: dict[str, str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for item, s in assignment.items():
        out[s] = out.get(s, 0) + ctx.options[item][s].cost.cost_cents
    return out


def _valid(ctx: _Context, assignment: dict[str, str]) -> bool:
    if set(assignment) != set(ctx.fillable) or any(s not in ctx.options[i] for i, s in assignment.items()):
        return False
    return all(sub >= ctx.terms[s].min_order_cents for s, sub in _subtotals(ctx, assignment).items())


# ------------------------------------------------------------------ the three plans

def plan_simple(ctx: _Context) -> Plan:
    assignment = {item: _cheapest(ctx, item) for item in ctx.fillable}
    return _build(ctx, "simple", "Cheapest per item (ignores minimums)", assignment, "rule")


def plan_one_supplier(ctx: _Context) -> tuple[Plan | None, str]:
    if not ctx.fillable:
        return None, "Nothing to buy."
    complete = [s.code for s in ctx.problem.suppliers if all(s.code in ctx.options[i] for i in ctx.fillable)]
    if not complete:
        return None, "No single supplier can supply every item."
    candidates = []
    short_notes = []
    for code in complete:
        t = ctx.terms[code]
        subtotal = sum(ctx.options[i][code].cost.cost_cents for i in ctx.fillable)
        if subtotal < t.min_order_cents:
            short_notes.append(f"{t.name} has every item, but the order would be {format_cents(subtotal)}, "
                               f"below its {format_cents(t.min_order_cents)} minimum.")
            continue
        candidates.append((subtotal + t.delivery_fee_cents, ctx.rank[code], code))
    if not candidates:
        return None, " ".join(short_notes)
    _, _, code = min(candidates)
    plan = _build(ctx, "one_supplier", f"One supplier ({ctx.terms[code].name})",
                  {i: code for i in ctx.fillable}, "rule")
    return plan, f"{ctx.terms[code].name} is the cheapest single supplier with every item."


def build_model(ctx: _Context) -> tuple[MilpModel, dict[tuple[str, str], str], dict[str, str]]:
    item_no = {item: n for n, item in enumerate(ctx.fillable)}
    used = sorted({s for i in ctx.fillable for s in ctx.options[i]}, key=ctx.rank.__getitem__)
    x = {(i, s): f"x_{item_no[i]}_{ctx.rank[s]}" for i in ctx.fillable for s in ctx.options[i]}
    y = {s: f"y_{ctx.rank[s]}" for s in used}
    cost = {(i, s): ctx.options[i][s].cost.cost_cents for (i, s) in x}

    objective = {x[k]: cost[k] for k in x}
    for s in used:
        objective[y[s]] = ctx.terms[s].delivery_fee_cents
    constraints = []
    for i in ctx.fillable:
        constraints.append((f"assign_{item_no[i]}", {x[(i, s)]: 1 for s in ctx.options[i]}, "==", 1))
    for (i, s), name in x.items():
        constraints.append((f"link_{item_no[i]}_{ctx.rank[s]}", {name: 1, y[s]: -1}, "<=", 0))
    for s in used:
        mine = [(i, s) for i in ctx.fillable if (i, s) in x]
        constraints.append((f"minimum_{ctx.rank[s]}",
                            {**{x[k]: cost[k] for k in mine}, y[s]: -ctx.terms[s].min_order_cents}, ">=", 0))
        constraints.append((f"used_{ctx.rank[s]}", {**{x[k]: -1 for k in mine}, y[s]: 1}, "<=", 0))
    model = MilpModel(tuple(list(x.values()) + list(y.values())), objective, tuple(constraints))
    return model, x, y


def _greedy_fallback(ctx: _Context) -> dict[str, str] | None:
    allowed = {s for i in ctx.fillable for s in ctx.options[i]}
    while True:
        assignment = {}
        for item in ctx.fillable:
            choice = _cheapest(ctx, item, allowed)
            if choice is None:
                return None
            assignment[item] = choice
        subtotals = _subtotals(ctx, assignment)
        short = [s for s, sub in subtotals.items() if sub < ctx.terms[s].min_order_cents]
        if not short:
            return assignment
        allowed.discard(min(short, key=lambda s: (subtotals[s], ctx.rank[s])))


def plan_best(ctx: _Context, one_supplier: Plan | None, solve: Solver, time_limit: float) -> Plan:
    title = "Best overall"
    if not ctx.fillable:
        return _build(ctx, "best", title, {}, "rule")

    model, x, _ = build_model(ctx)
    problem_note = None
    try:
        result = solve(model, time_limit)
    except Exception as exc:  # solver missing, crashed, or couldn't run
        result, problem_note = None, f"the optimizer couldn't run ({type(exc).__name__}: {exc})"

    if result is not None and result.values is not None:
        assignment = {i: s for (i, s), name in x.items() if result.values.get(name) == 1}
        if len(assignment) == len(ctx.fillable) and _valid(ctx, assignment):
            warnings = [] if result.proven_optimal else [
                f"The optimizer hit its {time_limit:g}-second limit. This is the best plan it found, "
                "but it may not be the cheapest possible."]
            return _build(ctx, "best", title, assignment, "solver", warnings)
        problem_note = "the optimizer's answer didn't pass the checks"
    elif result is not None:
        problem_note = (f"the optimizer found no plan (status: {result.status}); "
                        "most likely no combination meets every supplier's minimum")

    # Fallback: the better of the one-supplier plan and the greedy plan that respect minimums.
    candidates = []
    if one_supplier is not None and one_supplier.meets_minimums:
        candidates.append((one_supplier.total_cents, 0, one_supplier.assignment, "the one-supplier plan"))
    greedy = _greedy_fallback(ctx)
    if greedy is not None:
        g_total = sum(_subtotals(ctx, greedy).values()) + sum(ctx.terms[s].delivery_fee_cents for s in set(greedy.values()))
        candidates.append((g_total, 1, greedy, "a simple fallback plan"))
    if candidates:
        _, _, assignment, source = min(candidates, key=lambda c: (c[0], c[1]))
        warning = f"Using {source} because {problem_note}. It meets every minimum but may not be the cheapest."
    else:
        assignment = {item: _cheapest(ctx, item) for item in ctx.fillable}
        warning = (f"No plan meets every supplier's minimum order ({problem_note}). This plan shows the "
                   "cheapest prices; it can't be ordered until the minimums are met.")
    return _build(ctx, "best", title, assignment, "fallback", [warning])


def _explain(ctx: _Context, best: Plan, item: str, a_sup: str, b_sup: str) -> str:
    """Why the best plan buys `item` from b_sup instead of the cheapest-per-item a_sup.

    Tests the single move "buy this item from a_sup instead" against the best plan.
    """
    A, B = ctx.terms[a_sup], ctx.terms[b_sup]
    a_cost = ctx.options[item][a_sup].cost.cost_cents
    b_cost = ctx.options[item][b_sup].cost.cost_cents
    saving = b_cost - a_cost
    subtotals = _subtotals(ctx, best.assignment)
    a_used = a_sup in subtotals
    a_after = subtotals.get(a_sup, 0) + a_cost
    b_after = subtotals[b_sup] - b_cost

    cheaper = f"{A.name} is {format_cents(saving)} cheaper for this item" if saving else \
        f"{A.name} charges the same for this item"
    if not a_used and a_after < A.min_order_cents:
        return (f"{cheaper}, but an order of just this item at {A.name} would be {format_cents(a_after)}, "
                f"below its {format_cents(A.min_order_cents)} minimum.")
    if 0 < b_after < B.min_order_cents:
        return (f"{cheaper}, but moving it would leave {B.name} at {format_cents(b_after)}, "
                f"below its {format_cents(B.min_order_cents)} minimum.")
    added_fee = 0 if a_used else A.delivery_fee_cents
    dropped_fee = B.delivery_fee_cents if b_after == 0 else 0
    change = -saving + added_fee - dropped_fee
    if change > 0:
        fee_text = f"adds {A.name}'s {format_cents(added_fee)} delivery fee"
        if dropped_fee:
            fee_text += f" (and drops {B.name}'s {format_cents(dropped_fee)} fee)"
        return f"{cheaper}, but buying it there {fee_text}, so the total would go up by {format_cents(change)}."
    if change == 0:
        return f"Buying it from {A.name} instead would cost exactly the same overall. Either is fine."
    return (f"Buying it from {A.name} would save {format_cents(-change)}. This plan came from the fallback, "
            "not the optimizer, so it isn't the cheapest.")


def make_plans(problem: Problem, *, solve: Solver | None = None,
               time_limit: float = SOLVER_TIME_LIMIT_SECONDS) -> PlanSet:
    """All three plans, the savings of best vs one supplier, and why best differs from simple."""
    ctx = _prepare(problem)
    solve = solve or solve_with_pulp  # looked up at call time, so tests can swap it
    simple = plan_simple(ctx)
    one, note = plan_one_supplier(ctx)
    best = plan_best(ctx, one, solve, time_limit)

    savings = one.total_cents - best.total_cents if one is not None else None
    a, c = simple.assignment, best.assignment
    differences = tuple(
        Difference(item, ctx.needs[item].label, a[item], c[item], _explain(ctx, best, item, a[item], c[item]))
        for item in ctx.fillable if a[item] != c[item]
    )
    return PlanSet(simple, one, note, best, savings, differences, tuple(ctx.warnings))


def solver_check() -> tuple[bool, str]:
    """Solve a one-item problem to find out, at startup, whether PuLP/CBC works on this machine."""
    tiny = Problem((Need("x", Decimal(1)),), (Offer("a", "x", "a-x", Decimal(1), 100),),
                   (SupplierTerms("a", "A"),), order_date=date(2026, 1, 5))
    best = make_plans(tiny).best
    if best.solved_by == "solver":
        return True, "PuLP/CBC is working."
    return False, f"PuLP/CBC is not working, so the best plan will use the fallback. {best.warnings[0]}"
