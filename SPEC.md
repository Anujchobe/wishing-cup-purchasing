Build a small app that runs on my laptop. It helps a cafe and dessert company (Wishing Cup, Boston) buy ingredients from suppliers at the best price. Save this whole message as SPEC.md in the project folder. Then do only Step 1.

WHAT THE APP DOES
1. A manager types what they need, in plain words or in a form.
2. The app gets prices from 4 suppliers.
3. It works out the real cost of each option and picks the best plan.
4. It makes a shopping list for each supplier.
5. The owner checks the lists, can change them, and approves.
6. The app places the orders.
7. It saves everything so we can see reports later.

MUST HAVE
- Free. Runs on my laptop. No paid hosting.
- Starts with two commands: pip install -r requirements.txt, then python run.py. Works on Mac, Linux and Windows.
- Works with no AI key. With no key, the AI parts use simple text matching instead.
- The AI model can be changed by editing only the .env file (LLM_BASE_URL, LLM_API_KEY, LLM_MODEL). Use the OpenAI-style API so most providers work.
- Supports more than one location: 1 cafe and 1 factory now, about 10 cafes later.
- Supplier prices are fake test data. Label them as fake in the app. Do not scrape any website.
- Use a fixed random seed so the numbers are the same every time.
- After every step, the app still runs and all tests pass.

TOOLS
Python 3.11, Streamlit, SQLite with SQLModel, Pydantic, rapidfuzz, PuLP, openai, PyYAML, pytest, hypothesis. Pin exact versions in requirements.txt.

RULES
- Use AI only for messy text: reading the manager's request, matching an ingredient to the right supplier product, reading pack sizes. Code does all math and unit conversions, picks the plan, and places orders. A person approves.
- Store money as whole cents (integers), never floats. Store quantities with Decimal. Use Boston time for dates.
- Check every AI answer against a Pydantic model. The AI can only pick product codes from the list we give it. If the answer is bad, try once more. If it is still bad, use text matching and mark the item for a person to check.
- Compare the cost of what we actually have to buy. Suppliers sell full cases. Cases to buy = need ÷ case size, rounded up. Cost = cases × case price. Show how much extra we get.
- Save confirmed matches. Only ask the AI about products we have not seen before.

SAFETY CHECKS
- Before approval and again before ordering, check that:
  - every item is on exactly one supplier list or marked "can't fill"
  - the cases cover what is needed
  - totals add up from the lines
  - each supplier's minimum order is met
  - each supplier can deliver by the needed date
  - the products exist and are in stock
  - prices are not older than the limit
  If any check fails, block and say why in plain words.
- If a supplier does not answer: wait a short time, try 2 more times, then make a plan from the other suppliers and show a warning.
- Save each order as PENDING before contacting the supplier. Give each order a unique ID (request id + supplier id). A supplier that gets the same ID twice returns the same confirmation. When the app starts, finish any PENDING orders. An order must never be placed twice.
- If a price is 0 or less, or jumped more than 50% since last week, don't use it. Mark it for a person to check.
- If a quantity is more than 3 times the most this location ever ordered of that item, ask the person to confirm.
- If the current AI model has not passed the test cases (score below eval_threshold), every AI answer needs a person to confirm, and the app shows "model not tested".
- Add on/off switches in config to fake problems: a supplier is down, a supplier is slow, a supplier fails sometimes, the AI is down.

EXTRA FEATURES
- Savings replay: take the last 8 weeks of past orders, use the prices from those weeks, and show how much this app would have saved.
- Price-match email: when another supplier is cheaper than our main supplier by more than match_threshold, write an email to the main supplier's sales rep asking them to match. Code gives the numbers; AI only writes the wording; with no AI, use a template. A person sends it.
- Price watch: show when an item went up more than 10% at every supplier (market change) versus when only one supplier raised it.

DATABASE TABLES
- locations
- suppliers: min_order, delivery_fee, delivery_weekdays, lead_time_days, serves_locations
- products: our ingredient list, base unit
- supplier_items: sku, description, raw pack text, case_price_cents, in_stock
- price_snapshots: with a "don't use" flag
- item_mappings: our product to supplier sku, pack size in base units, confidence, source (ai / text match / person), needs_check
- requests and request_lines: location, needed-by date, original text, status
- quotes
- orders: one per request and supplier, status, confirmation id
- llm_calls: model, task, time taken, worked or not
- audit_log

FOLDERS
app/{db,models,units,pricing,optimizer,matching,intake,workflow,checks,po,insights}.py
app/suppliers/{base,mock}.py   app/llm/client.py   app/llm/prompts/
ui/app.py   run.py   scripts/seed.py   evals/   tests/   config.yaml   .env.example   README.md

STEPS
1 Project setup, config, database, fake data, run.py
2 Units, pricing, plan picker, tests
3 AI client, reading requests, matching products, model check
4 Order flow, safety checks, placing orders, purchase orders, log
5 Streamlit screens and reports
6 AI test cases, GitHub tests, README

STEP 1 (do this now)
- config.yaml:
  - suppliers Sysco, US Foods, PFG, and "Supplier 4 (placeholder)", each with a different minimum order, delivery fee, delivery days, lead time, and locations served
  - all fake problems off
  - main supplier: Sysco; match_threshold: 0.05
  - locations: "Court St Cafe" and "Factory"
  - approval: owner approves; prices expire after 24 hours
  - eval_threshold: 0.9
- Supplier base class with three functions: get_catalog(), get_quotes(skus), place_order(order, order_id), which returns a confirmation id.
- 4
