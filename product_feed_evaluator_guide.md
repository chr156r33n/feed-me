# 🧠 Product Feed Evaluation Agent — Full Instruction Guide

## 🎯 Objective
This tool evaluates **Google Shopping product feed quality** by checking how well each product entry answers typical buyer questions.

The outcome is a **CSV file** listing:
- Product URL  
- Combined product context (from selected fields)  
- List of inferred buyer questions  
- Judgements on how well each question is answered  
- A % score showing overall coverage  

This helps identify **content gaps** in your feed for optimization.

---

## 🧩 Core Workflow

### 1. Input
1. **Upload** a Google Shopping XML feed file or **paste** its public URL.  
2. **Specify** which fields to use for product “context” — for example:
   ```
   title, description, brand, price, color, size, material, availability
   ```
   > The field `link` (product URL) is always required.
3. Optionally define how many questions to generate per product (default: 12).

---

### 2. Pre-Processing
Before sending any data to GPT:
- Parse XML and extract product-level data.
- Clean text (strip HTML tags, excess whitespace, tracking parameters).
- De-duplicate by main URL (ignore variant duplicates unless unique URL).
- For products with `item_group_id`, prefer parent entries.
- Combine the selected fields into a **context string**:

```
Title: {title} | Brand: {brand} | Price: {price} | Description: {description} | Features: {features}
```

---

## 🤖 Model Interaction

The agent uses **two prompts** per unique product or product-type:

### A. Generate Questions
Use this prompt first — ideally reuse for all items sharing the same category or type to save cost.

```
System: You evaluate whether product feed data is sufficient for a buyer.

User:
Given the product summary below, list up to {N} distinct buyer questions a careful shopper would ask BEFORE purchase.

Cover: specs, sizing/fit, compatibility, use/setup, care, materials, safety, warranty/returns, delivery, sustainability, what's-in-the-box.

Return ONLY a JSON array of strings.

PRODUCT SUMMARY:
{title}
{brand} | {product_type}
{short_context_text}
```

Output must be a valid JSON array like:
```json
[
  "What material is the product made from?",
  "Is it suitable for outdoor use?",
  "Does it come with a warranty?"
]
```

---

### B. Judge Answers
Take the question list from (A) and assess if the feed content answers them.

```
System: You strictly judge if the product SUMMARY answers each buyer question.

User:
For each question, output an object:
{
  "question": str,
  "verdict": "yes"|"partial"|"no",
  "reason": str (<= 20 words)
}

Verdict rules:
- "yes" → explicitly and clearly answered.
- "partial" → hinted but incomplete or vague.
- "no" → not addressed.

Return JSON array only.

SUMMARY:
{name_plus_context}

QUESTIONS:
{questions_json}
```

Expected output example:
```json
[
  {"question": "What material is the product made from?", "verdict": "yes", "reason": "Described as stainless steel."},
  {"question": "Does it come with a warranty?", "verdict": "no", "reason": "No warranty details included."}
]
```

---

## 🧮 Scoring Rules

| Verdict | Score Value |
|----------|--------------|
| Yes      | 1.0 |
| Partial  | 0.5 |
| No       | 0.0 |

**Coverage % = (Yes + 0.5 × Partial) ÷ Total × 100**

Round to 1 decimal place.

---

## 🗂️ Output Schema

Each product’s output row includes:

| Field | Description |
|--------|-------------|
| `product_url` | The main `link` field. |
| `name_plus_context` | Concatenated fields (title, description, etc.). |
| `questions_json` | JSON array of inferred buyer questions. |
| `judgements_json` | JSON array of verdicts and reasons. |
| `yes_count` | Count of “yes” answers. |
| `partial_count` | Count of “partial” answers. |
| `no_count` | Count of “no” answers. |
| `coverage_pct` | Calculated % coverage score. |
| `missing_core_fields` | Comma list of absent essentials (`brand`, `gtin`, `price`, etc.). |

Output file name format:
```
feed_analysis_{YYYYMMDD}.csv
```

---

## ⚙️ Recommended Field Policy

### Always Required
- `link` (main URL)
- `title`
- `description`
- `image_link`

### Strongly Recommended
- `brand`
- `gtin` or `mpn`
- `price`
- `availability`
- `color`
- `size`
- `material`
- `product_type`
- `item_group_id`

---

## 🚦 Quality Checks (Optional, Low-Cost)

Before running prompts, flag obvious issues:

| Check | Description |
|--------|--------------|
| Missing essential fields | `brand`, `gtin`, `price`, `availability`, `image_link` |
| Title too short/long | `<20` or `>150` characters |
| Description too short | `<60` characters |
| Description too long | `>5000` characters |
| Contains ALL CAPS or emojis | Poor quality indicator |
| Broken image links | Invalid URL format or missing extension |

Add these as warning columns in the CSV.

---

## 🧠 Optimization & Efficiency

- **Question reuse:** Cache generated questions per product-type (`brand + category`) to reduce API calls.
- **Token trimming:** Cut long descriptions to ~1000 tokens.
- **Temperature:** `0` (for deterministic results).
- **Re-ask** once if invalid JSON is returned.
- **Batching:** Evaluate in groups of 20–50 products for performance.

---

## 💰 Cost & Sampling

- Run on a **sample of 200 items** first to verify usefulness.
- If question overlap is high, switch to category-level caching.
- Consider `gpt-4-turbo` for accuracy or `gpt-4o-mini` for speed.

---

## 🧾 Example Output (CSV)

| product_url | name_plus_context | questions_json | judgements_json | yes_count | partial_count | no_count | coverage_pct |
|--------------|------------------|----------------|-----------------|------------|----------------|-----------|----------------|
| https://store.com/p/123 | "Title: Modern Desk Lamp | Brand: Lumina | Price: £59 | Description: Metal lamp..." | `["What material...", "Does it have a dimmer?"]` | `[{"q":"What material","v":"yes"}, {"q":"Does it have a dimmer?","v":"no"}]` | 1 | 0 | 1 | 50.0 |

---

## 🧰 Optional Extensions
If you want to evolve the tool later:
- Add **topic classification** for which question types fail most often (e.g., “fit,” “warranty”).
- Create **category dashboards** for average coverage.
- Link flagged products back to the CMS or PIM for editing.
- Add **Readability/clarity** grading for title/description.

---

## 🧑‍💻 Agent Responsibilities
- Keep the system deterministic and light.
- Enforce valid JSON parsing.
- Handle feed parsing gracefully (UTF-8 only).
- Save output locally as CSV.
- Never store or send product data externally beyond OpenAI API calls.

---

## ✅ Success Criteria
- CSV outputs cleanly with expected fields.
- No empty product rows.
- Each product has a question list and grading results.
- Average coverage % matches human sense of completeness.
- Tool runs on 200+ SKUs without timing out.

---

## 📦 Minimal Directory Structure Example
```
product-feed-evaluator/
│
├── app.py                   # Streamlit or CLI entry point
├── prompts.py               # Holds both prompt templates
├── feed_parser.py           # XML parsing logic
├── evaluator.py             # GPT call + scoring
├── requirements.txt
├── sample_feed.xml
└── output/
    └── feed_analysis_20251007.csv
```

---

## 🧭 TL;DR Summary
1. **Input feed → pick fields → clean → concatenate context.**  
2. **Prompt 1:** Generate buyer questions.  
3. **Prompt 2:** Grade whether product info answers them.  
4. **Score coverage and export CSV.**  
5. **Review low-score products for feed improvement.**

That’s the entire loop — simple, explainable, and fast.
