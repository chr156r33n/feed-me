"""
Prompt templates for the Product Feed Evaluation Agent.
"""

QUESTION_GENERATION_PROMPT = """System: You evaluate whether product feed data is sufficient for a buyer.

User:
Given the product summary below, list up to {N} distinct buyer questions a careful shopper would ask BEFORE purchase.

Cover: specs, sizing/fit, compatibility, use/setup, care, materials, safety, warranty/returns, delivery, sustainability, what's-in-the-box.

Return ONLY a JSON array of strings.

PRODUCT SUMMARY:
{title}
{brand} | {product_type}
{short_context_text}"""

ANSWER_JUDGEMENT_PROMPT = """System: You strictly judge if the product SUMMARY answers each buyer question.

User:
For each question, output an object:
{{
  "question": str,
  "verdict": "yes"|"partial"|"no",
  "reason": str (<= 20 words)
}}

Verdict rules:
- "yes" → explicitly and clearly answered.
- "partial" → hinted but incomplete or vague.
- "no" → not addressed.

Return JSON array only.

SUMMARY:
{name_plus_context}

QUESTIONS:
{questions_json}"""