"""
Prompt templates for the Product Feed Evaluation Agent (v2).
Drop-in replacement for your existing prompt.py with stricter schemas and a QA pass.

Provided constants:
- QUESTION_GENERATION_PROMPT
- QUESTION_QA_PROMPT (new)
- ANSWER_JUDGEMENT_PROMPT

Placeholders expected in f-string formatting:
- {N}               # max number of questions to return
- {locale}          # e.g., 'en-GB'
- {title}
- {brand}
- {product_type}
- {short_context_text}
- {candidate_questions_json}  # JSON from generation step
- {name_plus_context}         # full concatenated product summary
- {final_questions_json}      # JSON from QA step
"""

QUESTION_GENERATION_PROMPT = """System: You design buyer-relevant questions that can be answered (or plausibly expected) from product feed data.

User:
Generate up to {N} HIGH-VALUE questions for this product. Questions must be specific to the product/category, not generic store policy FAQs.

Rules:
- Map each question to ONE taxonomy bucket from this fixed list:
  ["Specs","Fit","Compatibility","Use","Care","Materials","Safety","Warranty","Delivery","Returns","InBox","Sustainability","Certification","Ingredients"].
- Write concise, unambiguous questions (<= 18 words).
- Prefer attribute-seeking forms (units, ranges, inclusions, capacities, contents).
- Localize units/terms to {locale}.
- If the summary already answers a question explicitly, you may still include it; grading will score it as answered.
- Do NOT include storewide policy questions unless the summary explicitly mentions them.
- If product_type is niche, bias toward category-specific attributes.

Return ONLY a JSON array of objects with this schema:
[
  { "question": string, "taxonomy": string }
]

CONTEXT:
- title: {title}
- brand: {brand}
- product_type: {product_type}
- locale: {locale}
- summary: {short_context_text}
"""


QUESTION_QA_PROMPT = """System: You are a ruthless editor. You normalize, dedupe, and keep only the most decision-critical questions for the product.

User:
You are given:
A) product summary (below)
B) candidate questions (JSON array of {question, taxonomy})
C) max_n = {N}

Tasks:
1) Normalize wording (canonicalize synonyms; e.g., include/come with/contains -> "What's included").
2) Remove near-duplicates (>= 0.85 semantic similarity) and trivial variants.
3) Reject generic questions that could apply to any product without the category noun; rewrite to be category-specific when possible.
4) Mark required=true for questions that commonly block purchase decisions for this category:
   - Always required where applicable: Specs core (size/dimensions/capacity), InBox, Materials, Fit/Compatibility, Warranty/Returns if mentioned, Safety/Ingredients if applicable.
5) Rank by impact: required first, then remaining by likely buyer impact given the summary.
6) Truncate to max_n.

Return ONLY a JSON array with this schema:
[
  { "question": string, "taxonomy": string, "required": boolean }
]

CONTEXT:
- title: {title}
- brand: {brand}
- product_type: {product_type}
- locale: {locale}
- summary: {short_context_text}

CANDIDATE_QUESTIONS_JSON:
{candidate_questions_json}
"""


ANSWER_JUDGEMENT_PROMPT = """System: You strictly judge if the SUMMARY answers each buyer question. Use ONLY the SUMMARY; do not invent facts.

User:
For each question, output an object:
{
  "question": str,
  "taxonomy": str,
  "required": boolean,
  "verdict": "yes"|"partial"|"no",
  "reason": str  // <= 16 words; if verdict is "yes", quote the exact supporting phrase
}

Verdict rules:
- "yes": explicit, unambiguous answer with units/specs if applicable (quote brief phrase).
- "partial": hinted, missing units, vague, or incomplete.
- "no": not present in the SUMMARY.

Return ONLY a JSON array.

SUMMARY:
{name_plus_context}

QUESTIONS:
{final_questions_json}
"""
