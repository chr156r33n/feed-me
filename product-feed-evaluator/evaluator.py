"""
Core evaluation logic with GPT interaction and scoring for the Product Feed Evaluation Agent.
"""

import json
import time
from typing import List, Dict, Any, Optional
import openai
from prompts import QUESTION_GENERATION_PROMPT, ANSWER_JUDGEMENT_PROMPT
import pandas as pd


class ProductFeedEvaluator:
    """Handles GPT interactions and scoring for product feed evaluation."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", temperature: float = 0.0):
        """
        Initialize the evaluator.
        
        Args:
            api_key: OpenAI API key
            model: GPT model to use
            temperature: Model temperature (0 for deterministic results)
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.question_cache = {}  # Cache for question reuse by product type
    
    def generate_questions(self, product_context: str, product_type: str, brand: str, 
                         num_questions: int = 12) -> List[str]:
        """
        Generate buyer questions for a product.
        
        Args:
            product_context: Combined product context string
            product_type: Product type/category
            brand: Product brand
            num_questions: Number of questions to generate
            
        Returns:
            List of buyer questions
        """
        # Create cache key for question reuse
        cache_key = f"{brand}_{product_type}"
        
        if cache_key in self.question_cache:
            return self.question_cache[cache_key]
        
        # Extract basic info from context
        title = self._extract_field(product_context, "Title")
        brand_info = self._extract_field(product_context, "Brand")
        product_type_info = self._extract_field(product_context, "Product_Type")
        
        prompt = QUESTION_GENERATION_PROMPT.format(
            N=num_questions,
            title=title or "Product",
            brand=brand_info or brand or "Unknown",
            product_type=product_type_info or product_type or "General",
            short_context_text=product_context[:1000]  # Limit context length
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You evaluate whether product feed data is sufficient for a buyer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=1000
            )
            
            questions_text = response.choices[0].message.content.strip()
            questions = self._parse_json_array(questions_text)
            
            # Cache the questions for reuse
            self.question_cache[cache_key] = questions
            
            return questions
            
        except Exception as e:
            print(f"Error generating questions: {e}")
            return self._get_fallback_questions()
    
    def judge_answers(self, product_context: str, questions: List[str]) -> List[Dict[str, Any]]:
        """
        Judge whether product context answers the buyer questions.
        
        Args:
            product_context: Combined product context string
            questions: List of buyer questions
            
        Returns:
            List of judgement dictionaries
        """
        prompt = ANSWER_JUDGEMENT_PROMPT.format(
            name_plus_context=product_context[:2000],  # Limit context length
            questions_json=json.dumps(questions)
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You strictly judge if the product SUMMARY answers each buyer question."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=2000
            )
            
            judgements_text = response.choices[0].message.content.strip()
            judgements = self._parse_json_array(judgements_text)
            
            # Validate judgements format
            validated_judgements = []
            for judgement in judgements:
                if isinstance(judgement, dict) and all(key in judgement for key in ['question', 'verdict', 'reason']):
                    validated_judgements.append(judgement)
                else:
                    # Create fallback judgement
                    question = judgement.get('question', 'Unknown question') if isinstance(judgement, dict) else str(judgement)
                    validated_judgements.append({
                        'question': question,
                        'verdict': 'no',
                        'reason': 'Unable to parse judgement'
                    })
            
            return validated_judgements
            
        except Exception as e:
            print(f"Error judging answers: {e}")
            return self._get_fallback_judgements(questions)
    
    def calculate_coverage_score(self, judgements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate coverage score from judgements.
        
        Args:
            judgements: List of judgement dictionaries
            
        Returns:
            Dictionary with scoring results
        """
        yes_count = sum(1 for j in judgements if j.get('verdict') == 'yes')
        partial_count = sum(1 for j in judgements if j.get('verdict') == 'partial')
        no_count = sum(1 for j in judgements if j.get('verdict') == 'no')
        
        total = len(judgements)
        if total == 0:
            coverage_pct = 0.0
        else:
            coverage_pct = round((yes_count + 0.5 * partial_count) / total * 100, 1)
        
        return {
            'yes_count': yes_count,
            'partial_count': partial_count,
            'no_count': no_count,
            'coverage_pct': coverage_pct
        }
    
    def evaluate_product(self, product_data: Dict[str, Any], selected_fields: List[str], 
                       num_questions: int = 12) -> Dict[str, Any]:
        """
        Complete evaluation for a single product.
        
        Args:
            product_data: Product dictionary
            selected_fields: Fields used in context
            num_questions: Number of questions to generate
            
        Returns:
            Evaluation results dictionary
        """
        # Create product context
        context_parts = []
        for field in selected_fields:
            if field in product_data and product_data[field]:
                context_parts.append(f"{field.title()}: {product_data[field]}")
        
        product_context = " | ".join(context_parts)
        
        # Get product type and brand for caching
        product_type = product_data.get('product_type', 'General')
        brand = product_data.get('brand', 'Unknown')
        
        # Generate questions
        questions = self.generate_questions(
            product_context, product_type, brand, num_questions
        )
        
        # Judge answers
        judgements = self.judge_answers(product_context, questions)
        
        # Calculate scores
        scores = self.calculate_coverage_score(judgements)
        
        # Identify missing core fields
        missing_core_fields = []
        core_fields = ['brand', 'gtin', 'mpn', 'price', 'availability']
        for field in core_fields:
            if not product_data.get(field):
                missing_core_fields.append(field)
        
        return {
            'product_url': product_data.get('link', ''),
            'name_plus_context': product_context,
            'questions_json': json.dumps(questions),
            'judgements_json': json.dumps(judgements),
            'missing_core_fields': ', '.join(missing_core_fields),
            **scores
        }
    
    def evaluate_products_batch(self, products_df: pd.DataFrame, selected_fields: List[str],
                              num_questions: int = 12, batch_size: int = 20,
                              on_progress: Optional[Any] = None, debug: bool = False) -> pd.DataFrame:
        """
        Evaluate a batch of products with rate limiting.
        
        Args:
            products_df: DataFrame of products
            selected_fields: Fields to include in context
            num_questions: Number of questions per product
            batch_size: Number of products to process in each batch
            
        Returns:
            DataFrame with evaluation results
        """
        results = []
        total_products = len(products_df)
        
        for i in range(0, total_products, batch_size):
            batch = products_df.iloc[i:i + batch_size]
            if debug:
                print(f"Processing batch {i//batch_size + 1}/{(total_products-1)//batch_size + 1}")
            
            processed_in_loop = 0
            for _, product in batch.iterrows():
                try:
                    result = self.evaluate_product(
                        product.to_dict(), selected_fields, num_questions
                    )
                    results.append(result)
                    processed_in_loop += 1
                    if on_progress is not None:
                        on_progress({
                            "processed": min(i + processed_in_loop, total_products),
                            "total": total_products,
                            "current_product_url": result.get("product_url", ""),
                            "message": f"Processed: {result.get('product_url', '')}"
                        })
                    
                    # Rate limiting - small delay between API calls
                    time.sleep(0.1)
                    
                except Exception as e:
                    if debug:
                        print(f"Error evaluating product {product.get('link', 'unknown')}: {e}")
                    # Add error result
                    results.append({
                        'product_url': product.get('link', ''),
                        'name_plus_context': str(product.get('name_plus_context', '')),
                        'questions_json': '[]',
                        'judgements_json': '[]',
                        'missing_core_fields': '',
                        'yes_count': 0,
                        'partial_count': 0,
                        'no_count': 0,
                        'coverage_pct': 0.0
                    })
                    processed_in_loop += 1
                    if on_progress is not None:
                        on_progress({
                            "processed": min(i + processed_in_loop, total_products),
                            "total": total_products,
                            "current_product_url": product.get('link', ''),
                            "message": f"Error on: {product.get('link', '')} -> {e}"
                        })
            
            # Longer delay between batches
            if i + batch_size < total_products:
                time.sleep(2)
        
        return pd.DataFrame(results)
    
    def _extract_field(self, context: str, field_name: str) -> Optional[str]:
        """Extract a field value from context string."""
        pattern = f"{field_name}: ([^|]+)"
        match = re.search(pattern, context)
        return match.group(1).strip() if match else None
    
    def _parse_json_array(self, text: str) -> List[Any]:
        """Parse JSON array from text, with fallback handling."""
        try:
            # Try to find JSON array in the text
            start = text.find('[')
            end = text.rfind(']') + 1
            if start != -1 and end > start:
                json_text = text[start:end]
                return json.loads(json_text)
            else:
                # Try parsing the entire text
                return json.loads(text)
        except json.JSONDecodeError:
            # Fallback: try to extract array-like content
            lines = text.strip().split('\n')
            items = []
            for line in lines:
                line = line.strip()
                if line.startswith('"') and line.endswith('"'):
                    items.append(line[1:-1])  # Remove quotes
                elif line.startswith("'") and line.endswith("'"):
                    items.append(line[1:-1])  # Remove quotes
            return items
    
    def _get_fallback_questions(self) -> List[str]:
        """Fallback questions when API fails."""
        return [
            "What material is the product made from?",
            "What are the key specifications?",
            "Is it suitable for my intended use?",
            "What's included in the box?",
            "What are the dimensions and weight?",
            "Is there a warranty or return policy?",
            "How do I care for this product?",
            "Is it compatible with other products?",
            "What are the safety considerations?",
            "How is it delivered and installed?",
            "Is it environmentally friendly?",
            "What are the alternatives or similar products?"
        ]
    
    def _get_fallback_judgements(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Fallback judgements when API fails."""
        return [
            {
                'question': question,
                'verdict': 'no',
                'reason': 'Unable to evaluate due to API error'
            }
            for question in questions
        ]


# Add regex import at the top
import re