"""
Command-line interface for the Product Feed Evaluation Agent.
"""

import argparse
import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from feed_parser import FeedParser
from evaluator import ProductFeedEvaluator


def main():
    parser = argparse.ArgumentParser(description="Product Feed Evaluation Agent CLI")
    
    parser.add_argument(
        "feed_file",
        help="Path to XML feed file or URL"
    )
    
    parser.add_argument(
        "--api-key",
        help="OpenAI API key (or set OPENAI_API_KEY environment variable)"
    )
    
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        choices=["gpt-4o-mini", "gpt-4-turbo", "gpt-4"],
        help="GPT model to use (default: gpt-4o-mini)"
    )
    
    parser.add_argument(
        "--fields",
        nargs="+",
        default=["title", "description", "brand", "price", "color", "size", "material", "availability"],
        help="Fields to include in analysis"
    )
    
    parser.add_argument(
        "--questions",
        type=int,
        default=12,
        help="Number of questions per product (default: 12)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Batch size for processing (default: 20)"
    )
    
    parser.add_argument(
        "--output",
        help="Output CSV file path (default: auto-generated)"
    )
    
    parser.add_argument(
        "--sample",
        type=int,
        help="Process only first N products (for testing)"
    )
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key required. Set OPENAI_API_KEY environment variable or use --api-key")
        sys.exit(1)
    
    # Ensure link field is included
    if "link" not in args.fields:
        args.fields.append("link")
    
    try:
        print("🧠 Product Feed Evaluation Agent - CLI")
        print("=" * 50)
        
        # Initialize components
        parser_obj = FeedParser()
        evaluator = ProductFeedEvaluator(api_key, args.model)
        
        # Load feed
        print(f"📥 Loading feed from: {args.feed_file}")
        
        if args.feed_file.startswith("http"):
            import requests
            response = requests.get(args.feed_file, timeout=30)
            response.raise_for_status()
            xml_content = response.text
        else:
            with open(args.feed_file, 'r', encoding='utf-8') as f:
                xml_content = f.read()
        
        # Process feed
        print("🔧 Processing feed...")
        products_df = parser_obj.process_feed(xml_content, args.fields)
        
        if args.sample:
            products_df = products_df.head(args.sample)
            print(f"📊 Processing sample of {len(products_df)} products")
        else:
            print(f"📊 Found {len(products_df)} products")
        
        # Evaluate products
        print("🤖 Evaluating products with AI...")
        results_df = evaluator.evaluate_products_batch(
            products_df, args.fields, args.questions, args.batch_size
        )
        
        # Generate output filename
        if not args.output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output = f"output/feed_analysis_{timestamp}.csv"
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        # Save results
        results_df.to_csv(args.output, index=False)
        
        # Print summary
        avg_coverage = results_df['coverage_pct'].mean()
        high_coverage = len(results_df[results_df['coverage_pct'] >= 70])
        low_coverage = len(results_df[results_df['coverage_pct'] < 30])
        
        print("\n📈 Results Summary:")
        print(f"  Average Coverage: {avg_coverage:.1f}%")
        print(f"  High Coverage (≥70%): {high_coverage} products")
        print(f"  Low Coverage (<30%): {low_coverage} products")
        print(f"  Results saved to: {args.output}")
        
        print("\n✅ Evaluation complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()