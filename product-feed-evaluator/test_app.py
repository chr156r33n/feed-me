"""
Simple test script to verify the app components work correctly.
"""

import os
import sys
from dotenv import load_dotenv

def test_imports():
    """Test that all required modules can be imported."""
    try:
        import streamlit as st
        import openai
        import pandas as pd
        import lxml
        import requests
        from feed_parser import FeedParser
        from evaluator import ProductFeedEvaluator
        from prompts import QUESTION_GENERATION_PROMPT, ANSWER_JUDGEMENT_PROMPT
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_feed_parser():
    """Test feed parser with sample data."""
    try:
        from feed_parser import FeedParser
        parser = FeedParser()
        
        # Test with minimal XML
        sample_xml = '''<?xml version="1.0"?>
        <rss version="2.0">
            <channel>
                <item>
                    <title>Test Product</title>
                    <link>https://example.com/product</link>
                    <description>Test description</description>
                </item>
            </channel>
        </rss>'''
        
        products = parser.parse_xml_feed(sample_xml)
        assert len(products) == 1
        assert products[0]['title'] == 'Test Product'
        print("✅ Feed parser test passed")
        return True
    except Exception as e:
        print(f"❌ Feed parser test failed: {e}")
        return False

def test_environment():
    """Test environment setup."""
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print("✅ OpenAI API key found in environment")
        return True
    else:
        print("⚠️  OpenAI API key not found in environment")
        print("   Set OPENAI_API_KEY environment variable or create .env file")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Product Feed Evaluation Agent")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Feed Parser Test", test_feed_parser),
        ("Environment Test", test_environment)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}:")
        if test_func():
            passed += 1
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The app is ready to run.")
        print("\nTo start the app:")
        print("  streamlit run app.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()