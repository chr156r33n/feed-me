# 🧠 Product Feed Evaluation Agent

A tool that evaluates Google Shopping product feed quality by checking how well each product entry answers typical buyer questions.

## 🎯 Features

- **AI-Powered Analysis**: Uses GPT to generate buyer questions and evaluate how well product data answers them
- **Comprehensive Scoring**: Provides coverage percentages and detailed feedback
- **Quality Checks**: Identifies missing fields and content quality issues
- **Batch Processing**: Handles large feeds efficiently with rate limiting
- **Interactive Interface**: Easy-to-use Streamlit web interface
- **Export Results**: Download analysis as CSV for further review

## 🚀 Quick Start

### 1. Installation

```bash
cd product-feed-evaluator
pip install -r requirements.txt
```

### 2. Set up API Key

**Option 1: Environment Variable (Recommended)**
```bash
export OPENAI_API_KEY=your_openai_api_key_here
```

**Option 2: .env File**
Create a `.env` file in the project directory:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

**Option 3: Enter in App**
You can also enter your API key directly in the Streamlit interface.

> ⚠️ **Security Note**: Never commit API keys to version control. The `.env` file is already included in `.gitignore`.

### 3. Run the Application

**Option 1: Web Interface (Recommended)**
```bash
streamlit run app.py
```

**Option 2: Command Line Interface**
```bash
python cli.py sample_feed.xml
```

For CLI options:
```bash
python cli.py --help
```

## 📋 Usage

1. **Upload or paste your Google Shopping XML feed**
2. **Select which fields to include in the analysis**
3. **Configure the number of questions per product (default: 12)**
4. **Click "Start Evaluation" to begin the analysis**
5. **Review results and download the CSV report**

## 📊 Output

The tool generates a CSV file with the following columns:

- `product_url`: The product's main URL
- `name_plus_context`: Combined product information
- `questions_json`: Generated buyer questions
- `judgements_json`: AI evaluation of how well questions are answered
- `yes_count`: Number of fully answered questions
- `partial_count`: Number of partially answered questions
- `no_count`: Number of unanswered questions
- `coverage_pct`: Overall coverage percentage
- `missing_core_fields`: List of missing essential fields

## ⚙️ Configuration

### Field Selection
Choose which product fields to include in the analysis:
- **Always Required**: `link`, `title`, `description`, `image_link`
- **Strongly Recommended**: `brand`, `price`, `availability`, `color`, `size`, `material`

### Model Options
- **gpt-4o-mini**: Faster and cheaper (recommended for testing)
- **gpt-4-turbo**: More accurate results
- **gpt-4**: Highest accuracy (most expensive)

### Batch Processing
- Process products in batches to manage API rate limits
- Default batch size: 20 products
- Includes automatic delays between API calls

## 🧮 Scoring System

| Verdict | Score | Description |
|---------|-------|-------------|
| Yes | 1.0 | Question is explicitly and clearly answered |
| Partial | 0.5 | Question is hinted at but incomplete or vague |
| No | 0.0 | Question is not addressed at all |

**Coverage % = (Yes + 0.5 × Partial) ÷ Total × 100**

## 🔍 Quality Checks

The tool automatically flags:
- Missing essential fields (`brand`, `gtin`, `price`, `availability`)
- Title length issues (<20 or >150 characters)
- Description length issues (<60 or >5000 characters)
- Poor quality indicators (ALL CAPS, excessive emojis)
- Invalid image links

## 💰 Cost Optimization

- **Question Caching**: Reuses questions for similar product types
- **Token Trimming**: Limits context length to reduce costs
- **Batch Processing**: Efficient API usage with rate limiting
- **Sampling**: Start with 200 products to test effectiveness

## 📁 Project Structure

```
product-feed-evaluator/
├── app.py                   # Streamlit web application
├── cli.py                   # Command-line interface
├── prompts.py               # GPT prompt templates
├── feed_parser.py           # XML parsing and data cleaning
├── evaluator.py             # GPT interaction and scoring
├── requirements.txt         # Python dependencies
├── sample_feed.xml          # Example XML feed for testing
├── .env                     # Environment variables (create this - not in git)
├── .gitignore               # Git ignore file
└── output/                  # Generated CSV files
    └── feed_analysis_YYYYMMDD.csv
```

## 🧪 Testing

Use the included `sample_feed.xml` to test the tool with sample product data.

## 🔧 Troubleshooting

### Common Issues

1. **API Key Error**: Make sure your OpenAI API key is correctly set in environment variables or `.env` file
2. **XML Parse Error**: Ensure your feed is valid XML format
3. **Rate Limiting**: Reduce batch size if you encounter API rate limits
4. **Memory Issues**: Process smaller batches for very large feeds

### Performance Tips

- Use `gpt-4o-mini` for faster processing
- Start with a sample of 200 products
- Enable question caching for similar product types
- Process during off-peak hours to avoid rate limits

## 📈 Best Practices

1. **Start Small**: Test with 200 products first
2. **Review Results**: Check the generated questions make sense for your products
3. **Iterate**: Use results to improve your product feed
4. **Monitor Costs**: Keep track of API usage for large feeds
5. **Regular Analysis**: Run evaluations periodically to maintain feed quality

## 🤝 Contributing

This tool is designed to be simple, explainable, and fast. Feel free to extend it with additional features like:
- Topic classification for question types
- Category-specific dashboards
- Integration with CMS/PIM systems
- Readability scoring for descriptions

## 🔒 Security Best Practices

- **Never commit API keys** to version control
- Use environment variables for sensitive data
- The `.env` file is automatically ignored by git
- Consider using a secrets management service for production
- Rotate API keys regularly
- Monitor API usage for unexpected activity

## 📄 License

This project is open source and available under the MIT License.