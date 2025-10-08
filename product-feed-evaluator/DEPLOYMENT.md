# 🚀 Streamlit Deployment Guide

This guide covers deploying the Product Feed Evaluation Agent to Streamlit Cloud.

## 📋 Prerequisites

- GitHub account
- Streamlit Cloud account (free)
- OpenAI API key

## 🔧 Deployment Steps

### 1. Push to GitHub

1. Create a new repository on GitHub
2. Push your code to the repository:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/yourusername/your-repo-name.git
   git push -u origin main
   ```

### 2. Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Connect your GitHub repository
4. Set the following configuration:
   - **Repository**: `yourusername/your-repo-name`
   - **Branch**: `main`
   - **Main file path**: `app.py`

### 3. Configure Environment Variables

In the Streamlit Cloud dashboard:

1. Go to your app settings
2. Add the following secrets:
   ```
   OPENAI_API_KEY = "your_openai_api_key_here"
   ```

### 4. Deploy

Click "Deploy!" and wait for the app to build and deploy.

## 🔒 Security Notes

- **Never commit API keys** to your repository
- Use Streamlit's secrets management for sensitive data
- The `.env` file is git-ignored for local development
- API keys are only stored in Streamlit Cloud's secure environment

## 📊 App Configuration

The app will be available at: `https://your-app-name.streamlit.app`

## 🛠️ Local Development

For local development:

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variable
export OPENAI_API_KEY=your_key_here

# Run locally
streamlit run app.py
```

## 🔄 Updates

To update your deployed app:

1. Push changes to your GitHub repository
2. Streamlit Cloud will automatically redeploy
3. No manual intervention required

## 📈 Monitoring

- Monitor your app usage in the Streamlit Cloud dashboard
- Check OpenAI API usage in your OpenAI dashboard
- Set up billing alerts if needed

## 🆘 Troubleshooting

### Common Issues

1. **Build Failures**: Check that all dependencies are in `requirements.txt`
2. **API Key Errors**: Verify secrets are set correctly in Streamlit Cloud
3. **Memory Issues**: Consider upgrading to a paid Streamlit plan for larger feeds
4. **Rate Limits**: The app includes built-in rate limiting for API calls

### Support

- Streamlit Cloud documentation: [docs.streamlit.io](https://docs.streamlit.io)
- OpenAI API documentation: [platform.openai.com](https://platform.openai.com)