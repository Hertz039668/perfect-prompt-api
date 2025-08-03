# 🚀 Perfect Prompt API - Deployment Guide

This guide will walk you through deploying your Perfect Prompt API and integrating it with ChatGPT Custom GPT.

## 📋 Prerequisites

1. ✅ Python project with virtual environment
2. ✅ All dependencies installed
3. ✅ API server tested locally

## 🎯 Deployment Options

### Option 1: Railway (Recommended - Free & Easy)

**Why Railway?**
- ✅ Free tier available
- ✅ Automatic deployments from GitHub
- ✅ Built-in domain and HTTPS
- ✅ Easy environment variables management

**Steps:**

1. **Create GitHub Repository**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/perfect-prompt.git
   git push -u origin main
   ```

2. **Deploy to Railway**
   - Go to [railway.app](https://railway.app)
   - Sign up with GitHub
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your `perfect-prompt` repository
   - Railway will automatically detect Python and deploy

3. **Configure Environment Variables** (Optional)
   - In Railway dashboard, go to Variables tab
   - Add: `REQUIRE_API_KEY=false` (for easier testing)

4. **Get Your API URL**
   - Railway will provide a URL like: `https://your-app.railway.app`

### Option 2: Heroku

1. **Install Heroku CLI**
   ```bash
   # Download from https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Deploy**
   ```bash
   heroku login
   heroku create your-perfect-prompt-api
   git push heroku main
   ```

3. **Get Your URL**: `https://your-perfect-prompt-api.herokuapp.com`

### Option 3: Docker (Self-hosted)

1. **Build and Run**
   ```bash
   docker build -t perfect-prompt .
   docker run -p 8000:8000 perfect-prompt
   ```

2. **Your API**: `http://localhost:8000`

## 🧪 Test Your Deployed API

Once deployed, test your endpoints:

```bash
# Replace YOUR_API_URL with your actual deployed URL
curl -X POST "YOUR_API_URL/api/v1/analyze" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a blog post about AI"}'
```

**Expected Response:**
```json
{
  "success": true,
  "prompt": "Write a blog post about AI",
  "analysis": {
    "clarity_score": 0.85,
    "specificity_score": 0.65,
    "engagement_score": 0.75
  },
  "suggestions": [
    "Consider adding target audience specification",
    "Include desired length or format details"
  ]
}
```

## 🤖 ChatGPT Custom GPT Integration

### Step 1: Create Custom GPT

1. **Go to ChatGPT**
   - Visit [chat.openai.com](https://chat.openai.com)
   - Click your profile → "My GPTs"
   - Click "Create a GPT"

2. **Configure Basic Settings**
   - **Name**: "Perfect Prompt Optimizer"
   - **Description**: "AI-powered prompt optimization using advanced NLP and ML techniques"
   - **Instructions**: 
   ```
   You are an AI prompt optimization assistant powered by the Perfect Prompt API. 
   
   Your role:
   1. Analyze user prompts for clarity, specificity, and effectiveness
   2. Provide optimized versions of prompts
   3. Explain improvements made
   4. Offer suggestions for better prompt engineering
   
   Always use the Perfect Prompt API to analyze and optimize prompts before providing recommendations.
   ```

### Step 2: Add API Integration

1. **In the Configure tab, scroll to Actions**
2. **Click "Create new action"**
3. **Import the Schema**:
   - Copy the content from `deploy/chatgpt-schema.json`
   - Paste it in the Schema field

4. **Configure Authentication**:
   - Authentication Type: "None" (if REQUIRE_API_KEY=false)
   - Or "API Key" if you enabled authentication

5. **Set Server URL**:
   - Replace `YOUR_API_URL` in the schema with your actual deployed URL
   - Example: `https://your-app.railway.app`

### Step 3: Test Integration

Ask your Custom GPT:
```
"Analyze this prompt: 'Write something good about technology'"
```

The GPT should:
1. Call your API's `/analyze` endpoint
2. Show analysis results
3. Provide optimization suggestions

### Step 4: Advanced Usage Examples

```
"Optimize this prompt for better results: 'Create a marketing email'"

"Compare these two prompts and tell me which is better:
1. 'Write a report'
2. 'Write a comprehensive quarterly sales report for the executive team, including trends, forecasts, and actionable recommendations'"

"Analyze and optimize this prompt for an AI assistant: 'Help me with my code'"
```

## 🔧 Troubleshooting

### Common Issues:

1. **API Not Responding**
   - Check if your deployment is running
   - Verify the URL is correct
   - Check logs in your deployment platform

2. **CORS Errors in ChatGPT**
   - CORS is already configured in the server
   - Ensure your deployed API includes CORS headers

3. **Schema Import Issues**
   - Make sure the server URL in the schema matches your deployment
   - Verify JSON syntax is valid

4. **Authentication Errors**
   - If using API key auth, set it in ChatGPT action configuration
   - For testing, set `REQUIRE_API_KEY=false`

## 📊 Monitoring Your API

### Check API Health
```bash
curl YOUR_API_URL/health
```

### View API Documentation
Visit: `YOUR_API_URL/docs` (Interactive Swagger UI)

## 🎉 Success!

Your Perfect Prompt API is now:
- ✅ Deployed and accessible
- ✅ Integrated with ChatGPT Custom GPT
- ✅ Ready to optimize prompts for better AI interactions

### What's Next?

1. **Share your Custom GPT** with others
2. **Monitor usage** through your deployment platform
3. **Collect feedback** and improve the optimization algorithms
4. **Scale up** if you get high usage

## 🔗 Quick Links

- **Railway Dashboard**: [railway.app/dashboard](https://railway.app/dashboard)
- **Heroku Dashboard**: [dashboard.heroku.com](https://dashboard.heroku.com)
- **ChatGPT GPTs**: [chat.openai.com/gpts/mine](https://chat.openai.com/gpts/mine)
- **API Documentation**: `YOUR_API_URL/docs`

---

Need help? Check the detailed guides in the `deploy/` folder or raise an issue on GitHub!
