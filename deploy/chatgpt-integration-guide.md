# ChatGPT Custom GPT Integration Guide

## Step 1: Deploy Your API
1. Choose a deployment platform (Railway recommended for free tier)
2. Deploy using the guides in `/deploy/` folder
3. Note your API URL (e.g., `https://your-app.railway.app`)
4. Test the API is working: `curl https://your-app.railway.app/health`

## Step 2: Create Custom GPT
1. Go to https://chat.openai.com
2. Click your profile → "My GPTs"
3. Click "Create a GPT"
4. Choose "Configure" tab

## Step 3: Configure GPT Settings

### Basic Information
- **Name**: Perfect Prompt Optimizer
- **Description**: AI-powered prompt optimization assistant that analyzes and improves prompts for better AI performance
- **Instructions**: 
```
You are Perfect Prompt Optimizer, an AI assistant specialized in analyzing and optimizing prompts for better AI model performance. You help users:

1. ANALYZE prompts for effectiveness metrics (clarity, specificity, efficiency)
2. OPTIMIZE prompts using different strategies (clarity-focused, efficiency-focused, comprehensive, creative)
3. COMPARE multiple prompts to find the best one
4. PROVIDE suggestions for improvement

Always use the Perfect Prompt API to:
- Analyze prompts when users ask for feedback
- Optimize prompts when users want improvements
- Compare prompts when users provide multiple options

Be helpful, clear, and provide actionable insights based on the API results.
```

### Conversation Starters
Add these example prompts:
- "Analyze this prompt: 'Write a story about AI'"
- "Optimize my prompt for better clarity"
- "Compare these three prompts and tell me which is best"
- "Help me improve this prompt for my AI model"

## Step 4: Add API Integration

### In the Configure tab, scroll to "Actions"
1. Click "Create new action"
2. Copy the schema from `/deploy/chatgpt-schema.json`
3. Paste it in the Schema field
4. Update the server URL to your deployed API
5. Set Authentication to "None" (or add API key if you implement auth)

## Step 5: Test Your GPT
1. Click "Preview" to test
2. Try asking: "Analyze this prompt: 'Write code'"
3. Verify it calls your API and returns results
4. Test optimization: "Optimize this prompt: 'Create content'"

## Step 6: Publish (Optional)
1. Click "Save" 
2. Choose "Only me" for private use
3. Or "Public" to share with others

## Example Usage After Setup

**User**: "Analyze this prompt: 'Write something about machine learning'"

**GPT Response**: 
"I'll analyze your prompt using the Perfect Prompt API...

**Analysis Results:**
- Clarity Score: 0.92/1.0 ✅
- Specificity Score: 0.12/1.0 ⚠️  
- Efficiency Score: 0.99/1.0 ✅

**Suggestions for Improvement:**
- Add more specific details about what type of content you want
- Include examples or constraints to improve precision
- Consider specifying the audience or purpose

**Would you like me to optimize this prompt for you?**"

## Troubleshooting

### Common Issues:
1. **API not responding**: Check your deployment is running
2. **CORS errors**: Ensure your API allows ChatGPT's domain
3. **Schema errors**: Validate JSON schema format
4. **Authentication**: Add API keys if needed

### Testing Your API:
```bash
# Test health
curl https://your-api.railway.app/health

# Test analyze
curl -X POST https://your-api.railway.app/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test prompt"}'

# Test optimize
curl -X POST https://your-api.railway.app/api/v1/optimize \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test", "strategy": "comprehensive"}'
```

Your Perfect Prompt API is now integrated with ChatGPT! 🎉
