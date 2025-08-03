# Railway Deployment Guide for Perfect Prompt API

## Prerequisites
1. Create account at https://railway.app
2. Install Railway CLI: `npm install -g @railway/cli`
3. Login: `railway login`

## Deployment Steps

### 1. Initialize Railway Project
```bash
cd "C:\VS Code\Perfect Prompt"
railway login
railway init
railway link
```

### 2. Set Environment Variables
```bash
railway variables set ENVIRONMENT=production
railway variables set LOG_LEVEL=info
railway variables set PORT=8000
```

### 3. Deploy
```bash
railway up
```

### 4. Custom Domain (Optional)
- Go to Railway dashboard
- Select your project
- Go to Settings > Domains
- Add custom domain or use provided railway.app domain

## Your API will be available at:
https://your-project-name.railway.app

## Test your deployed API:
curl https://your-project-name.railway.app/health
