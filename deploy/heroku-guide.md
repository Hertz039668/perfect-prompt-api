# Heroku Deployment Guide

## Prerequisites
1. Create account at https://heroku.com
2. Install Heroku CLI
3. Login: `heroku login`

## Deployment Steps

### 1. Create Heroku App
```bash
cd "C:\VS Code\Perfect Prompt"
heroku create your-perfect-prompt-api
```

### 2. Set Environment Variables
```bash
heroku config:set ENVIRONMENT=production
heroku config:set LOG_LEVEL=info
```

### 3. Deploy
```bash
git add .
git commit -m "Deploy Perfect Prompt API"
git push heroku main
```

### 4. Scale App
```bash
heroku ps:scale web=1
```

## Your API will be available at:
https://your-perfect-prompt-api.herokuapp.com
