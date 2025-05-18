# Vercel Deployment Instructions

## Prerequisites
- A Vercel account (https://vercel.com)
- Vercel CLI installed (optional for command-line deployment)

## Deployment Steps

### 1. Using Vercel Web UI
1. Go to [Vercel](https://vercel.com) and sign in
2. Click "Add New" → "Project"
3. Import your GitHub repository or upload your project directly
4. Configure project settings:
   - Framework Preset: Other
   - Build Command: Leave empty
   - Output Directory: Leave empty
   - Install Command: `pip install -r requirements.txt`
5. Add environment variables if needed
6. Click "Deploy"

### 2. Using Vercel CLI
1. Install Vercel CLI: `npm i -g vercel`
2. Log in to Vercel: `vercel login`
3. Navigate to your project directory
4. Run: `vercel`
5. Follow the prompts to configure your deployment

## Configuration Files
The following files have been created/modified for Vercel deployment:
- `vercel.json`: Main configuration file
- `wsgi.py`: WSGI entry point
- `.vercelignore`: Files to exclude from deployment
- `runtime.txt`: Python version specification

## Troubleshooting
- Check Vercel logs for deployment errors
- Ensure all dependencies are listed in requirements.txt
- Make sure static assets are properly configured
- Verify Python version compatibility