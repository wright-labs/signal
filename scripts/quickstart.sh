#!/bin/bash
# Quick start script for Signal

set -e  # Exit on error

echo "=================================="
echo "Signal - Quick Start Setup"
echo "=================================="
echo ""

# Check if Modal is installed
if ! command -v modal &> /dev/null; then
    echo "❌ Modal CLI not found. Installing..."
    pip install modal
else
    echo "✓ Modal CLI found"
fi

# Check if Modal is authenticated
if ! modal token show &> /dev/null; then
    echo ""
    echo "❌ Modal not authenticated."
    echo "Please run: modal setup"
    exit 1
else
    echo "✓ Modal authenticated"
fi

# Check for Modal secret
echo ""
echo "Checking for HuggingFace secret..."
if modal secret list | grep -q "secrets-hf-wandb"; then
    echo "✓ HuggingFace secret found"
else
    echo "❌ HuggingFace secret not found"
    echo ""
    read -p "Enter your HuggingFace token: " HF_TOKEN
    
    if [ -z "$HF_TOKEN" ]; then
        echo "❌ No token provided. Exiting."
        exit 1
    fi
    
    modal secret create secrets-hf-wandb HUGGINGFACE_TOKEN="$HF_TOKEN"
    echo "✓ Created HuggingFace secret"
fi

# Check for Modal volume
echo ""
echo "Checking for Modal volume..."
if modal volume list | grep -q "signal-data"; then
    echo "✓ Volume 'signal-data' exists"
else
    echo "Creating Modal volume..."
    modal volume create signal-data
    echo "✓ Created volume 'signal-data'"
fi

# Create data directory
echo ""
echo "Creating data directory..."
mkdir -p data
echo "✓ Created data directory"

# Deploy Modal functions
echo ""
echo "Deploying Modal functions..."
echo "(This may take 10-15 minutes on first run)"
echo ""
modal deploy modal_runtime/primitives.py

echo ""
echo "✓ Modal functions deployed"

# Generate API key
echo ""
echo "Generating API key..."
USER_ID="user_$(date +%s)"
API_KEY=$(python scripts/manage_keys.py generate "$USER_ID" --description "Quickstart key" 2>/dev/null | grep "API Key:" | awk '{print $3}')

if [ -z "$API_KEY" ]; then
    echo "❌ Failed to generate API key"
    exit 1
fi

echo "✓ Generated API key for user: $USER_ID"
echo ""
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "Your API Key: $API_KEY"
echo ""
echo "⚠️  Save this key securely - it won't be shown again!"
echo ""
echo "Next steps:"
echo "1. Start the API server:"
echo "   python api/main.py"
echo ""
echo "2. In another terminal, run the example:"
echo "   python examples/sft_example.py"
echo "   (Update the API key in the script first)"
echo ""
echo "3. Check the documentation:"
echo "   cat README.md"
echo ""

# Create .env file with API key
echo "API_KEY=$API_KEY" > .env.example
echo "BASE_URL=http://localhost:8000" >> .env.example
echo ""
echo "✓ Created .env.example with your API key"
echo ""

