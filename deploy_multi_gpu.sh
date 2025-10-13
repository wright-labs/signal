#!/bin/bash
# Deploy multi-GPU training sessions to Modal

set -e

echo "================================="
echo "Multi-GPU Deployment Script"
echo "================================="
echo ""

# Check Modal CLI
if ! command -v modal &> /dev/null; then
    echo "❌ Modal CLI not found. Install with: pip install modal"
    exit 1
fi

echo "✓ Modal CLI found"
echo ""

# Check authentication
if ! modal token show &> /dev/null; then
    echo "❌ Not authenticated with Modal. Run: modal token set --token-id YOUR_TOKEN_ID --token-secret YOUR_SECRET"
    exit 1
fi

echo "✓ Authenticated with Modal"
echo ""

# Deploy multi-GPU training sessions
echo "Deploying multi-GPU training sessions..."
echo "This will create 8 training session variants:"
echo "  - l40s:1, l40s:2, l40s:4"
echo "  - a100:1, a100:2, a100:4"
echo "  - h100:1, h100:2"
echo ""

read -p "Continue with deployment? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled"
    exit 1
fi

echo ""
echo "Deploying..."
modal deploy modal_runtime/multi_gpu_session.py

echo ""
echo "================================="
echo "✅ Deployment Complete!"
echo "================================="
echo ""
echo "Available GPU configurations:"
echo "  - l40s:1  (Single L40S)"
echo "  - l40s:2  (2x L40S)"
echo "  - l40s:4  (4x L40S)"
echo "  - a100:1  (Single A100)"
echo "  - a100:2  (2x A100)"
echo "  - a100:4  (4x A100)"
echo "  - h100:1  (Single H100)"
echo "  - h100:2  (2x H100)"
echo ""
echo "Usage with SDK:"
echo "  client.create_run("
echo "      base_model='meta-llama/Llama-3.2-3B',"
echo "      gpu_config='l40s:2'  # ← Specify GPU config"
echo "  )"
echo ""
echo "Test with:"
echo "  python tests/test_multi_gpu_llama.py"
echo ""

