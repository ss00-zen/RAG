#!/bin/bash
# Start LiteLLM Proxy Server

set -e

echo "Starting LiteLLM Proxy Server..."
echo "================================"

# Load environment variables from .env
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "✓ Loaded environment variables from .env"
fi

# Verify NVIDIA_API_KEY is set
if [ -z "$NVIDIA_API_KEY" ]; then
    echo "❌ Error: NVIDIA_API_KEY is not set"
    exit 1
fi

echo "✓ NVIDIA_API_KEY is set"

# Start LiteLLM proxy on port 4000
# --config: YAML configuration file
# --port: Listen on port 4000
# --debug: Enable debug logging
python -m litellm.proxy.proxy_cli \
    --config litellm_config.yaml \
    --port 4000 \
    --debug
