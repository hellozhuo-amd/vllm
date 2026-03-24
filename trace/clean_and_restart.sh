#!/bin/bash
# Script to clear all caches and ensure clean restart of vLLM server
#
INSTALL_PATH=$1

set -e

echo "=== Cleaning vLLM caches and preparing for restart ==="

# 1. Stop any running vLLM processes
echo "[1/6] Stopping any running vLLM processes..."
pkill -f "vllm serve" || echo "No vLLM server processes found"
sleep 2

# 2. Clear Python bytecode cache
echo "[2/6] Clearing Python __pycache__ directories..."
cd $INSTALL_PATH
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
rm /root/.cache/vllm -rf

# 3. Clear torch compilation cache
echo "[3/6] Clearing torch compilation cache..."
rm -rf ~/.cache/torch_extensions/* 2>/dev/null || true
rm -rf ~/.triton/cache/* 2>/dev/null || true
# Also check XDG cache
rm -rf ${XDG_CACHE_HOME:-$HOME/.cache}/torch_extensions/* 2>/dev/null || true
rm -rf ${XDG_CACHE_HOME:-$HOME/.cache}/triton/* 2>/dev/null || true

# 4. Clear CUDA kernel cache
echo "[4/6] Clearing CUDA kernel cache..."
rm -rf ~/.nv/* 2>/dev/null || true

bash trace/start_vllm.sh
