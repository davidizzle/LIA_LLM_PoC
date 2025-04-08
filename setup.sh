#!/bin/bash
echo "Running setup.sh..."

# Check if git-lfs is installed
if command -v git-lfs &> /dev/null; then
    echo "git-lfs is available, running pull..."
    git lfs pull
else
    echo "git-lfs not installed in this environment. Skipping pull."
fi