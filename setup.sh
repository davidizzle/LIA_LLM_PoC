#!/bin/bash

# Initialize LFS and pull all LFS-tracked files
apt-get update && apt-get install -y git-lfs
git lfs install
git lfs pull
