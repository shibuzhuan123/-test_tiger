#!/bin/bash
# Jaguar ReID Training Script
# Usage: bash run_train.sh

# Activate jaguar environment
source ~/venv/jaguar/bin/activate

cd /home/fei/kaggle

echo "========================================"
echo " Jaguar ReID Training"
echo "========================================"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Environment: jaguar"
echo "Python: $(which python)"
echo ""

# Check GPU
echo "=== Checking GPU ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
    echo ""
    # Check CUDA availability in Python
    python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
else
    echo "No GPU detected, using CPU"
fi
echo ""

# Check data
echo "=== Checking Data ==="
echo "Train CSV: $(ls -lh train.csv 2>/dev/null | awk '{print $9}' || echo 'Not found')"
echo "Test CSV: $(ls -lh test.csv 2>/dev/null | awk '{print $9}' || echo 'Not found')"
echo "Train images: $(ls data/train/*.png 2>/dev/null | wc -l)"
echo "Test images: $(find data/test -name '*.png' 2>/dev/null | wc -l)"
echo ""

# Check packages
echo "=== Checking Packages ==="
python -c "import torch; import timm; import pandas; print('✓ All packages ready')"
echo ""

# Run training
echo "=== Starting Training ==="
nohup python train.py > train.log 2>&1 &
echo "Training started! PID: $!"
echo $! > train.pid
echo ""
echo "Monitor with: tail -f train.log"
echo "Stop with: kill \$(cat train.pid)"
echo "========================================"
