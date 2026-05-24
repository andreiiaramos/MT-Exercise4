if python3 -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    DEVICE_NAME=cuda
    CONFIG_DIR=configs/cuda
else
    DEVICE_NAME=cpu
    CONFIG_DIR=configs
fi
export DEVICE_NAME CONFIG_DIR
