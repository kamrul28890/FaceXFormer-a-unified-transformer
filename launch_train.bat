@echo off
REM Launch script for multi-GPU distributed training on Windows
REM 
REM Usage:
REM   Single GPU:   launch_train.bat
REM   Multi-GPU:    launch_train.bat 4  (for 4 GPUs)

setlocal

REM Get number of GPUs from argument or default to 1
if "%1"=="" (
    set NUM_GPUS=1
) else (
    set NUM_GPUS=%1
)

echo ======================================
if %NUM_GPUS%==1 (
    echo Single GPU Training
    echo ======================================
    python train_simple.py
) else (
    echo Multi-GPU Distributed Training
    echo Number of GPUs: %NUM_GPUS%
    echo ======================================
    
    python -m torch.distributed.run ^
        --standalone ^
        --nproc_per_node=%NUM_GPUS% ^
        train.py
)

endlocal
