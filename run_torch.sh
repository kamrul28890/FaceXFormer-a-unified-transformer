#!/bin/bash
# The host (master) and number of nodes are passed as arguments
HOST=$1
NODES=$2

# Determine the local rank (node index) from MPI environment variable
# OMPI_COMM_WORLD_RANK is standard for TACC's OpenMPI
LOCAL_RANK=${OMPI_COMM_WORLD_RANK}

# Launch the training script using torchrun
# Since each Vista GH node has 1 GPU, we set --nproc_per_node=1
torchrun --nproc_per_node=1          --nnodes=$NODES          --node_rank=${LOCAL_RANK}          --master_addr=$HOST          --master_port=29500          train.py
