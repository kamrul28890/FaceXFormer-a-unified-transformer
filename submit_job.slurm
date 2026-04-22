#!/bin/bash
#SBATCH -J trainMain      # Job name
#SBATCH -o train_output.%j.out       # Output file (%j = job ID)
#SBATCH -e train_error.%j.out        # Error file (%j = job ID)
#SBATCH -t 9:00:00                  # Wall time (9 hours)
#SBATCH -N 4                         # Number of nodes
#SBATCH -n 4                         # Total tasks (1 per node)
#SBATCH -p gh                        # GPU partition
#SBATCH -A TG-NAIRR250282            # Account name or Project Charge Code
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=arko@purdue.edu

# 1. Load necessary modules
module load gcc    # Load GCC (standard for Vista)
module load cuda     # Load CUDA (adjust version if needed)
module load python3       # Load Python

# 2. Activate your Virtual Environment
# You must have created this beforehand (e.g., python3 -m venv $SCRATCH/myenv)
source $SCRATCH/python-envs/test/bin/activate

# 3. Setup Distributed Training Variables
# Get the hostname of the master node (first node in the list)
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# Set a master port (arbitrary free port)
MASTER_PORT=29500

# 4. Create a temporary run script for torchrun
# This script will be executed on every node by mpirun
cat << EOF > ./run_torch.sh
#!/bin/bash
# The host (master) and number of nodes are passed as arguments
HOST=\$1
NODES=\$2

# Determine the local rank (node index) from MPI environment variable
# OMPI_COMM_WORLD_RANK is standard for TACC's OpenMPI
LOCAL_RANK=\${OMPI_COMM_WORLD_RANK}

# Launch the training script using torchrun
# Since each Vista GH node has 1 GPU, we set --nproc_per_node=1
torchrun --nproc_per_node=1 \
         --nnodes=\$NODES \
         --node_rank=\${LOCAL_RANK} \
         --master_addr=\$HOST \
         --master_port=$MASTER_PORT \
         train.py
EOF

# Make the script executable
chmod +x ./run_torch.sh

# 5. Launch the job across all nodes
# Use mpirun with standard TACC flags to run one instance of the script per node
mpirun -np $SLURM_NPROCS \
       --map-by ppr:1:node \
       ./run_torch.sh $MASTER_ADDR $SLURM_NNODES
