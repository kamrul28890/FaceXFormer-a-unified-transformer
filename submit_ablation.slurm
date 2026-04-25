#!/bin/bash
#SBATCH -J ablation_full               # Job name (override with --job-name on sbatch)
#SBATCH -o ablation_full_%j.out        # Stdout (%j = job ID)
#SBATCH -e ablation_full_%j.err        # Stderr
#SBATCH -t 1:00:00                     # Wall time (same as main training job)
#SBATCH -N 4                           # Number of nodes
#SBATCH -n 4                           # Total tasks (1 per node)
#SBATCH -p gh-dev                          # GPU partition
#SBATCH -A TG-NAIRR250282              # Account name / project charge code
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=arko@purdue.edu

# ============================================================================
# CHOOSE THE VARIANT TO RUN
#
# Set ABLATION_VARIANT to one of:
#   full                     (unmodified model)
#   standard_cross_attention (one attention direction zeroed out)
#   unbalanced_sampler       (balanced-batch sampler disabled)
#   uniform_loss             (all task loss weights = 1.0)
#
# Submit one job per variant, e.g.:
#   ABLATION_VARIANT=full                     sbatch submit_ablation_full.slurm
#   ABLATION_VARIANT=standard_cross_attention sbatch submit_ablation_full.slurm
#   ABLATION_VARIANT=unbalanced_sampler       sbatch submit_ablation_full.slurm
#   ABLATION_VARIANT=uniform_loss             sbatch submit_ablation_full.slurm
# ============================================================================
ABLATION_VARIANT=${ABLATION_VARIANT:-full}

# 1. Load necessary modules
module load gcc
module load cuda
module load python3

# 2. Activate virtual environment
source $SCRATCH/python-envs/test/bin/activate

# 3. Setup distributed training variables
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29502   # Different from train (29500) and ablation_tiny (29501)

# 4. Create a per-job run script for torchrun
cat << EOF > ./run_ablation_full.sh
#!/bin/bash
HOST=\$1
NODES=\$2
LOCAL_RANK=\${OMPI_COMM_WORLD_RANK}

torchrun --nproc_per_node=1 \
         --nnodes=\$NODES \
         --node_rank=\${LOCAL_RANK} \
         --master_addr=\$HOST \
         --master_port=$MASTER_PORT \
         ablation_study.py \
             --variant $ABLATION_VARIANT \
             --dataset-root ../facexformer-my/datasets \
             --output-dir results/ablation_full
EOF

chmod +x ./run_ablation_full.sh

# 5. Launch across all nodes (identical to submit_job.slurm)
mpirun -np $SLURM_NPROCS \
       --map-by ppr:1:node \
       ./run_ablation_full.sh $MASTER_ADDR $SLURM_NNODES
