"""
Evaluation-only script for FaceXFormer-main.
Run test set evaluation on a trained checkpoint without training.

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pth
    
For multi-GPU:
    python -m torch.distributed.launch --nproc_per_node=4 evaluate.py --checkpoint checkpoints/best_model.pth
"""

import torch
import argparse
import sys
from pathlib import Path
import json
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Import from train.py
from train import (
    setup_distributed, cleanup_distributed, validate_per_dataset,
    config, FaceXFormer, MultiTaskLoss
)
from scripts.small_run_common import build_eval_datasets, load_checkpoint_if_available, resolve_dataset_root


DEFAULT_TASKS = [
    "segmentation",
    "landmark",
    "headpose",
    "attribute",
    "age",
    "gender",
    "race",
    "visibility",
]


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate FaceXFormer-main on test set')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--dataset-root', default='datasets', help='Repo-local dataset root. Nested datasets/datasets is auto-detected.')
    parser.add_argument('--tasks', nargs='+', default=DEFAULT_TASKS, choices=DEFAULT_TASKS, help='Tasks to evaluate.')
    parser.add_argument('--max-samples', type=int, default=0, help='Max samples per dataset. Use 0 for all available samples.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--allow-partial-checkpoint', action='store_true', help='Load only shape-compatible checkpoint tensors.')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return
    
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"FaceXFormer-main Test Evaluation")
        print(f"{'='*60}")
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Tasks: {', '.join(args.tasks)}")
        print(f"Number of GPUs: {world_size}")
        print(f"{'='*60}\n")
    
    # Set device
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    # Load test datasets
    if rank == 0:
        print("Loading test datasets...")
    
    try:
        dataset_root = resolve_dataset_root(Path(args.dataset_root))
        test_datasets, missing = build_eval_datasets(dataset_root, args.tasks, args.max_samples, args.seed)
        if missing and rank == 0:
            print(f"Skipped unavailable datasets: {', '.join(missing)}")
        if not test_datasets:
            raise FileNotFoundError(f"No requested evaluation datasets found under {dataset_root}")
        
        if rank == 0:
            print(f"Dataset root: {dataset_root}")
            print(f"✓ All test datasets loaded successfully\n")
    
    except FileNotFoundError as e:
        if rank == 0:
            print(f"\n❌ Dataset loading error: {e}")
            print(f"\nPlease ensure datasets are downloaded and extracted to {config.DATASET_ROOT}/")
        cleanup_distributed()
        return
    
    # Create model
    if rank == 0:
        print("Creating model...")
    
    model = FaceXFormer().to(device)
    
    # Load checkpoint
    if rank == 0:
        print(f"Loading checkpoint: {args.checkpoint}")
    
    loaded = load_checkpoint_if_available(
        model,
        args.checkpoint,
        device,
        strict=not args.allow_partial_checkpoint,
    )
    
    
    
    if rank == 0:
        status = "loaded" if loaded else "not loaded"
        print(f"Checkpoint {status}\n")
    
    # Wrap model with DDP if multi-GPU
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[local_rank], 
            output_device=local_rank,
            find_unused_parameters=True
        )
    
    # Create loss function
    criterion = MultiTaskLoss(config.LOSS_WEIGHTS).to(device)
    
    # Run evaluation
    test_results = validate_per_dataset(
        model, test_datasets, criterion, device, rank, world_size
    )
    
    # Save results (rank 0 only)
    if rank == 0:
        import json
        output_dir = Path(args.checkpoint).parent
        results_file = output_dir / 'test_results.json'
        
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2, cls=NumpyEncoder)
        
        print(f"\n✓ Test results saved to: {results_file}")
        
        # Print summary table
        print(f"\n{'='*100}")
        print("📊 TEST SET SUMMARY")
        print(f"{'='*100}")
        print(f"{'Task':<20} {'Dataset':<30} {'Loss':<15} {'Metric':<15} {'Value':<15}")
        print(f"{'─'*100}")
        
        metric_names = {
            'segmentation': 'F1-Score',
            'landmark': 'NME (%)',
            'headpose': 'MAE (deg)',
            'attribute': 'Accuracy',
            'age': 'MAE (years)',
            'gender': 'Accuracy',
            'race': 'Accuracy',
            'visibility': 'Recall@P80'
        }
        
        for task_name, task_results in test_results.items():
            if task_name == 'total':
                continue
            
            metric_name = metric_names[task_name]
            
            first = True
            for dataset_name, metrics in task_results.items():
                if dataset_name == 'overall':
                    continue
                value = metrics.get('normalized_metric', metrics['metric'])
                unit = metrics.get('normalized_metric_unit', '')
                gap = metrics.get('normalized_gap_metric_minus_target')
                suffix = f" {unit}" if unit else ""
                if gap is not None:
                    suffix += f" gap={gap:.4f}"
                
                if first:
                    print(f"{task_name:<20} {dataset_name:<30} {metrics['loss']:.6f}      {metric_name:<15} {value:.4f}{suffix}")
                    first = False
                else:
                    print(f"{'':<20} {dataset_name:<30} {metrics['loss']:.6f}      {metric_name:<15} {value:.4f}{suffix}")
            
            # Print overall for this task
            if 'overall' in task_results:
                overall = task_results['overall']
                value = overall.get('normalized_metric', overall['metric'])
                unit = overall.get('normalized_metric_unit', '')
                suffix = f" {unit}" if unit else ""
                print(f"{'':<20} {'[OVERALL]':<30} {overall['loss']:.6f}      {metric_name:<15} {value:.4f}{suffix}")
            print(f"{'─'*100}")
        
        print(f"{'TOTAL (All Tasks)':<20} {'':<30} {test_results['total']:.6f}")
        print(f"{'='*100}\n")
    
    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    main()
