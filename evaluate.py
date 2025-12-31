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
from pathlib import Path

# Import from train.py
from train import (
    setup_distributed, cleanup_distributed, validate_per_dataset,
    config, FaceXFormer, MultiTaskLoss
)
from datasets import (
    CelebAMaskHQDataset, W300Dataset, W300LPDataset, CelebADataset,
    UTKFaceDataset, FairFaceDataset, COFWDataset, W300VWDataset,
    BIWIDataset, LFWADataset, MultiLabelDatasetWrapper
)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate FaceXFormer-main on test set')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
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
        print(f"Number of GPUs: {world_size}")
        print(f"{'='*60}\n")
    
    # Set device
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    # Load test datasets
    if rank == 0:
        print("Loading test datasets...")
    
    try:
        celebamask_test = CelebAMaskHQDataset('test')
        w300_test = W300Dataset('test')
        w300vw_test = W300VWDataset('test')
        biwi_test = BIWIDataset('test')
        celeba_test = CelebADataset('test', rank=rank, world_size=world_size)
        lfwa_test = LFWADataset('test')
        utkface_test = UTKFaceDataset('test')
        fairface_test = FairFaceDataset('test')
        cofw_test = COFWDataset('test')
        
        test_datasets = {
            'segmentation': [celebamask_test],
            'landmark': [w300_test, w300vw_test],
            'headpose': [biwi_test],
            'attribute': [celeba_test, lfwa_test],
            'age': [MultiLabelDatasetWrapper(utkface_test, 'age'),
                    MultiLabelDatasetWrapper(fairface_test, 'age')],
            'gender': [MultiLabelDatasetWrapper(utkface_test, 'gender'),
                       MultiLabelDatasetWrapper(fairface_test, 'gender')],
            'race': [MultiLabelDatasetWrapper(utkface_test, 'race'),
                     MultiLabelDatasetWrapper(fairface_test, 'race')],
            'visibility': [cofw_test]
        }
        
        if rank == 0:
            print(f"✓ All test datasets loaded successfully\n")
    
    except FileNotFoundError as e:
        if rank == 0:
            print(f"\n❌ Dataset loading error: {e}")
            print("\nPlease ensure datasets are downloaded and extracted to ./datasets/")
        cleanup_distributed()
        return
    
    # Create model
    if rank == 0:
        print("Creating model...")
    
    model = FaceXFormer().to(device)
    
    # Load checkpoint
    if rank == 0:
        print(f"Loading checkpoint: {args.checkpoint}")
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Handle DDP vs non-DDP checkpoint
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Remove 'module.' prefix if present (from DDP)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')  # remove 'module.' prefix
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    
    if rank == 0:
        epoch = checkpoint.get('epoch', 'unknown')
        train_loss = checkpoint.get('train_loss', 'unknown')
        print(f"✓ Checkpoint loaded (epoch: {epoch}, train_loss: {train_loss})\n")
    
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
            json.dump(test_results, f, indent=2)
        
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
                
                if first:
                    print(f"{task_name:<20} {dataset_name:<30} {metrics['loss']:.6f}      {metric_name:<15} {metrics['metric']:.4f}")
                    first = False
                else:
                    print(f"{'':<20} {dataset_name:<30} {metrics['loss']:.6f}      {metric_name:<15} {metrics['metric']:.4f}")
            
            # Print overall for this task
            if 'overall' in task_results:
                print(f"{'':<20} {'[OVERALL]':<30} {task_results['overall']['loss']:.6f}      {metric_name:<15} {task_results['overall']['metric']:.4f}")
            print(f"{'─'*100}")
        
        print(f"{'TOTAL (All Tasks)':<20} {'':<30} {test_results['total']:.6f}")
        print(f"{'='*100}\n")
    
    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    main()
