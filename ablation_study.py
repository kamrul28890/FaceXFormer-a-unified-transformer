"""
Full ablation study for FaceXFormer-main on the complete dataset.

Mirrors train.py exactly for dataset loading, distributed setup, and the
training loop. The only differences are:
  - A single ablation variant is selected via --variant (one SLURM job per variant)
  - apply_variant() modifies the model before DDP wrapping
  - Checkpoints and results are saved under results/ablation_full/<variant>/

Supported variants:
  full                  - unmodified bidirectional cross-attention model
  standard_cross_attention - cross_attn_image_to_token zeroed out (one direction)
  unbalanced_sampler    - balanced-batch sampler disabled
  uniform_loss          - all task loss weights set to 1.0

Usage (single-GPU local test):
    python ablation_study.py --variant full

Usage via SLURM (see submit_ablation_full.slurm):
    sbatch submit_ablation_full.slurm
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

warnings.filterwarnings('ignore', message='Grad strides do not match bucket view strides')

from config import config
from datasets import (
    BIWIDataset,
    CelebADataset,
    CelebAMaskHQDataset,
    COFWDataset,
    FairFaceDataset,
    LFWADataset,
    MultiLabelDatasetWrapper,
    UTKFaceDataset,
    W300Dataset,
    W300LPDataset,
    W300VWDataset,
    create_multi_task_dataloader,
)
from losses import MultiTaskLoss
from network.models.facexformer import FaceXFormer
from train import (
    NumpyEncoder,
    cleanup_distributed,
    save_checkpoint,
    setup_distributed,
    train_one_epoch,
    validate_per_dataset,
)


VALID_VARIANTS = ["full", "standard_cross_attention", "unbalanced_sampler", "uniform_loss"]


# ---------------------------------------------------------------------------
# Variant helpers (same logic as scripts/ablation_study_tiny.py)
# ---------------------------------------------------------------------------

class ZeroAttention(torch.nn.Module):
    """Replaces cross_attn_image_to_token with zeros to simulate standard cross-attention."""

    def forward(self, q, k, v):
        return torch.zeros_like(q)


def apply_variant(model: torch.nn.Module, variant: str) -> None:
    """Mutate model in-place for the named ablation variant. Call before DDP wrapping."""
    if variant == "standard_cross_attention":
        for layer in model.face_decoder.transformer.layers:
            layer.cross_attn_image_to_token = ZeroAttention()
            # norm4 is the LayerNorm paired with cross_attn_image_to_token.  In true
            # standard cross-attention this entire block doesn't exist, so image
            # embeddings must pass through unchanged.  Leaving norm4 active
            # normalises the keys every layer even when attention output is zero,
            # which keeps embedding statistics similar to the full model and
            # artificially narrows the performance gap.
            layer.norm4 = torch.nn.Identity()
    elif variant in {"full", "unbalanced_sampler", "uniform_loss"}:
        pass
    else:
        raise ValueError(f"Unknown variant: {variant!r}. Choose from {VALID_VARIANTS}")


def loss_weights_for_variant(variant: str) -> dict:
    if variant == "uniform_loss":
        return {key: 1.0 for key in config.LOSS_WEIGHTS}
    return dict(config.LOSS_WEIGHTS)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Full ablation study for FaceXFormer-main.")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="Local rank set by torchrun (do not set manually).")
    parser.add_argument("--variant", default="full", choices=VALID_VARIANTS,
                        help="Ablation variant to run. Submit one SLURM job per variant.")
    parser.add_argument("--dataset-root", default="../facexformer-my/datasets",
                        help="Path to dataset root. Defaults to sibling repo ../facexformer-my/datasets.")
    parser.add_argument("--resume", default=None,
                        help="Resume from a checkpoint inside the variant's output directory.")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS,
                        help=f"Number of training epochs (default: {config.NUM_EPOCHS} from config).")
    parser.add_argument("--output-dir", default="results/ablation_full",
                        help="Root output directory; variant name is appended automatically.")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Distributed setup (identical to train.py)
    # ------------------------------------------------------------------
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    variant = args.variant
    output_dir = Path(args.output_dir) / variant
    checkpoint_dir = output_dir / "checkpoints"

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"FaceXFormer-main Full Ablation Study")
        print(f"{'='*60}")
        print(f"Variant:          {variant}")
        print(f"Dataset root:     {args.dataset_root}")
        print(f"Number of GPUs:   {world_size}")
        print(f"Batch size/GPU:   {config.BATCH_SIZE}")
        print(f"Effective batch:  {config.BATCH_SIZE * world_size}")
        print(f"Learning rate:    {config.LEARNING_RATE}")
        print(f"Epochs:           {args.epochs}")
        print(f"Output dir:       {output_dir}")
        print(f"{'='*60}\n")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Dataset loading (identical structure to train.py)
    # ------------------------------------------------------------------
    dataset_root = args.dataset_root

    if rank == 0:
        print("Loading datasets...")

    try:
        celebamask_train = CelebAMaskHQDataset('train', dataset_root=dataset_root)
        celebamask_test  = CelebAMaskHQDataset('test',  dataset_root=dataset_root)

        w300_train = W300Dataset('train', dataset_root=dataset_root)
        w300_test  = W300Dataset('test',  dataset_root=dataset_root)

        w300lp_train = W300LPDataset('train', dataset_root=dataset_root)

        celeba_train = CelebADataset('train', dataset_root=dataset_root, rank=rank, world_size=world_size)
        celeba_test  = CelebADataset('test',  dataset_root=dataset_root, rank=rank, world_size=world_size)

        utkface_train = UTKFaceDataset('train', dataset_root=dataset_root)
        utkface_test  = UTKFaceDataset('test',  dataset_root=dataset_root)

        fairface_train = FairFaceDataset('train', dataset_root=dataset_root)
        fairface_test  = FairFaceDataset('test',  dataset_root=dataset_root)

        cofw_train = COFWDataset('train', dataset_root=dataset_root)
        cofw_test  = COFWDataset('test',  dataset_root=dataset_root)

        w300vw_test = W300VWDataset('test', dataset_root=dataset_root)
        biwi_test   = BIWIDataset('test',  dataset_root=dataset_root)
        lfwa_test   = LFWADataset('test',  dataset_root=dataset_root)

    except FileNotFoundError as exc:
        if rank == 0:
            print(f"\nDataset loading error: {exc}")
            print("Check --dataset-root and ensure datasets are present.")
        cleanup_distributed()
        return

    train_datasets = {
        'segmentation': [celebamask_train],
        'landmark':     [w300_train],
        'headpose':     [w300lp_train],
        'attribute':    [celeba_train],
        'age':          [MultiLabelDatasetWrapper(utkface_train, 'age'),
                         MultiLabelDatasetWrapper(fairface_train, 'age')],
        'gender':       [MultiLabelDatasetWrapper(utkface_train, 'gender'),
                         MultiLabelDatasetWrapper(fairface_train, 'gender')],
        'race':         [MultiLabelDatasetWrapper(utkface_train, 'race'),
                         MultiLabelDatasetWrapper(fairface_train, 'race')],
        'visibility':   [cofw_train],
    }

    test_datasets = {
        'segmentation': [celebamask_test],
        'landmark':     [w300_test, w300vw_test],
        'headpose':     [biwi_test],
        'attribute':    [celeba_test, lfwa_test],
        'age':          [MultiLabelDatasetWrapper(utkface_test, 'age'),
                         MultiLabelDatasetWrapper(fairface_test, 'age')],
        'gender':       [MultiLabelDatasetWrapper(utkface_test, 'gender'),
                         MultiLabelDatasetWrapper(fairface_test, 'gender')],
        'race':         [MultiLabelDatasetWrapper(utkface_test, 'race'),
                         MultiLabelDatasetWrapper(fairface_test, 'race')],
        'visibility':   [cofw_test],
    }

    if rank == 0:
        total_train = sum(sum(len(ds) for ds in v) for v in train_datasets.values())
        print(f"All datasets loaded. Total training samples: {total_train:,}")

    # ------------------------------------------------------------------
    # Dataloaders — pass use_balanced_batches=False for unbalanced_sampler
    # ------------------------------------------------------------------
    use_balanced = variant != "unbalanced_sampler"
    # unbalanced_sampler must also skip upsampling; otherwise UpsampledMultiTaskDataset
    # equalises task counts before batching, making the variant identical to 'full'.
    use_upsampling = variant != "unbalanced_sampler"

    train_loader = create_multi_task_dataloader(
        train_datasets,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        use_upsampling=use_upsampling,
        rank=rank,
        world_size=world_size,
        use_balanced_batches=use_balanced,
    )

    # ------------------------------------------------------------------
    # Model, variant modification, DDP wrapping
    # ------------------------------------------------------------------
    if rank == 0:
        print(f"\nCreating model (variant: {variant})...")

    model = FaceXFormer().to(device)
    apply_variant(model, variant)       # must happen BEFORE DDP wrapping

    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model ready. Parameters: {total_params:,}")

    # ------------------------------------------------------------------
    # Loss, optimiser, scheduler (identical to train.py)
    # ------------------------------------------------------------------
    criterion = MultiTaskLoss(loss_weights_for_variant(variant)).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    def lr_lambda(epoch):
        if epoch < config.LR_DECAY_EPOCHS[0]:
            return 1.0
        elif epoch < config.LR_DECAY_EPOCHS[1]:
            return config.LR_DECAY_FACTOR
        return config.LR_DECAY_FACTOR ** 2

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # ------------------------------------------------------------------
    # Optional resume
    # ------------------------------------------------------------------
    start_epoch = 1
    if args.resume:
        if rank == 0:
            print(f"\nResuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        state = checkpoint['model_state_dict']
        if isinstance(model, DDP):
            model.module.load_state_dict(state)
        else:
            model.load_state_dict(state)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint.get('scheduler_state_dict') and scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if rank == 0:
            print(f"Resumed from epoch {checkpoint['epoch']}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_train_loss = float('inf')

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            if world_size > 1:
                if hasattr(train_loader, 'batch_sampler') and hasattr(train_loader.batch_sampler, 'set_epoch'):
                    train_loader.batch_sampler.set_epoch(epoch)
                elif hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
                    train_loader.sampler.set_epoch(epoch)

            if rank == 0:
                print(f"\n{'='*60}\nEpoch {epoch}/{args.epochs}  [variant: {variant}]\n{'='*60}")

            if world_size > 1:
                dist.barrier()

            train_loss, train_task_losses = train_one_epoch(
                model, train_loader, criterion, optimizer, epoch, device, rank
            )

            if rank == 0:
                print(f"\nTrain Loss: {train_loss:.4f}")
                for task_name, task_loss in train_task_losses.items():
                    print(f"  {task_name}: {task_loss:.4f}")

            scheduler.step()

            if rank == 0:
                if train_loss < best_train_loss:
                    best_train_loss = train_loss
                    save_checkpoint(
                        model, optimizer, scheduler, epoch, train_loss,
                        str(checkpoint_dir / 'best_model.pth'),
                        rank,
                    )
                if epoch % config.SAVE_FREQ == 0:
                    save_checkpoint(
                        model, optimizer, scheduler, epoch, train_loss,
                        str(checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'),
                        rank,
                    )

        # ------------------------------------------------------------------
        # Final evaluation (identical to train.py)
        # ------------------------------------------------------------------
        if rank == 0:
            print(f"\n{'='*60}\nTraining complete. Best train loss: {best_train_loss:.4f}\n{'='*60}\n")

        if world_size > 1:
            dist.barrier()

        test_results = validate_per_dataset(
            model, test_datasets, criterion, device, rank, world_size
        )

        if rank == 0:
            results_file = output_dir / 'test_results.json'
            with results_file.open('w') as f:
                json.dump(test_results, f, indent=2, cls=NumpyEncoder)
            print(f"\nTest results saved to: {results_file}")

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
