"""
Training loop for Stage 1 and Stage 2.
Includes: learning rate warmup, cosine annealing, early stopping,
checkpoint saving, and per-epoch logging.
"""

import os
import time
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, epoch):
    model.train()
    running_loss = correct = total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        # Mixed precision forward pass — faster on Colab GPU
        with autocast():
            outputs = model(images)
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()

        # Gradient clipping — stabilizes ViT training
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted  = outputs.max(1)
        total        += labels.size(0)
        correct      += predicted.eq(labels).sum().item()

        if batch_idx % 50 == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx}/{len(loader)} "
                  f"| Loss: {loss.item():.4f} "
                  f"| Acc: {100.*correct/total:.2f}%")

    return running_loss / len(loader), 100. * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = correct = total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with autocast():
            outputs = model(images)
            loss    = criterion(outputs, labels)

        running_loss += loss.item()
        _, predicted  = outputs.max(1)
        total        += labels.size(0)
        correct      += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100. * correct / total


def train(
    model,
    train_loader,
    val_loader,
    save_path: str,
    stage: int = 1,
    num_epochs: int = 30,
    lr: float = 1e-4,
    weight_decay: float = 0.05,
    warmup_epochs: int = 5,
    patience: int = 7,
    class_weights: torch.Tensor = None,
    device: str = 'cuda'
):
    """
    Full training loop with warmup + cosine LR, early stopping,
    mixed precision, and best-model checkpointing.

    Args:
        model          : PyTorch model
        train_loader   : training DataLoader
        val_loader     : validation DataLoader
        save_path      : directory to save checkpoints
        stage          : 1 or 2 (affects LR and logging)
        num_epochs     : max epochs
        lr             : peak learning rate
        weight_decay   : AdamW weight decay
        warmup_epochs  : linear warmup before cosine decay
        patience       : early stopping patience
        class_weights  : tensor of per-class weights for loss
        device         : 'cuda' or 'cpu'
    """
    os.makedirs(save_path, exist_ok=True)
    model = model.to(device)

    # Loss with optional class weighting
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # AdamW — standard for ViT training
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )

    # LR schedule: linear warmup then cosine annealing
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor = 0.1,
        end_factor   = 1.0,
        total_iters  = warmup_epochs
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max  = num_epochs - warmup_epochs,
        eta_min = lr * 0.01
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers = [warmup_scheduler, cosine_scheduler],
        milestones = [warmup_epochs]
    )

    # Mixed precision scaler
    scaler = GradScaler()

    # Tracking
    best_val_acc  = 0.0
    best_epoch    = 0
    patience_ctr  = 0
    history       = []

    print(f"\n{'='*60}")
    print(f"Stage {stage} Training")
    print(f"Epochs: {num_epochs} | LR: {lr} | Warmup: {warmup_epochs}")
    print(f"Device: {device} | Early stop patience: {patience}")
    print(f"{'='*60}\n")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, epoch
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch:03d}/{num_epochs} | "
              f"Time: {elapsed:.0f}s | "
              f"LR: {current_lr:.6f}")
        print(f"  Train → Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"  Val   → Loss: {val_loss:.4f}   | Acc: {val_acc:.2f}%")

        # Log history
        history.append({
            'epoch': epoch, 'lr': current_lr,
            'train_loss': train_loss, 'train_acc': train_acc,
            'val_loss': val_loss,   'val_acc': val_acc
        })

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            patience_ctr = 0
            ckpt_path = os.path.join(save_path, f'stage{stage}_best.pth')
            torch.save({
                'epoch'            : epoch,
                'model_state_dict' : model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc'          : val_acc,
                'val_loss'         : val_loss,
            }, ckpt_path)
            print(f"  >>> New best saved: {val_acc:.2f}% at epoch {epoch}")
        else:
            patience_ctr += 1
            print(f"  No improvement ({patience_ctr}/{patience})")

        # Early stopping
        if patience_ctr >= patience:
            print(f"\nEarly stopping at epoch {epoch}. "
                  f"Best val acc: {best_val_acc:.2f}% at epoch {best_epoch}")
            break

    # Save training history as JSON for plotting in paper
    history_path = os.path.join(save_path, f'stage{stage}_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nHistory saved to {history_path}")
    print(f"Best val accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")

    return history, best_val_acc