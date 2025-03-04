import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb

from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights,
    resnet152, ResNet152_Weights,
    vit_b_16, ViT_B_16_Weights
)

from datareader import get_data_loaders
from utils import check_set_gpu

def get_model(model_name, num_classes):
    """
    Create a model with pretrained weights and modified classifier layer
    
    Args:
        model_name (str): Name of the model to use
        num_classes (int): Number of output classes
        
    Returns:
        model: PyTorch model
    """
    if model_name == "efficientnet":
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        model = efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
    elif model_name == "shufflenet":
        weights = ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1
        model = shufflenet_v2_x0_5(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == "resnet152":
        weights = ResNet152_Weights.IMAGENET1K_V1
        model = resnet152(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == "vit":
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        model = vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model

def train_one_epoch(model, train_loader, criterion, optimizer, epoch, device):
    """Train the model for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (i+1) % 10 == 0:  # Print every 10 mini-batches
            print(f'Epoch: {epoch+1}, Batch: {i+1}/{len(train_loader)}, '
                  f'Loss: {running_loss/(i+1):.4f}, '
                  f'Acc: {100.*correct/total:.2f}%')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    return val_loss, val_acc

def save_checkpoint(model, optimizer, epoch, val_acc, filename):
    """Save model checkpoint"""
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'val_acc': val_acc
    }
    torch.save(state, filename)

def plot_training_history(train_losses, val_losses, train_accs, val_accs, filename='training_history.png'):
    """Plot training and validation history"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curves')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def train_one_fold(model_name, fold_idx, n_folds, batch_size, lr, epochs, patience, device, use_wandb=True, run_id=None, fastmode=False):
    """Train a model on one fold"""
    # Generate fold-specific name
    fold_name = f"fold_{fold_idx+1}_of_{n_folds}"
    
    # Join the main run with a fold-specific group if using wandb
    if use_wandb:
        if run_id is None:
            # This is the first fold, create a new wandb run
            wandb_run = wandb.init(
                project="coffee-classification",
                name=f"{model_name}_{datetime.now().strftime('%Y%m%d-%H%M')}",
                group=f"{model_name}_5fold_{datetime.now().strftime('%Y%m%d-%H%M')}",
                config={
                    "model": model_name,
                    "batch_size": batch_size,
                    "learning_rate": lr,
                    "epochs": epochs,
                    "patience": patience,
                    "n_folds": n_folds
                }
            )
            run_id = wandb.run.id
        else:
            # Continue the existing run with a new name
            wandb_run = wandb.init(
                project="coffee-classification",
                name=f"{model_name}_{fold_name}",
                group=f"{model_name}_5fold_{datetime.now().strftime('%Y%m%d-%H%M')}",
                id=run_id,
                resume="allow"
            )
    
    print(f"\n{'='*20}\nTraining fold {fold_idx+1} of {n_folds}\n{'='*20}")
    
    # Create output directory
    os.makedirs('models', exist_ok=True)
    
    # Get data loaders for this specific fold
    train_loader, val_loader, classes = get_data_loaders(
        batch_size=batch_size, fold_idx=fold_idx, n_folds=n_folds, fastmode=fastmode
    )
    num_classes = len(classes)
    
    # Get model
    model = get_model(model_name, num_classes)
    model = model.to(device)
    
    # Log model architecture to wandb if enabled
    if use_wandb:
        wandb.watch(model, log="all")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Initialize variables
    best_val_acc = 0.0
    epochs_no_improve = 0
    early_stop = False
    
    # History for plotting
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    # Track the previous learning rate for manual verbose output
    prev_lr = optimizer.param_groups[0]['lr']
    
    # In fast mode, reduce the number of epochs
    if fastmode:
        print(f"Fast mode enabled: Running only {min(3, epochs)} epochs")
        epochs = min(3, epochs)
    
    start_time = time.time()
    for epoch in range(epochs):
        if early_stop:
            print("Early stopping triggered!")
            break
            
        print(f"\nFold {fold_idx+1}/{n_folds}, Epoch {epoch+1}/{epochs}")
        print("-" * 20)
        
        # Train and validate
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Manually check if learning rate changed and log it
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != prev_lr:
            print(f"Learning rate changed from {prev_lr:.6f} to {current_lr:.6f}")
            prev_lr = current_lr
        
        # Save history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Log metrics to wandb if enabled
        if use_wandb:
            wandb.log({
                "fold": fold_idx + 1,
                f"fold_{fold_idx+1}/train_loss": train_loss,
                f"fold_{fold_idx+1}/train_acc": train_acc,
                f"fold_{fold_idx+1}/val_loss": val_loss,
                f"fold_{fold_idx+1}/val_acc": val_acc,
                f"fold_{fold_idx+1}/learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch + 1
            })
        
        # Print epoch summary
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save model if validation accuracy improves
        if val_acc > best_val_acc:
            print(f"Validation accuracy improved from {best_val_acc:.2f}% to {val_acc:.2f}%")
            best_val_acc = val_acc
            model_filename = f"models/{model_name}_fold{fold_idx+1}of{n_folds}_best.pth"
            save_checkpoint(model, optimizer, epoch, val_acc, model_filename)
            
            # Save best model to wandb if enabled
            if use_wandb:
                wandb.save(model_filename)
                
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs")
            
        # Early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement")
            early_stop = True
    
    # Plot training history for this fold
    plot_filename = f"training_history_fold{fold_idx+1}.png"
    plot_training_history(train_losses, val_losses, train_accs, val_accs, filename=plot_filename)
    if use_wandb:
        wandb.log({f"fold_{fold_idx+1}/training_history": wandb.Image(plot_filename)})
    
    # Close wandb run for this fold
    if use_wandb and fold_idx < n_folds - 1:  # Don't finish the last fold yet
        wandb.finish()
    
    # Return best validation accuracy for this fold
    return best_val_acc, run_id

def train_kfold(model_name, batch_size=32, lr=0.001, epochs=30, patience=5, device_override=None, use_wandb=True, n_folds=5, fastmode=False):
    """Train with k-fold cross-validation"""
    # Set device using the utility function
    device = check_set_gpu(device_override)
    
    # Initialize list to store results for each fold
    fold_accuracies = []
    run_id = None  # For wandb continuity between folds
    
    # In fast mode, reduce the number of folds if needed
    if fastmode and n_folds > 2:
        print(f"Fast mode enabled: Running only 2 folds instead of {n_folds}")
        n_folds = 2
    
    # Train each fold
    for fold_idx in range(n_folds):
        best_val_acc, run_id = train_one_fold(
            model_name, fold_idx, n_folds,
            batch_size, lr, epochs, patience,
            device, use_wandb, run_id, fastmode
        )
        fold_accuracies.append(best_val_acc)
    
    # Calculate and print cross-validation results
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    print(f"\n{'='*50}")
    print(f"Cross-Validation Results for {model_name}")
    print(f"{'='*50}")
    for fold_idx, acc in enumerate(fold_accuracies):
        print(f"Fold {fold_idx+1}: {acc:.2f}%")
    print(f"{'='*50}")
    print(f"Mean Accuracy: {mean_accuracy:.2f}% ± {std_accuracy:.2f}%")
    print(f"{'='*50}")
    
    # Log final cross-validation results to wandb
    if use_wandb:
        wandb.log({
            "cv_mean_accuracy": mean_accuracy,
            "cv_std_accuracy": std_accuracy,
            "cv_fold_accuracies": fold_accuracies
        })
        
        # Create a summary plot of fold accuracies
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, n_folds+1), fold_accuracies)
        plt.axhline(y=mean_accuracy, color='r', linestyle='-', label=f'Mean: {mean_accuracy:.2f}%')
        plt.axhline(y=mean_accuracy+std_accuracy, color='r', linestyle='--')
        plt.axhline(y=mean_accuracy-std_accuracy, color='r', linestyle='--', label=f'Std: ±{std_accuracy:.2f}%')
        plt.xlabel('Fold')
        plt.ylabel('Validation Accuracy (%)')
        plt.title(f'Cross-Validation Results - {model_name}')
        plt.legend()
        plt.tight_layout()
        plt.savefig('cv_results.png')
        wandb.log({"cv_results": wandb.Image('cv_results.png')})
        
        # Finish wandb run
        wandb.finish()
    
    return mean_accuracy, std_accuracy, fold_accuracies

def main():
    """Main function to parse arguments and start training"""
    parser = argparse.ArgumentParser(description="Train coffee classification models")
    parser.add_argument("--model", type=str, choices=["efficientnet", "shufflenet", "resnet152", "vit", "all"], 
                        default="all", help="Model architecture to use, 'all' to run all models sequentially")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--device", type=str, choices=["cuda", "mps", "cpu"], 
                        default=None, help="Device to use (overrides automatic detection)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--fastmode", action="store_true", 
                        help="Enable fast mode: uses less data, fewer epochs, and fewer folds for quick testing")
    
    args = parser.parse_args()
    
    # Define all available models
    all_models = ["efficientnet", "shufflenet", "resnet152", "vit"]
    
    # Determine which models to run
    models_to_run = all_models if args.model == "all" else [args.model]
    
    # Run training for each selected model
    for model_name in models_to_run:
        print(f"\n\n{'='*60}")
        print(f"Starting training for model: {model_name}")
        print(f"{'='*60}\n")
        
        train_kfold(model_name, args.batch_size, args.lr, args.epochs, args.patience, 
                    args.device, not args.no_wandb, args.folds, args.fastmode)
        
        print(f"\n{'='*60}")
        print(f"Completed training for model: {model_name}")
        print(f"{'='*60}\n")
        
        # Small delay between models to ensure wandb runs are separate
        time.sleep(2)

if __name__ == "__main__":
    main()
