import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import KFold

class CoffeeDataset(Dataset):
    def __init__(self, root_dir="dataset_kopi", transform=None, split="train", fold_idx=0, n_folds=5, seed=42, fastmode=False):
        """
        Args:
            root_dir (string): Directory with all the images organized in class folders
            transform (callable, optional): Optional transform to be applied on a sample
            split (string): 'train' or 'val' to specify the dataset split
            fold_idx (int): Current fold index (0 to n_folds-1)
            n_folds (int): Number of folds for cross-validation
            seed (int): Random seed for reproducibility
            fastmode (bool): If True, use a small subset of data for quick testing
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.fold_idx = fold_idx
        self.n_folds = n_folds
        self.fastmode = fastmode
        
        # Get all class folders (labels)
        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.classes.sort()  # Sort to ensure consistent class indices
        
        # Create class to index mapping
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Collect all image paths and their corresponding labels
        self.image_paths = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            image_files = [img_name for img_name in os.listdir(class_dir) 
                          if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            
            # In fast mode, use only a small subset of images per class
            if fastmode:
                # Use at most 5 images per class for quick testing
                image_files = image_files[:5]
                
            for img_name in image_files:
                self.image_paths.append(os.path.join(class_dir, img_name))
                self.labels.append(class_idx)
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        # Use KFold from scikit-learn for consistent fold splitting
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        
        # Convert to numpy arrays for indexing
        indices = np.arange(len(self.image_paths))
        
        # Get the train and validation indices for the current fold
        fold_splits = list(kf.split(indices))
        train_indices, val_indices = fold_splits[fold_idx]
        
        # Assign the appropriate indices based on the split
        if self.split == "train":
            self.indices = indices[train_indices]
        else:  # validation set
            self.indices = indices[val_indices]
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Use the mapped index
        actual_idx = self.indices[idx]
        
        img_path = self.image_paths[actual_idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[actual_idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_data_loaders(batch_size=32, num_workers=4, fold_idx=0, n_folds=5, fastmode=False):
    """
    Create data loaders for training and validation for a specific fold
    
    Args:
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for data loading
        fold_idx (int): Index of the current fold
        n_folds (int): Total number of folds
        fastmode (bool): If True, use a small subset of data for quick testing
        
    Returns:
        train_loader, val_loader: DataLoader objects for training and validation
    """
    # Define transforms for training and validation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets for the specific fold
    train_dataset = CoffeeDataset(split="train", transform=train_transform, fold_idx=fold_idx, n_folds=n_folds, fastmode=fastmode)
    val_dataset = CoffeeDataset(split="val", transform=val_transform, fold_idx=fold_idx, n_folds=n_folds, fastmode=fastmode)
    
    if fastmode:
        print(f"Fast mode enabled: Using {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset.classes


# Example usage
if __name__ == "__main__":
    # Test for fold 0 of 5
    train_loader, val_loader, classes = get_data_loaders(fold_idx=0, n_folds=5)
    print(f"Classes: {classes}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    
    # Get a sample batch
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Verify different folds have different splits
    for fold in range(5):
        train_loader, val_loader, _ = get_data_loaders(fold_idx=fold, n_folds=5)
        print(f"Fold {fold}: Train batches={len(train_loader)}, Val batches={len(val_loader)}")

    try:
        loaders = get_data_loaders(fold_idx=0, n_folds=5)
        print("Success: get_data_loaders accepts fold_idx and n_folds parameters")
    except TypeError as e:
        print(f"Error: {e}")
