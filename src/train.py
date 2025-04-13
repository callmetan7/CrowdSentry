import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset
from src.dataset import CrowdDataset
from src.model import MCNN
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR


def train(
        data_dir_A="data/processed/part_A",
        data_dir_B="data/processed/part_B",
        model_path="mcnn_combined.pth",
        pretrained_model="./models/basicModel.pth",                  # Path to pre-trained model for fine-tuning
        epochs=30,
        batch_size=8,
        learning_rate=1e-4,
        validation_split=0.2
):
    """
    Trains or fine-tunes the MCNN model.

    Args:
        data_dir_A (str): Path to Part_A data.
        data_dir_B (str): Path to Part_B data.
        model_path (str): Path to save the trained model.
        pretrained_model (str): Path to a pre-trained model to fine-tune.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.
        validation_split (float): Proportion of data used for validation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data augmentation and preprocessing
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
        transforms.ToTensor()
    ])

    # Load datasets for Part_A and Part_B
    dataset_A = CrowdDataset(data_dir_A, transform=transform)
    dataset_B = CrowdDataset(data_dir_B, transform=transform)

    # Combine Part_A and Part_B datasets
    full_dataset = ConcatDataset([dataset_A, dataset_B])

    # Split into training and validation datasets
    train_size = int((1 - validation_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model
    model = MCNN().to(device)
    if pretrained_model:
        print(f"Loading pre-trained model from {pretrained_model}")
        model.load_state_dict(torch.load(pretrained_model))

    # Loss, optimizer, and scheduler
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)

    best_val_loss = float('inf')
    patience, patience_counter = 10, 0

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, density_maps in train_loader:
            images, density_maps = images.to(device), density_maps.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, density_maps)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, density_maps in val_loader:
                images, density_maps = images.to(device), density_maps.to(device)
                outputs = model(images)
                loss = criterion(outputs, density_maps)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {avg_val_loss}")

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            print(f"Best model saved at epoch {epoch+1} with validation loss: {best_val_loss}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

        scheduler.step()

    print("Training complete.")