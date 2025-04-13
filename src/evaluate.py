import torch
from torch.utils.data import DataLoader
from src.dataset import CrowdDataset
from src.model import MCNN
import numpy as np
import matplotlib.pyplot as plt

def evaluate(modelPath, dataDirA, dataDirB=None, batch=4):
    """
    Evaluates the model on the test dataset(s).

    Args:
        model_path (str): Path to the trained model.
        data_dir_A (str): Path to Part_A test dataset.
        data_dir_B (str, optional): Path to Part_B test dataset. Defaults to None.
        batch_size (int): Batch size for evaluation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = MCNN().to(device)
    model.load_state_dict(torch.load(modelPath))
    model.eval()

    # Load test datasets
    testDataA = CrowdDataset(dataDirA)
    testLoadA = DataLoader(testDataA, batch_size=batch)

    maeA, mseA = evaluate_dataset(model, testLoadA, len(testDataA), device)
    print("Results for Part_A:")
    print(f"Mean Absolute Error (MAE): {maeA}")
    print(f"Mean Squared Error (MSE): {mseA}")
    plot_metrics(maeA, mseA, "Part_A")  # Plot metrics for Part_A

    if dataDirB:
        test_dataset_B = CrowdDataset(dataDirB)
        test_loader_B = DataLoader(test_dataset_B, batch_size=batch)

        mae_B, mse_B = evaluate_dataset(model, test_loader_B, len(test_dataset_B), device)
        print("Results for Part_B:")
        print(f"Mean Absolute Error (MAE): {mae_B}")
        print(f"Mean Squared Error (MSE): {mse_B}")
        plot_metrics(mae_B, mse_B, "Part_B")  # Plot metrics for Part_B

def evaluate_dataset(model, test_loader, dataset_size, device):
    mae, mse = 0.0, 0.0

    with torch.no_grad():
        for images, density_maps in test_loader:
            images, density_maps = images.to(device), density_maps.to(device)

            # Predict density maps
            outputs = model(images)

            # Calculate errors
            mae += torch.sum(torch.abs(outputs.sum(dim=(2, 3)) - density_maps.sum(dim=(2, 3)))).item()
            mse += torch.sum((outputs.sum(dim=(2, 3)) - density_maps.sum(dim=(2, 3))) ** 2).item()

    mae /= dataset_size
    mse = (mse / dataset_size) ** 0.5

    return mae, mse

def plot_metrics(mae, mse, dataset_name):
    """
    Plots the MAE and MSE for a given dataset.

    Args:
        mae (float): Mean Absolute Error.
        mse (float): Mean Squared Error.
        dataset_name (str): Name of the dataset (e.g., "Part_A" or "Part_B").
    """
    metrics = ['MAE', 'MSE']
    values = [mae, mse]

    plt.bar(metrics, values, color=['blue', 'orange'])
    plt.title(f"Error Metrics for {dataset_name}")
    plt.ylabel("Error")
    plt.show()