import torch
import torch.nn.functional as F
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scienceplots
import beaupy
from rich.console import Console
from scipy.interpolate import griddata

from util import (
    select_project,
    select_group,
    select_seed,
    select_device,
    load_model,
    load_data,
    load_study,
    load_best_model,
)


def test_model(model, dl_val, device):
    model.eval()
    total_loss = 0
    all_x1 = []
    all_x2 = []
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in dl_val:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = F.mse_loss(y_pred, y)
            total_loss += loss.item()
            all_x1.extend(x[0:1].cpu().numpy())
            all_x2.extend(x[1:2].cpu().numpy())
            all_preds.extend(y_pred.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    return total_loss / len(dl_val), np.array(all_x1), np.array(all_x2), np.array(all_preds), np.array(all_targets)


def main():
    # Test run
    console.print("[bold green]Analyzing the model...[/bold green]")
    console.print("Select a project to analyze:")
    project = select_project()
    console.print("Select a group to analyze:")
    group_name = select_group(project)
    console.print("Select a seed to analyze:")
    seed = select_seed(project, group_name)
    console.print("Select a device:")
    device = select_device()
    model, config = load_model(project, group_name, seed)
    model = model.to(device)

    # Load the best model
    # study_name = "Optimize_Template"
    # model, config = load_best_model(project, study_name)
    # device = select_device()
    # model = model.to(device)

    _, dl_val = load_data()  # Assuming this is implemented in util.py

    val_loss, x1, x2, preds, targets = test_model(model, dl_val, device)
    print(f"Validation Loss: {val_loss}")

    # Additional custom analysis can be added here

    # Plotting
    # Sort the data for better visualization
    #indices = np.argsort(x1)
    #x1 = x1[indices]
    #x2 = x2[indices]
    #preds = preds[indices]
    #targets = targets[indices]
    with plt.style.context(["science", "nature"]):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(x1, x2, targets, '.', label="Data", alpha=0.5, color="blue", ms=5, mew=0)
        ax.plot(x1, x2, preds, 'r.', label="Model", alpha=0.5, ms=5, mew=0)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend()
        ax.autoscale(tight=True)
        fig.savefig("predictions_2.png", dpi=600, bbox_inches="tight")

    abs_error = np.abs(targets - preds).flatten()
    with plt.style.context(["science", "nature"]):
        fig, ax = plt.subplots()
        ax.plot(x1, abs_error, '.', label="Absolute Error", alpha=0.5, color="blue", ms=4, mew=0)
        ax.set_xlabel("x1")
        ax.set_ylabel("Absolute Error")
        ax.set_ylim(0, 1)
        ax.legend()
        fig.savefig("abs_error_x1_2.png", dpi=600, bbox_inches="tight")


if __name__ == "__main__":
    console = Console()
    main()
