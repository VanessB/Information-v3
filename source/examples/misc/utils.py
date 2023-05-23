import json
import os
import torch

import matplotlib.pyplot as plt
import pandas as pd

from collections import defaultdict
from datetime import datetime
from pathlib import Path


def show_images(images: list, labels: list[str]=None, n_cols: int=3):
    """
    Show list of images
    
    Parameters
    ----------
    images : list
    
    """
    
    if labels is None:
        labels = [f"pic. {index}" for index in range(1, len(images) + 1)]
    else:
        assert len(images) == len(labels)
        
    n_rows = len(images) // n_cols + (1 if len(images) % n_cols else 0)
    
    fig = plt.figure()
    for index, (image, label) in enumerate(zip(images, labels)):
        plt.subplot(n_rows, n_cols, index+1)
        plt.tight_layout()
        plt.imshow(image[0], cmap='gray', interpolation='none')
        plt.title(label)
        plt.xticks([])
        plt.yticks([])
    
    plt.show();
    
    
    
def plot_MI_planes(MI_X_L: dict, MI_L_Y: dict, filtered_MI_X_L: dict=None, filtered_MI_L_Y: dict=None,
                   n_columns: int=3) -> None:
    """
    Plot information plane data for each layer in a subplot.
    
    Parameters
    ----------
    MI_X_L : dict
        Raw I(X;L) data (with errorbars).
    MI_L_Y : dict
        Raw I(L;Y) data (with errorbars).
    filtered_MI_X_L : dict
        Filtered I(X;L) data.
    filtered_MI_L_Y : dict
        Filtered I(L;Y) data.
    """
    
    assert len(MI_X_L) == len(MI_L_Y)
    
    filtered_provided = (not filtered_MI_X_L is None) and (not filtered_MI_L_Y is None)
    
    # Number of rows.
    n_rows = len(MI_X_L) // n_columns + (len(MI_X_L) % n_columns != 0)
    
    width = 6
    height = 4
    fig, ax = plt.subplots(n_rows, n_columns, figsize=(width * n_columns, height * n_rows))
    for index, layer_name in enumerate(MI_X_L.keys()):
        row_index = index // n_columns
        column_index = index % n_columns
        subplot_ax = ax[row_index, column_index]
        
        subplot_ax.grid(color='#000000', alpha=0.15, linestyle='-', linewidth=1, which='major')
        subplot_ax.grid(color='#000000', alpha=0.1, linestyle='-', linewidth=0.5, which='minor')
        subplot_ax.set_title(str(layer_name))
        
        x =     [item[0] for item in MI_X_L[layer_name]]
        x_err = [item[1] for item in MI_X_L[layer_name]]
        y =     [item[0] for item in MI_L_Y[layer_name]]
        y_err = [item[1] for item in MI_L_Y[layer_name]]
        
        if filtered_provided:
            subplot_ax.errorbar(x, y, x_err, y_err, ls='none', solid_capstyle='projecting', capsize=3, alpha=0.25, color='lightblue')
            subplot_ax.plot(filtered_MI_X_L[layer_name], filtered_MI_L_Y[layer_name], color='red')
        else:
            subplot_ax.plot(x, y)
            
    plt.show();


    
def split_lists(list_of_tuples: list()) -> tuple():
    """
    List of tuples to tuple of lists.
    """
    
    assert len(list_of_tuples) > 0
    
    lists = [[] for index in range(len(list_of_tuples[0]))]
    for element in list_of_tuples:
        for index, subelement in enumerate(element):
            lists[index].append(subelement)
            
    return tuple(lists)



def get_outputs(model, dataloader, device) -> list():
    """
    Get outputs of the model.
    """
    
    # Exit training mode.
    was_in_training = model.training
    model.eval()
    
    outputs = []
    with torch.no_grad():
        for index, batch in enumerate(dataloader):
            x, y = batch
            
            outputs.append(model(x.to(device)).detach().cpu())
            
    outputs = torch.cat(outputs)
    
    # Return to the original mode.
    model.train(was_in_training)
    
    return outputs



def get_layers(model, dataloader, device) -> list():
    """
    Get outputs of all layers.
    """
    
    # Exit training mode.
    was_in_training = model.training
    model.eval()
    
    outputs = defaultdict(list)
    with torch.no_grad():
        for index, batch in enumerate(dataloader):
            x, y = batch
            
            layers = model(x.to(device), all_layers=True)
            for layer_name, layer in layers.items():
                outputs[layer_name].append(layer.detach().cpu())
            
    # "Transpose".
    outputs = {layer_name: torch.cat(batches) for layer_name, batches in outputs.items()}
    
    # Return to the original mode.
    model.train(was_in_training)
    
    return outputs



def save_results(results: dict, settings: dict, path: Path):
    """
    Save IP experiments results (parameters, metrics, IP data, ...).
    """
    
    directory_path = path / (datetime.now().strftime("%d-%b-%Y_%H:%M:%S") + "/")
    os.makedirs(directory_path, exist_ok=True)
    
    # Metrics.
    metrics = pd.DataFrame()
    for metric_name, metric_values in results['metrics'].items():
        metrics[metric_name] = metric_values
    metrics.to_csv(directory_path / "metrics.csv", index=False)
        
    # Mutual information.
    for layer_name in results["MI_X_L"].keys():
        MI_dataframe = pd.DataFrame()
        
        MI_dataframe["I(X;L)"] = [value[0] for value in results["MI_X_L"][layer_name]]
        MI_dataframe["I(L;Y)"] = [value[0] for value in results["MI_L_Y"][layer_name]]
        
        MI_dataframe["std I(X;L)"] = [value[1] for value in results["MI_X_L"][layer_name]]
        MI_dataframe["std I(L;Y)"] = [value[1] for value in results["MI_L_Y"][layer_name]]
        
        MI_dataframe["filtered I(X;L)"] = results["filtered_MI_X_L"][layer_name]
        MI_dataframe["filtered I(L;Y)"] = results["filtered_MI_L_Y"][layer_name]
        
        MI_dataframe.to_csv(directory_path / f"{layer_name}.csv", index=False)
        
    # Settings.
    with open(directory_path / "settings.json", 'w') as outfile:
        json.dump(settings, outfile)