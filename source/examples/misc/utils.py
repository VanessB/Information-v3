import json
import os
import torch

import matplotlib.pyplot as plt
import pandas as pd

from collections import defaultdict
from datetime import datetime
from pathlib import Path


def show_images(images: list(), labels: list()=None, n_cols: int=3):
    """
    Вывести изображения из списка.
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


    
def split_lists(list_of_tuples: list()) -> tuple():
    """
    Список кортежей в кортеж списков.
    """
    
    assert len(list_of_tuples) > 0
    
    lists = [[] for index in range(len(list_of_tuples[0]))]
    for element in list_of_tuples:
        for index, subelement in enumerate(element):
            lists[index].append(subelement)
            
    return tuple(lists)



def get_outputs(model, dataloader, device) -> list():
    """
    Получение наборов данных по слоям.
    """
    
    # Выход из режима обучения.
    was_in_training = model.training
    model.eval()
    
    outputs = []
    with torch.no_grad():
        for index, batch in enumerate(dataloader):
            x, y = batch
            
            outputs.append(model(x.to(device)).detach().cpu())
            
    outputs = torch.cat(outputs)
    
    # Возвращение модели к исходному режиму.
    model.train(was_in_training)
    
    return outputs



def get_layers(model, dataloader, device) -> list():
    """
    Получение наборов данных по слоям.
    """
    
    # Выход из режима обучения.
    was_in_training = model.training
    model.eval()
    
    outputs = defaultdict(list)
    with torch.no_grad():
        for index, batch in enumerate(dataloader):
            x, y = batch
            
            layers = model(x.to(device), all_layers=True)
            for layer_name, layer in layers.items():
                outputs[layer_name].append(layer.detach().cpu())
            
    # "Транспонирование".
    outputs = {layer_name: torch.cat(batches) for layer_name, batches in outputs.items()}
    
    # Возвращение модели к исходному режиму.
    model.train(was_in_training)
    
    return outputs


def save_results(results: dict, settings: dict, path: Path):
    """
    Сохранения результатов (информационная плоскость и пр.)
    """
    
    directory_path = path / (datetime.now().strftime("%d-%b-%Y_%H:%M:%S") + "/")
    os.makedirs(directory_path, exist_ok=True)
    
    # Метрики.
    metrics = pd.DataFrame()
    for metric_name, metric_values in results['metrics'].items():
        metrics[metric_name] = metric_values
    metrics.to_csv(directory_path / "metrics.csv", index=False)
        
    # Взаимная информация.
    for layer_name in results["MI_X_L"].keys():
        MI_dataframe = pd.DataFrame()
        
        MI_dataframe["I(X;L)"] = [value[0] for value in results["MI_X_L"][layer_name]]
        MI_dataframe["I(L;Y)"] = [value[0] for value in results["MI_L_Y"][layer_name]]
        
        MI_dataframe["std I(X;L)"] = [value[1] for value in results["MI_X_L"][layer_name]]
        MI_dataframe["std I(L;Y)"] = [value[1] for value in results["MI_L_Y"][layer_name]]
        
        MI_dataframe["filtered I(X;L)"] = results["filtered_MI_X_L"][layer_name]
        MI_dataframe["filtered I(L;Y)"] = results["filtered_MI_L_Y"][layer_name]
        
        MI_dataframe.to_csv(directory_path / f"{layer_name}.csv", index=False)
        
    # Настройки.
    with open(directory_path / "settings.json", 'w') as outfile:
        json.dump(settings, outfile)