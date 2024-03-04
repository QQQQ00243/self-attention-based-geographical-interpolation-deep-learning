import torch


def recursive_device(dct: dict, device):
    for key, value in dct.items():
        if isinstance(value, dict):
            recursive_device(value, device)
        elif isinstance(value, torch.Tensor):
            dct[key] = value.to(device)
            