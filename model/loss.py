import torch


def nll_loss(output, target):

    return torch.nn.functional.nll_loss(output, target)


def mse_loss(output,target):

    return torch.nn.functional.mse_loss(output, target)


def cross_entropy_loss(output,target):
    loss=torch.nn.CrossEntropyLoss()
    return loss(output,target)
