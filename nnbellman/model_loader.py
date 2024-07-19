import torch
import model.model as module_arch
from utils.util import prepare_device


def load_model(model_path):
    '''Load a model from a particular .pth file'''
    
    device,_ = prepare_device(1)

    modelinfo = torch.load(model_path, map_location=device)

    config = modelinfo['config']

    model = getattr(module_arch,config['arch']['type'])(**config['arch']['args'])

    model.load_state_dict(modelinfo['state_dict'])

    return model

