import argparse
import collections
import torch
import numpy as np
import pandas as pd
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from parse_config import _update_config
from trainer import Trainer
from utils import prepare_device, write_json








def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()
    modification = {'data_loader;args;n_training': data_loader.sampler.__len__()}
    config = _update_config(config, modification)
    modification = {'data_loader;args;n_validation': data_loader.valid_sampler.__len__()}
    config = _update_config(config, modification)
    write_json(config.__getdict__(), f"{config.get_path()}/config.json")


    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    if device == "mps":#######Is this the correct place to put this? can I move the others here?
        torch.mps.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--a', '--architecture'], type=str, target='arch;type'),
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--mn', '--max_nodes'], type=int, target='arch;args;max_nodes'),
        CustomArgs(['--s', '--seed'], type=int, target='seed'),
        CustomArgs(['--i', '--in_csv'], type=str, target='data_loader;args;input_csv_file'),
        CustomArgs(['--o', '--out_csv'], type=str, target='data_loader;args;output_csv_file'),
        CustomArgs(['--n', '--name'], type=str, target='name')
    ]
    
    config = ConfigParser.from_args(args, options)
    if config["data_loader"]["args"]["scale_output"] in ["both", "cons", "consumption","true", True] and ("cons_scale" not in config["data_loader"]["args"] or config["data_loader"]["args"]["cons_scale"] is None):
        output_data = pd.read_csv(config["data_loader"]["args"]["output_csv_file"])

        cons_scale = np.max(output_data["Consumption"])
        modification = {'data_loader;args;cons_scale': cons_scale}
        config = _update_config(config, modification)
        write_json(config.__getdict__(), f"{config.get_path()}/config.json")

    if config["data_loader"]["args"]["scale_output"] in ["both", "i_a","equation", "true", True] and ("i_a_scale" not in config["data_loader"]["args"] or config["data_loader"]["args"]["i_a_scale"] is None):

        i_a_scale = np.max(config["possible_i_a"])
        modification = {'data_loader;args;i_a_scale': i_a_scale}
        config = _update_config(config, modification)
        write_json(config.__getdict__(), f"{config.get_path()}/config.json")




        

    # fix random seeds for reproducibility
    SEED = config["seed"]
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    print(SEED)
    main(config)
