import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            #target = target.view(-1)
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())

            if any(met.__name__ in ['n_wrong_i_a', 'n_exceeding_i_a_k'] for met in self.metric_ftns):
                possible_targets = self.config.__getitem__('possible_i_a')
                if possible_targets==None:
                    possible_targets = torch.unique(target[:,0]).to(self.device)
                else:
                    possible_targets = torch.tensor(possible_targets).to(self.device)
                if 'cons_scale' in self.config.__getitem__('data_loader')['args']:
                    cons_scale = self.config.__getitem__('data_loader')['args']['cons_scale']
                else:
                    cons_scale = 1
                if 'i_a_scale' in self.config.__getitem__('data_loader')['args']:
                    i_a_scale = self.config.__getitem__('data_loader')['args']['i_a_scale']
                else:
                    i_a_scale = 1
                if 'input_scale' in self.config.__getitem__('data_loader')['args']:
                    input_scale = self.config["data_loader"]["args"].get("input_scale")
                else:
                    input_scale = {}

            for met in self.metric_ftns:

                if met.__name__ == 'n_wrong_i_a':
                    self.train_metrics.update(met.__name__, met(output, target, possible_targets, i_a_scale), aggregation="total")
                elif met.__name__ == 'n_exceeding_i_a_k':
                    self.train_metrics.update(met.__name__, met(data, output, possible_targets, cons_scale,i_a_scale,input_scale), aggregation="total")
                elif met.__name__ == 'n_exceeding_k':
                    self.train_metrics.update(met.__name__, met(data, output, cons_scale,input_scale), aggregation="total")
                elif "falsepositive" in met.__name__ or "falsenegative" in met.__name__:
                    self.train_metrics.update(met.__name__, met(output, target), aggregation="total")
                elif "consumption" in met.__name__:
                    self.train_metrics.update(met.__name__, met(output, target, cons_scale))
                elif "i_a" in met.__name__:
                    self.train_metrics.update(met.__name__, met(output, target, i_a_scale))
                else:
                    self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())

                if any(met.__name__ in ['n_wrong_i_a', 'n_exceeding_i_a_k'] for met in self.metric_ftns):
                    possible_targets = self.config.__getitem__('possible_i_a')
                    if possible_targets==None:
                        possible_targets = torch.unique(target[:,0]).to(self.device)
                    else:
                        possible_targets = torch.tensor(possible_targets).to(self.device)
                if 'cons_scale' in self.config.__getitem__('data_loader')['args']:
                    cons_scale = self.config.__getitem__('data_loader')['args']['cons_scale']
                else:
                    cons_scale = 1
                if 'i_a_scale' in self.config.__getitem__('data_loader')['args']:
                    i_a_scale = self.config.__getitem__('data_loader')['args']['i_a_scale']
                else:
                    i_a_scale = 1
                if 'input_scale' in self.config.__getitem__('data_loader')['args']:
                    input_scale = self.config["data_loader"]["args"].get("input_scale")
                else:
                    input_scale = {}

                for met in self.metric_ftns:
                    if met.__name__ == 'n_wrong_i_a':
                        self.valid_metrics.update(met.__name__, met(output, target, possible_targets, i_a_scale), aggregation="total")
                    elif met.__name__ == 'n_exceeding_i_a_k':
                        self.valid_metrics.update(met.__name__, met(data, output, possible_targets, cons_scale, i_a_scale,input_scale), aggregation="total")
                    elif met.__name__ == 'n_exceeding_k':
                        self.valid_metrics.update(met.__name__, met(data, output, cons_scale, input_scale), aggregation="total")
                    elif "falsepositive" in met.__name__ or "falsenegative" in met.__name__:
                        self.valid_metrics.update(met.__name__, met(output, target),aggregation="total")
                    elif "consumption" in met.__name__:
                        self.valid_metrics.update(met.__name__, met(output, target, cons_scale))
                    elif "i_a" in met.__name__:
                        self.valid_metrics.update(met.__name__, met(output, target, i_a_scale))
                    else: 
                        self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
