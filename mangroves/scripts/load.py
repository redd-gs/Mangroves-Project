import torch.optim as Optimizer
import torch.optim.lr_scheduler as LR_Scheduler
import pytorch_lightning.callbacks as Callbacks
import pytorch_lightning.loggers as Loggers
import torchvision.transforms as T
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
import mangroves.datasets as datasets
import mangroves.models as models
import mangroves.modules as modules
import mangroves.transforms as mT
from mangroves.scripts.data import MangroveDataset, MangroveDataModule
from ruamel.yaml import YAML
from typing import List
import os


def load_datamodule_from_config(path_config: List[str]) -> LightningDataModule:
    with open(path_config, "r") as config:
        yaml = YAML(typ="safe")
        config = yaml.load(config)

    config['path'] = os.path.expanduser(config['path'])  # Convert ~ to /home/user

    return MangroveDataModule(**config)


def load_litmodule_from_config(path_config: str) -> LightningModule:
    with open(path_config, "r") as config:
        yaml = YAML(typ="safe")
        config = yaml.load(config)

    # Load Model
    model_options = config.get('model_options')
    assert model_options is not None, "model_options must be provided in the configuration file."

    model_class = getattr(models, model_options.get('model_class'))
    assert model_class is not None, f"Model class {model_options['model_class']} not found."
        
    net = model_class(**model_options['parameters'])

    # Load LitModule
    litmodule_options = config.get('litmodule_options')
    assert litmodule_options is not None, "litmodule_options must be provided in the configuration file."

    litmodule_class = getattr(modules, litmodule_options.get("litmodule_class"))
    assert litmodule_class is not None, f"LitModule class {litmodule_options['litmodule_class']} not found."

    if litmodule_options.get('optimizers') is not None:  #TODO: Allow multiple optimizers
        optimizer = getattr(Optimizer, litmodule_options['optimizers']['name'])(net.parameters(), **litmodule_options['optimizers']['parameters'])
    else:
        optimizer = None

    if litmodule_options.get('lr_schedulers') is not None:
        lr_scheduler = getattr(LR_Scheduler, litmodule_options['lr_schedulers']['name'])(optimizer, **litmodule_options['lr_schedulers']['parameters'])
    else:
        lr_scheduler = None

    litmodule_parameters = {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'parameters': litmodule_options.get('parameters', {})}
    litmodule = litmodule_class(net, **litmodule_parameters)

    if config.get('checkpoint_path', None) is not None:
        litmodule_class.load_from_checkpoint(config['checkpoint_path'], net=litmodule.net)    

    return litmodule


def load_trainer_from_config(path_config: str) -> Trainer:
    with open(path_config, "r") as config:
        yaml = YAML(typ="safe")
        config = yaml.load(config)

    trainer_parameters = config['hyperparameters']

    callbacks = []
    for c, p in zip(config['callbacks']['name'], config['callbacks']['parameters']):
        callbacks.append(getattr(Callbacks, c)(**p))
    trainer_parameters['callbacks'] = callbacks

    trainer_parameters['logger'] = getattr(Loggers, config['logger']['name'])(**config['logger']['parameters'])

    return Trainer(**trainer_parameters)
