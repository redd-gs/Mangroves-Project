import argparse
import os
import logging
from mangroves.scripts.load import load_datamodule_from_config, load_litmodule_from_config, load_trainer_from_config
from pytorch_lightning import seed_everything

seed_everything(42, workers=True)
logging.basicConfig(level=logging.INFO)


def check_args(args):
    """
    Check if the required arguments are provided and valid.
    """
    assert args.litmodule_config is not None, "LightningModule configuration file is required."
    assert args.datamodule_config is not None, "LightningDataModule configuration file is required."
    assert args.trainer_config is not None, "Trainer configuration file is required."
    assert os.path.exists(args.litmodule_config), "LightningModule configuration file does not exist."
    assert os.path.exists(all([f for f in args.datamodule_config])), "LightningDataModule configuration file(s) do(es) not exist."
    assert os.path.exists(args.trainer_config), "Trainer configuration file does not exist."


def build_argparser():
    """
    Build the argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Mangrove Project"
    )
    parser.add_argument("--litmodule_config", required=True, help="(Mandatory) Path to the LightningModule configuration file.")
    parser.add_argument("--datamodule_config", required=True, nargs="*", help="(Mandatory) Path to the LightningDataModule configuration file(s).")
    parser.add_argument("--trainer_config", required=True, help="(Mandatory) Path to the Trainer configuration file.")
    parser.add_argument("--train", action="store_true", help="(Optional) Train the model.")
    parser.add_argument("--test", action="store_true", help="(Optional) Test the model.")

    return parser


def main():
    """
    Entry point of the script.
    """
    parser = build_argparser()
    args = parser.parse_args()

    datamodules = load_datamodule_from_config(args.datamodule_config)
    for datamodule in datamodules:
        logging.info(f"Processing {datamodule.datamodule_name}...")
        datamodule.setup()
        logging.info(f"Training dataset size: {len(datamodule.train_dataset)}.")
        logging.info(f"Validation dataset size: {len(datamodule.val_dataset)}.")
        logging.info(f"Test dataset size: {len(datamodule.test_dataset)}.")

    litmodule = load_litmodule_from_config(args.litmodule_config)
    logging.info(f"Number of parameters: {litmodule.net.num_params()}")

    trainer = load_trainer_from_config(args.trainer_config)
    logging.info(f"Number of epochs: {trainer.max_epochs}")
    logging.info(f"Running on {trainer.device_ids} GPUs.")

    if args.train:
        train_dataloaders = [datamodule.train_dataloader() for datamodule in datamodules]
        val_dataloaders = [datamodule.val_dataloader() for datamodule in datamodules]
        trainer.fit(litmodule, train_dataloaders, val_dataloaders)

    if args.test:
        test_dataloaders = [datamodule.test_dataloader() for datamodule in datamodules]
        trainer.test(dataloaders=test_dataloaders)