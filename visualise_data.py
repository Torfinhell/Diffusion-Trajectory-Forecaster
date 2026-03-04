import warnings

import hydra

from src.data_module import DiffusionTrackerDataModule, visualize_data
from src.utils import process_hparams

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="visualize")
def main(cfg):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        cfg (DictConfig): hydra experiment config.
    """
    hparams = process_hparams(cfg, print_hparams=True)
    data_module = DiffusionTrackerDataModule(hparams.data_module)
    data_module.setup("fit")
    visualize_data(data_module.train_dataloader(), hparams.model, hparams.viz)


if __name__ == "__main__":
    main()
