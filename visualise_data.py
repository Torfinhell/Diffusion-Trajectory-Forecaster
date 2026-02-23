import warnings

import hydra

from src.data_module import DiffusionTrackerDataModule, visualize_data

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="visualize")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    data_module = DiffusionTrackerDataModule(**config.data_module)
    data_module.setup("fit")
    visualize_data(data_module.train_dataloader(), config.model, config.viz)


if __name__ == "__main__":
    main()
