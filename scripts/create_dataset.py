import logging

import hydra

from src.data_module.data_module import instantiate_dataset_split

LOGGER = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path="../src/configs", config_name="create_dataset"
)
def main(cfg) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    for split in ("train", "val", "test"):
        ds = instantiate_dataset_split(cfg.dataset.data, split)
        LOGGER.info("Built %s at %s", split, ds.local)


if __name__ == "__main__":
    main()
