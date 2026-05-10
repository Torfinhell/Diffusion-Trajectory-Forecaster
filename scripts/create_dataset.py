import logging

import hydra

from src.data_module.wb_dataset import Dataset

LOGGER = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../src/configs/feat_extract",
    config_name="small_no_scenes",
)
def main(cfg) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    import time 
    start = time.perf_counter()
    for split in ["train"]:
        split_cfg = cfg.data[split]
        builder = Dataset(flush_every=int(split_cfg.get("flush_every", 512)))
        output_root = builder.create_split(
            split=split,
            artifact_cfg=split_cfg,
            creation_cfg=split_cfg.creation,
        )
        LOGGER.info("Created %s dataset at %s", split, output_root)
    print(f"total split create time = {time.perf_counter() - start:.2f})")

if __name__ == "__main__":
    main()
