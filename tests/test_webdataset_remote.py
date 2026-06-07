from pathlib import Path

from omegaconf import OmegaConf

from src.data_module.wb_dataset import Dataset, _s3_join


def test_s3_path_uses_remote_mode():
    split_cfg = OmegaConf.create(
        {
            "processed_path": "s3://bucket/path/train.wds",
            "local_cache_path": "/tmp/train-cache",
        }
    )

    metadata = {
        "num_shards": 2,
        "shard_pattern": "shard-%06d.tar",
    }

    assert Dataset._build_remote_shard_sources(split_cfg.processed_path, metadata) == [
        "pipe:aws s3 cp s3://bucket/path/train.wds/shard-000000.tar -",
        "pipe:aws s3 cp s3://bucket/path/train.wds/shard-000001.tar -",
    ]


def test_local_output_root_uses_local_cache_for_s3_artifact():
    split_cfg = OmegaConf.create(
        {
            "processed_path": "s3://bucket/path/train.wds",
            "local_cache_path": "data/cache/train.wds",
        }
    )

    output_root = Dataset._resolve_local_output_root(split_cfg)

    assert output_root == Path.cwd() / "data/cache/train.wds"


def test_s3_join_builds_expected_url():
    assert _s3_join("s3://bucket/datasets/train.wds", "shard-000000.tar") == (
        "s3://bucket/datasets/train.wds/shard-000000.tar"
    )
