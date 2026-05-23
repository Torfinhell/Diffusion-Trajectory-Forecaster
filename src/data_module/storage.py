import json
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import boto3
import botocore
import numpy as np
import webdataset as wds

FORMAT: str = "diffusion_tracker_webdataset_v2"
STREAM_CMD: str = "aws s3 cp {url} -"


class S3Storage:
    """Manages interactions with dataset files stored in an Amazon S3 bucket."""

    def __init__(self, prefix: str) -> None:
        """Initializes the S3 client and parses the target S3 URI into bucket and prefix components."""
        p = urlparse(str(prefix).rstrip("/"))
        assert p.scheme == "s3" and p.netloc, prefix
        self.bucket: str = p.netloc
        self.prefix: str = p.path.lstrip("/").rstrip("/")
        self._client: Any = boto3.client("s3")

    @classmethod
    def for_split(cls, dataset_root: str, split: str) -> "S3Storage":
        """Class method to instantiate the storage object using a structured dataset root and a split name string."""
        return cls(f"{str(dataset_root).rstrip('/')}/{split}.wds")

    def key(self, name: str) -> str:
        """Constructs the full S3 object key by appending the filename to the initialized prefix path."""
        return f"{self.prefix}/{name}" if self.prefix else name

    def url(self, name: str) -> str:
        """Generates the complete s3:// URL for a specific file name inside the storage context."""
        return f"s3://{self.bucket}/{self.key(name)}"

    def exists(self) -> bool:
        """Checks if the metadata index file exists in the designated S3 location."""
        try:
            self._client.head_object(Bucket=self.bucket, Key=self.key("index.json"))
            return True
        except botocore.exceptions.ClientError:
            return False

    def read_index(self) -> Dict[str, Any]:
        """Downloads, decodes, and parses the index.json file from S3 into a dictionary while verifying the file format."""
        body = self._client.get_object(Bucket=self.bucket, Key=self.key("index.json"))[
            "Body"
        ]
        meta = json.loads(body.read().decode())
        assert meta.get("format") == FORMAT, meta
        return meta

    def write_index(self, text: str) -> None:
        """Uploads a string representation of the JSON index metadata to the designated S3 location."""
        self._client.put_object(
            Bucket=self.bucket, Key=self.key("index.json"), Body=text.encode()
        )

    def download(self, name: str, dest: Path) -> None:
        """Downloads a targeted file from S3 to a local file system path and creates missing parent directories."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._client.download_file(self.bucket, self.key(name), str(dest))

    def upload(self, name: str, src: Path) -> None:
        """Uploads a specified local file to the configured S3 bucket location."""
        self._client.upload_file(str(src), self.bucket, self.key(name))

    def sync_to(self, local: Path) -> Dict[str, Any]:
        """Downloads the entire remote dataset split, including the index and all corresponding shard archives, to a local directory."""
        local.mkdir(parents=True, exist_ok=True)
        if not (local / "index.json").is_file():
            self.download("index.json", local / "index.json")
        meta = json.loads((local / "index.json").read_text(encoding="utf-8"))
        assert meta.get("format") == FORMAT, meta
        pattern = str(meta.get("shard_pattern", "shard-%06d.tar"))
        for i in range(int(meta["num_shards"])):
            name = pattern % i
            dest = local / name
            if not dest.is_file():
                self.download(name, dest)
        return meta

    def stream_sources(self, meta: Dict[str, Any]) -> List[str]:
        """Generates a list of pipe-compatible commands used by WebDataset to stream tar shards directly from S3 using the AWS CLI."""
        pattern = str(meta.get("shard_pattern", "shard-%06d.tar"))
        n = int(meta["num_shards"])
        assert n > 0, self.prefix
        return [
            "pipe:" + STREAM_CMD.format(url=self.url(pattern % i)) for i in range(n)
        ]


def read_local_index(local: Path) -> Dict[str, Any]:
    """Reads and validates the format of an index.json file from a local directory path."""
    meta = json.loads((local / "index.json").read_text(encoding="utf-8"))
    assert meta.get("format") == FORMAT, meta
    return meta


def write_webdataset(
    local: Path, samples: Any, flush_every: int, remote: Optional[S3Storage]
) -> Path:
    """Serializes an iterable collection of samples into WebDataset tar shards containing NumPy arrays and pickled data objects. Handles automatic chunking and conditional uploading to an S3 bucket destination."""
    if local.exists():
        return local
    local.mkdir(parents=True, exist_ok=True)
    total, num_shards = 0, 0

    def post(shard_path: str) -> None:
        nonlocal num_shards
        num_shards += 1
        if remote:
            p = Path(shard_path)
            remote.upload(p.name, p)
            p.unlink(missing_ok=True)

    with wds.ShardWriter(
        str(local / "shard-%06d.tar"), maxcount=flush_every, post=post, verbose=0
    ) as sink:
        for i, sample in enumerate(samples):
            row = {"__key__": f"{i:09d}"}
            for k, v in sample.items():
                if k == "scenario" and v is not None:
                    row["scenario.pkl"] = pickle.dumps(v)
                elif k != "scenario":
                    row[f"{k}.npy"] = np.asarray(v)
            sink.write(row)
            total += 1

    meta = {
        "format": FORMAT,
        "num_samples": total,
        "num_shards": num_shards,
        "shard_glob": "shard-*.tar",
        "shard_pattern": "shard-%06d.tar",
    }
    assert total > 0
    text = json.dumps(meta, indent=2, sort_keys=True)
    (local / "index.json").write_text(text, encoding="utf-8")
    if remote:
        remote.write_index(text)
        shutil.rmtree(local, ignore_errors=True)
    return local


def decode_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Processes and reconstructs a raw WebDataset dictionary sample back into standard Python objects and NumPy arrays."""
    skip = {"__key__", "__url__", "__local_path__"}
    out = {}
    for key, value in sample.items():
        if key in skip:
            continue
        if key == "scenario.pkl":
            out["scenario"] = value
        elif key.endswith(".npy"):
            out[key[:-4]] = value
        else:
            raise ValueError(f"unsupported field {key}")
    out.setdefault("scenario", None)
    return out
