from pathlib import Path


def sanitize_checkpoint_name_component(value, fallback="unknown"):
    text = str(value).strip() if value is not None else ""
    if not text:
        text = fallback
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
    cleaned = cleaned.strip("_")
    return cleaned or fallback


def build_checkpoint_run_directory(root: Path, logger) -> Path:
    version = sanitize_checkpoint_name_component(getattr(logger, "version", None), "local")
    return root / version
