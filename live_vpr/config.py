from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "live_vpr.yaml"


def _import_yaml():
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyYAML is required for YAML configuration support. Install it with `pip install PyYAML`."
        ) from exc
    return yaml


def _flatten_config_tree(data: dict[str, Any], flat: dict[str, Any]) -> None:
    for key, value in data.items():
        if isinstance(value, dict):
            _flatten_config_tree(value, flat)
            continue
        if key in flat:
            raise ValueError(f"Duplicate config key found while flattening YAML: {key}")
        flat[key] = value


def _parser_destinations(parser: argparse.ArgumentParser) -> set[str]:
    return {
        action.dest
        for action in parser._actions
        if action.dest not in {"help"}
    }


def load_config_defaults(config_path: str | Path, valid_keys: set[str]) -> tuple[dict[str, Any], str]:
    yaml = _import_yaml()
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping at the top level: {path}")

    flat: dict[str, Any] = {}
    _flatten_config_tree(data, flat)

    unknown = sorted(set(flat) - valid_keys)
    if unknown:
        raise ValueError(
            f"Unknown config keys in {path}: {', '.join(unknown)}"
        )

    return flat, str(path)


def parse_args_with_config(parser: argparse.ArgumentParser, argv: list[str] | None = None):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str)
    pre_args, _ = pre_parser.parse_known_args(argv)

    config_path = pre_args.config
    if not config_path and DEFAULT_CONFIG_PATH.exists():
        config_path = str(DEFAULT_CONFIG_PATH)

    if config_path:
        defaults, resolved_path = load_config_defaults(config_path, _parser_destinations(parser))
        parser.set_defaults(**defaults)
        parser.set_defaults(config=resolved_path)

    args = parser.parse_args(argv)
    if getattr(args, "config", None):
        args.config = str(Path(args.config).expanduser().resolve())
    return args
