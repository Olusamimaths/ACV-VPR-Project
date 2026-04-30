from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import shutil

DEFAULT_OUTPUT_ROOT = "output_images"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "run"


def _next_run_index(runs_dir: Path) -> int:
    max_index = 0
    for child in runs_dir.iterdir():
        if not child.is_dir():
            continue
        match = re.match(r"^(\d+)_", child.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return max_index + 1


@dataclass
class ExperimentRunOutput:
    base_dir: Path
    run_dir: Path
    legacy_dir: Path

    @classmethod
    def create(
        cls,
        base_dir: str = DEFAULT_OUTPUT_ROOT,
        run_slug: str = "run",
        *,
        category: str | None = None,
    ) -> "ExperimentRunOutput":
        base_path = Path(base_dir)
        base_path.mkdir(parents=True, exist_ok=True)

        runs_dir = base_path / "runs"
        category_slug = _slugify(category) if category else "misc"
        if category:
            runs_dir = runs_dir / category_slug
        runs_dir.mkdir(parents=True, exist_ok=True)

        run_index = _next_run_index(runs_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = runs_dir / f"{run_index:04d}_{_slugify(run_slug)}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        legacy_dir = base_path / "legacy" / category_slug
        legacy_dir.mkdir(parents=True, exist_ok=True)

        return cls(base_dir=base_path, run_dir=run_dir, legacy_dir=legacy_dir)

    def run_path(self, filename: str) -> str:
        return str(self.run_dir / filename)

    def legacy_path(self, filename: str) -> str:
        self.legacy_dir.mkdir(parents=True, exist_ok=True)
        return str(self.legacy_dir / filename)

    def savefig(
        self,
        fig,
        filename: str,
        *,
        legacy_filename: str | None = None,
        dpi: int = 150,
        bbox_inches: str | None = None,
    ) -> str:
        run_path = self.run_dir / filename
        fig.savefig(run_path, dpi=dpi, bbox_inches=bbox_inches)

        legacy_name = legacy_filename or filename
        self.legacy_dir.mkdir(parents=True, exist_ok=True)
        legacy_path = self.legacy_dir / legacy_name
        fig.savefig(legacy_path, dpi=dpi, bbox_inches=bbox_inches)
        return str(run_path)

    def copy_to_legacy(self, source_path: str, legacy_filename: str | None = None) -> str:
        source = Path(source_path)
        legacy_name = legacy_filename or source.name
        self.legacy_dir.mkdir(parents=True, exist_ok=True)
        legacy_path = self.legacy_dir / legacy_name
        shutil.copy2(source, legacy_path)
        return str(legacy_path)

    def write_text(self, filename: str, text: str, *, legacy_filename: str | None = None) -> str:
        run_path = self.run_dir / filename
        run_path.write_text(text, encoding="utf-8")

        legacy_name = legacy_filename or filename
        self.legacy_dir.mkdir(parents=True, exist_ok=True)
        legacy_path = self.legacy_dir / legacy_name
        legacy_path.write_text(text, encoding="utf-8")
        return str(run_path)
