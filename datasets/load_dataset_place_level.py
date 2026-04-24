#   =====================================================================
#   Place-level campus dataset loader
#   Built as a companion to the original CampusDataset loader so the
#   previous 1-to-1 evaluation path remains unchanged.
#   =====================================================================
#
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image


@dataclass
class CampusPlaceLevelSummary:
    database_images: int
    query_images: int
    matched_queries: int
    unmatched_queries: int
    rematched_from_npm: int
    unique_places: int


class CampusPlaceLevelDataset:
    """
    Loader for the manually rematched campus dataset.

    Unlike the original CampusDataset, this loader evaluates at the
    place level. A query can have multiple valid positive reference
    images via the `valid_day_images` column in night_rematches.csv.
    """

    def __init__(self, destination: str = "custom_dataset_place_level/"):
        self.destination = Path(destination)
        self.day_dir = self.destination / "day_images"
        self.night_dir = self.destination / "night_images"
        self.day_index_csv = self.destination / "day_place_index.csv"
        self.night_rematches_csv = self.destination / "night_rematches.csv"
        self.summary: CampusPlaceLevelSummary | None = None

    def load(self) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
        print("===== Load custom campus dataset (place-level): day_images --> night_images")

        self._validate_layout()

        fns_db = sorted(glob(str(self.day_dir / "*.jpg")))
        fns_q = sorted(glob(str(self.night_dir / "*.jpg")))
        imgs_db = [np.array(Image.open(fn)) for fn in fns_db]
        imgs_q = [np.array(Image.open(fn)) for fn in fns_q]

        day_lookup = {os.path.basename(fn): idx for idx, fn in enumerate(fns_db)}
        query_lookup = [os.path.basename(fn) for fn in fns_q]

        day_rows = self._load_day_rows()
        night_rows = self._load_night_rows()

        GThard = self._create_ground_truth(day_lookup, query_lookup, night_rows)

        # The place-level positives are already expanded explicitly, so the
        # soft ground truth is kept identical to the hard ground truth.
        GTsoft = GThard.copy()

        matched_queries = int(np.sum(GThard.any(axis=0)))
        unmatched_queries = int(len(imgs_q) - matched_queries)
        rematched_from_npm = sum(
            1 for row in night_rows.values()
            if row["original_label"].strip() == "marked_npm"
            and row["rematch_decision"].strip() == "rematched_from_npm"
        )
        unique_places = len({row["place_id"] for row in day_rows.values() if row["place_id"]})

        self.summary = CampusPlaceLevelSummary(
            database_images=len(imgs_db),
            query_images=len(imgs_q),
            matched_queries=matched_queries,
            unmatched_queries=unmatched_queries,
            rematched_from_npm=rematched_from_npm,
            unique_places=unique_places,
        )

        print(f"  Loaded {len(imgs_db)} day images (database)")
        print(f"  Loaded {len(imgs_q)} night images (queries)")
        print(f"  Place-level matched queries: {matched_queries}")
        print(f"  Remaining no-match queries: {unmatched_queries}")
        print(f"  Queries rematched from original -npm labels: {rematched_from_npm}")
        print(f"  Unique place IDs: {unique_places}")

        return imgs_db, imgs_q, GThard, GTsoft

    def _validate_layout(self) -> None:
        required_paths = [
            self.day_dir,
            self.night_dir,
            self.day_index_csv,
            self.night_rematches_csv,
        ]
        for path in required_paths:
            if not path.exists():
                raise FileNotFoundError(f"Required place-level dataset path not found: {path}")

    def _load_day_rows(self) -> dict[str, dict[str, str]]:
        with self.day_index_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        return {row["day_image"].strip(): row for row in rows}

    def _load_night_rows(self) -> dict[str, dict[str, str]]:
        with self.night_rematches_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        return {row["night_image"].strip(): row for row in rows}

    def _create_ground_truth(
        self,
        day_lookup: dict[str, int],
        query_lookup: list[str],
        night_rows: dict[str, dict[str, str]],
    ) -> np.ndarray:
        gt = np.zeros((len(day_lookup), len(query_lookup)), dtype=bool)

        for q_idx, night_name in enumerate(query_lookup):
            row = night_rows.get(night_name)
            if row is None:
                continue

            valid_names = [
                part.strip()
                for part in row.get("valid_day_images", "").split("|")
                if part.strip()
            ]
            for valid_name in valid_names:
                db_idx = day_lookup.get(valid_name)
                if db_idx is not None:
                    gt[db_idx, q_idx] = True

        return gt
