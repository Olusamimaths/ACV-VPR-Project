from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .database import ReferenceMap
from .online import LocalizationResult
from .sources import _import_cv2


@dataclass
class LiveDisplay:
    reference_map: ReferenceMap
    show_top_k: bool = True

    def render(
        self,
        frame_bgr: np.ndarray,
        result: Optional[LocalizationResult],
        threshold: float,
        fps: float,
        result_age_ms: float = 0.0,
        process_fps: float | None = None,
        inference_active: bool = True,
        inference_index: int | None = None,
    ) -> np.ndarray:
        cv2 = _import_cv2()
        display = frame_bgr.copy()
        height, width = display.shape[:2]

        cv2.rectangle(display, (0, 0), (width, 118), (0, 0, 0), -1)

        if result is None and not inference_active:
            status_color = (0, 170, 255)
            status = "INFERENCE PAUSED"
        elif result is None:
            status_color = (180, 180, 180)
            status = "WAITING FOR FIRST INFERENCE"
        elif result.recognized:
            status_color = (0, 200, 0)
            status = f"MATCH #{result.best_match_idx}  score={result.best_score:.3f}"
        else:
            status_color = (0, 0, 220)
            status = f"UNKNOWN  best={result.best_score:.3f}"

        cv2.putText(display, status, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)
        cadence = "every frame" if not process_fps or process_fps <= 0 else f"{process_fps:.2f} fps"
        latency_ms = 0.0 if result is None else result.extraction_time_ms
        cv2.putText(
            display,
            f"threshold={threshold:.2f}  latency={latency_ms:.0f}ms  age={result_age_ms:.0f}ms  view_fps={fps:.1f}",
            (12, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (210, 210, 210),
            1,
        )
        cv2.putText(
            display,
            f"inference cadence={cadence}",
            (12, 74),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 180),
            1,
        )
        if inference_index is not None:
            cv2.putText(
                display,
                f"inference #{inference_index}",
                (12, 92),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (180, 180, 180),
                1,
            )
        cv2.putText(
            display,
            "controls: i start/pause  q quit  s save frame  t top-k  +/- threshold",
            (12, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (180, 180, 180),
            1,
        )

        if result is not None:
            self._draw_best_match(display, result, width)
        if self.show_top_k and result is not None:
            self._draw_top_k_panel(display, result)
        return display

    def _draw_best_match(self, display: np.ndarray, result: LocalizationResult, display_width: int) -> None:
        if not result.recognized:
            return

        cv2 = _import_cv2()
        ref_path = self.reference_map.image_paths[result.best_match_idx]
        ref_img = cv2.imread(ref_path)
        if ref_img is None:
            return

        ref_img = cv2.resize(ref_img, (200, 150))
        x0, y0 = display_width - 210, 82
        display[y0 : y0 + 150, x0 : x0 + 200] = ref_img
        cv2.rectangle(display, (x0 - 4, y0 - 4), (x0 + 204, y0 + 154), (0, 200, 0), 2)
        label = Path(ref_path).name
        cv2.putText(display, label[:28], (x0, y0 + 170), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)

    def _draw_top_k_panel(self, display: np.ndarray, result: LocalizationResult) -> None:
        cv2 = _import_cv2()
        height, width = display.shape[:2]
        panel_height = 112
        panel_top = max(118, height - panel_height)

        cv2.rectangle(display, (0, panel_top), (width, height), (24, 24, 24), -1)
        cv2.putText(display, "Top matches", (12, panel_top + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

        thumb_w, thumb_h = 110, 74
        x = 12
        y = panel_top + 26
        for idx, score in zip(result.top_k_indices, result.top_k_scores):
            if x + thumb_w > width - 12:
                break
            ref_img = cv2.imread(self.reference_map.image_paths[idx])
            if ref_img is not None:
                ref_img = cv2.resize(ref_img, (thumb_w, thumb_h))
                display[y : y + thumb_h, x : x + thumb_w] = ref_img
                text_color = (0, 200, 0) if score >= result.best_score else (190, 190, 190)
                cv2.putText(
                    display,
                    f"#{idx} {score:.2f}",
                    (x, y + thumb_h + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    text_color,
                    1,
                )
            x += thumb_w + 10
