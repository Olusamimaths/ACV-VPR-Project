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

    header_height: int = 146
    top_k_panel_height: int = 156
    best_match_size: tuple[int, int] = (220, 165)
    top_k_thumb_size: tuple[int, int] = (125, 84)

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
        header_height = min(self.header_height, max(118, height // 4))

        cv2.rectangle(display, (0, 0), (width, header_height), (0, 0, 0), -1)

        status, status_color = self._status_text(result, inference_active)
        cv2.putText(display, status, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.78, status_color, 2)

        latency_ms = 0.0 if result is None else result.extraction_time_ms
        cadence = "every frame" if not process_fps or process_fps <= 0 else f"{process_fps:.2f} fps"
        stats_lines = [
            f"threshold={threshold:.2f}   latency={latency_ms:.0f}ms   result_age={result_age_ms:.0f}ms   view_fps={fps:.1f}",
            f"inference cadence={cadence}",
            f"inference #{inference_index}" if inference_index is not None else "inference not started yet",
            "controls: i start/pause  q quit  s save frame  t top-k  +/- threshold",
        ]

        y = 58
        for line in stats_lines:
            cv2.putText(display, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1)
            y += 22

        cv2.putText(
            display,
            "Live query frame",
            (14, header_height + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (235, 235, 235),
            1,
        )

        if result is not None:
            self._draw_best_match(display, result, header_height)
        if self.show_top_k and result is not None:
            self._draw_top_k_panel(display, result, header_height)
        return display

    def render_inference_report(
        self,
        query_frame_bgr: np.ndarray,
        result: LocalizationResult,
        threshold: float,
        fps: float,
        result_age_ms: float = 0.0,
        process_fps: float | None = None,
        inference_index: int | None = None,
    ) -> np.ndarray:
        cv2 = _import_cv2()
        padding = 20
        section_gap = 18
        query_panel = self._resize_to_fit(query_frame_bgr, max_width=980, max_height=620)

        best_label = "Best reference match" if result.recognized else "Best reference candidate"
        best_ref = self._load_reference_image(result.best_match_idx)
        best_ref = self._resize_to_fit(best_ref, max_width=420, max_height=315) if best_ref is not None else None

        right_panel_width = 440
        content_height = max(query_panel.shape[0], 360)
        header_height = 184
        top_k_height = 224
        canvas_width = query_panel.shape[1] + right_panel_width + (padding * 2) + section_gap
        canvas_height = header_height + content_height + top_k_height + (padding * 3)

        canvas = np.full((canvas_height, canvas_width, 3), 20, dtype=np.uint8)

        cv2.rectangle(canvas, (0, 0), (canvas_width, header_height), (8, 8, 8), -1)
        status, status_color = self._status_text(result, inference_active=True)
        cv2.putText(canvas, status, (padding, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)

        cadence = "every frame" if not process_fps or process_fps <= 0 else f"{process_fps:.2f} fps"
        header_lines = [
            f"threshold={threshold:.2f}   latency={result.extraction_time_ms:.0f}ms   result_age={result_age_ms:.0f}ms   view_fps={fps:.1f}",
            f"inference cadence={cadence}",
            f"inference #{inference_index}" if inference_index is not None else "inference index unavailable",
            f"best reference index=#{result.best_match_idx}   best score={result.best_score:.3f}",
        ]
        y = 72
        for line in header_lines:
            cv2.putText(canvas, line, (padding, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (225, 225, 225), 1)
            y += 28

        query_x = padding
        query_y = header_height + padding
        cv2.putText(canvas, "Live query frame", (query_x, query_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (240, 240, 240), 2)
        canvas[query_y : query_y + query_panel.shape[0], query_x : query_x + query_panel.shape[1]] = query_panel
        cv2.rectangle(
            canvas,
            (query_x - 2, query_y - 2),
            (query_x + query_panel.shape[1] + 2, query_y + query_panel.shape[0] + 2),
            (110, 110, 110),
            2,
        )

        right_x = query_x + query_panel.shape[1] + section_gap
        right_y = query_y
        cv2.putText(canvas, best_label, (right_x, right_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (240, 240, 240), 2)
        self._draw_reference_report_panel(canvas, right_x, right_y, right_panel_width, best_ref, result)

        top_panel_y = header_height + content_height + (padding * 2)
        self._draw_top_k_report_panel(canvas, top_panel_y, canvas_width - (padding * 2), result, padding)
        return canvas

    def _status_text(self, result: Optional[LocalizationResult], inference_active: bool) -> tuple[str, tuple[int, int, int]]:
        if result is None and not inference_active:
            return "INFERENCE PAUSED", (0, 170, 255)
        if result is None:
            return "WAITING FOR FIRST INFERENCE", (180, 180, 180)
        if result.recognized:
            return f"MATCH #{result.best_match_idx}   score={result.best_score:.3f}", (0, 200, 0)
        return f"UNKNOWN   best candidate=#{result.best_match_idx}   score={result.best_score:.3f}", (0, 140, 255)

    def _draw_best_match(self, display: np.ndarray, result: LocalizationResult, header_height: int) -> None:
        cv2 = _import_cv2()
        ref_img = self._load_reference_image(result.best_match_idx)
        if ref_img is None:
            return

        thumb_w, thumb_h = self.best_match_size
        ref_img = cv2.resize(ref_img, (thumb_w, thumb_h))
        height, width = display.shape[:2]
        x0 = max(12, width - thumb_w - 18)
        y0 = min(max(header_height + 20, 20), max(20, height - self.top_k_panel_height - thumb_h - 52))

        label = "Best reference match" if result.recognized else "Best reference candidate"
        border_color = (0, 200, 0) if result.recognized else (0, 140, 255)
        cv2.putText(display, label, (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1)
        display[y0 : y0 + thumb_h, x0 : x0 + thumb_w] = ref_img
        cv2.rectangle(display, (x0 - 3, y0 - 3), (x0 + thumb_w + 3, y0 + thumb_h + 3), border_color, 2)

        file_label = self._truncate_text(Path(self.reference_map.image_paths[result.best_match_idx]).name, 34)
        cv2.putText(display, f"Ref #{result.best_match_idx}  score={result.best_score:.3f}", (x0, y0 + thumb_h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, border_color, 1)
        cv2.putText(display, file_label, (x0, y0 + thumb_h + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

    def _draw_top_k_panel(self, display: np.ndarray, result: LocalizationResult, header_height: int) -> None:
        cv2 = _import_cv2()
        height, width = display.shape[:2]
        panel_height = min(self.top_k_panel_height, max(132, height // 4))
        panel_top = max(header_height + 210, height - panel_height)

        cv2.rectangle(display, (0, panel_top), (width, height), (24, 24, 24), -1)
        cv2.putText(display, "Top reference candidates", (12, panel_top + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (220, 220, 220), 1)

        thumb_w, thumb_h = self.top_k_thumb_size
        x = 12
        y = panel_top + 30
        for idx, score in zip(result.top_k_indices, result.top_k_scores):
            if x + thumb_w > width - 12:
                break
            ref_img = self._load_reference_image(idx)
            if ref_img is not None:
                ref_img = cv2.resize(ref_img, (thumb_w, thumb_h))
                display[y : y + thumb_h, x : x + thumb_w] = ref_img
                label_color = (0, 200, 0) if idx == result.best_match_idx else (210, 210, 210)
                cv2.rectangle(display, (x - 2, y - 2), (x + thumb_w + 2, y + thumb_h + 2), label_color, 1)
                cv2.putText(display, f"Ref #{idx}  {score:.2f}", (x, y + thumb_h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, label_color, 1)
                file_label = self._truncate_text(Path(self.reference_map.image_paths[idx]).stem, 18)
                cv2.putText(display, file_label, (x, y + thumb_h + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 190, 190), 1)
            x += thumb_w + 12

    def _draw_reference_report_panel(
        self,
        canvas: np.ndarray,
        panel_x: int,
        panel_y: int,
        panel_width: int,
        ref_img: np.ndarray | None,
        result: LocalizationResult,
    ) -> None:
        cv2 = _import_cv2()
        panel_height = 360
        cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (35, 35, 35), -1)

        border_color = (0, 200, 0) if result.recognized else (0, 140, 255)
        if ref_img is not None:
            img_x = panel_x + 10
            img_y = panel_y + 10
            canvas[img_y : img_y + ref_img.shape[0], img_x : img_x + ref_img.shape[1]] = ref_img
            cv2.rectangle(canvas, (img_x - 2, img_y - 2), (img_x + ref_img.shape[1] + 2, img_y + ref_img.shape[0] + 2), border_color, 2)
            text_y = img_y + ref_img.shape[0] + 28
        else:
            cv2.putText(canvas, "Reference image unavailable", (panel_x + 12, panel_y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1)
            text_y = panel_y + 72

        lines = [
            f"Ref index: #{result.best_match_idx}",
            f"Best score: {result.best_score:.3f}",
            f"Decision: {'MATCH' if result.recognized else 'UNKNOWN'}",
            self._truncate_text(Path(self.reference_map.image_paths[result.best_match_idx]).name, 42),
        ]
        for line in lines:
            cv2.putText(canvas, line, (panel_x + 12, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (225, 225, 225), 1)
            text_y += 28

    def _draw_top_k_report_panel(
        self,
        canvas: np.ndarray,
        panel_top: int,
        panel_width: int,
        result: LocalizationResult,
        padding: int,
    ) -> None:
        cv2 = _import_cv2()
        panel_height = canvas.shape[0] - panel_top - padding
        cv2.rectangle(canvas, (padding, panel_top), (padding + panel_width, panel_top + panel_height), (28, 28, 28), -1)
        cv2.putText(canvas, "Top reference candidates", (padding + 12, panel_top + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (235, 235, 235), 2)

        thumb_w, thumb_h = 155, 104
        x = padding + 12
        y = panel_top + 42
        for idx, score in zip(result.top_k_indices, result.top_k_scores):
            if x + thumb_w > padding + panel_width - 12:
                break
            ref_img = self._load_reference_image(idx)
            if ref_img is not None:
                ref_img = cv2.resize(ref_img, (thumb_w, thumb_h))
                canvas[y : y + thumb_h, x : x + thumb_w] = ref_img
            border_color = (0, 200, 0) if idx == result.best_match_idx else (140, 140, 140)
            cv2.rectangle(canvas, (x - 2, y - 2), (x + thumb_w + 2, y + thumb_h + 2), border_color, 2)
            cv2.putText(canvas, f"Ref #{idx}   score={score:.3f}", (x, y + thumb_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46, border_color, 1)
            file_label = self._truncate_text(Path(self.reference_map.image_paths[idx]).name, 22)
            cv2.putText(canvas, file_label, (x, y + thumb_h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (215, 215, 215), 1)
            x += thumb_w + 18

    def _load_reference_image(self, index: int) -> np.ndarray | None:
        cv2 = _import_cv2()
        ref_path = self.reference_map.image_paths[index]
        return cv2.imread(ref_path)

    def _resize_to_fit(self, image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
        cv2 = _import_cv2()
        height, width = image.shape[:2]
        scale = min(max_width / max(width, 1), max_height / max(height, 1))
        if scale >= 1.0:
            return image.copy()
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))
        return cv2.resize(image, (new_width, new_height))

    def _truncate_text(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3] + "..."
