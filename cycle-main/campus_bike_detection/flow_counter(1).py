from __future__ import annotations

from campus_bike_detection.models import CountLine, Track


class FlowCounter:
    """Direction-aware counting with jitter, debounce, and short-term duplicate suppression."""

    def __init__(
        self,
        line: CountLine,
        direction: str = "both",
        min_cross: float = 0.003,
        debounce_frames: int = 5,
        duplicate_window_frames: int = 30,
        duplicate_distance: float = 0.06,
    ) -> None:
        self.line = line
        self.direction = direction
        self.min_cross = min_cross
        self.debounce_frames = debounce_frames
        self.duplicate_window_frames = duplicate_window_frames
        self.duplicate_distance = duplicate_distance

        self.counted_ids: set[int] = set()
        self.last_side: dict[int, float] = {}
        self.last_count_frame: dict[int, int] = {}
        self.total = 0
        self.forward = 0
        self.backward = 0

        # Keep recent crossing events so the same bike is not counted twice
        # when tracker IDs switch around the counting line.
        self._recent_events: list[tuple[int, float, float, int]] = []

    def _point_side(self, p: tuple[float, float]) -> float:
        x1, y1 = self.line.start
        x2, y2 = self.line.end
        px, py = p
        return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

    def _is_allowed_direction(self, prev: float, cur: float) -> bool:
        if self.direction == "both":
            return True
        if self.direction == "forward":
            return prev < 0 < cur
        if self.direction == "backward":
            return prev > 0 > cur
        return False

    def _direction_sign(self, prev: float, cur: float) -> int:
        if prev < 0 < cur:
            return 1
        if prev > 0 > cur:
            return -1
        return 0

    def _is_duplicate_event(self, frame_idx: int, cx: float, cy: float, direction_sign: int) -> bool:
        fresh_events: list[tuple[int, float, float, int]] = []
        is_duplicate = False

        for event_frame, ex, ey, ed in self._recent_events:
            if frame_idx - event_frame > self.duplicate_window_frames:
                continue
            fresh_events.append((event_frame, ex, ey, ed))

            if ed != direction_sign:
                continue

            dx = ex - cx
            dy = ey - cy
            if (dx * dx + dy * dy) ** 0.5 <= self.duplicate_distance:
                is_duplicate = True

        self._recent_events = fresh_events
        return is_duplicate

    def update(self, tracks: list[Track], frame_idx: int) -> int:
        for track in tracks:
            cx = (track.bbox[0] + track.bbox[2]) * 0.5
            cy = (track.bbox[1] + track.bbox[3]) * 0.5
            cur_side = self._point_side((cx, cy))

            prev_side = self.last_side.get(track.track_id)
            self.last_side[track.track_id] = cur_side
            if prev_side is None:
                continue

            if track.track_id in self.counted_ids:
                continue

            if prev_side * cur_side >= 0:
                continue

            if not self._is_allowed_direction(prev_side, cur_side):
                continue

            if min(abs(prev_side), abs(cur_side)) < self.min_cross:
                continue

            last_count = self.last_count_frame.get(track.track_id, -10**9)
            if frame_idx - last_count <= self.debounce_frames:
                continue

            direction_sign = self._direction_sign(prev_side, cur_side)
            if direction_sign == 0:
                continue
            if self._is_duplicate_event(frame_idx, cx, cy, direction_sign):
                continue

            self.last_count_frame[track.track_id] = frame_idx
            self.counted_ids.add(track.track_id)
            self._recent_events.append((frame_idx, cx, cy, direction_sign))
            self.total += 1
            if direction_sign > 0:
                self.forward += 1
            else:
                self.backward += 1

        return self.total

    def snapshot_counts(self) -> dict[str, int]:
        return {
            self.line.line_id: self.total,
            f"{self.line.line_id}_forward": self.forward,
            f"{self.line.line_id}_backward": self.backward,
        }
