from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from campus_bike_detection.models import Detection, Track

_F = np.eye(8, dtype=np.float32)
_F[0, 4] = _F[1, 5] = _F[2, 6] = _F[3, 7] = 1.0
_H = np.zeros((4, 8), dtype=np.float32)
_H[0, 0] = _H[1, 1] = _H[2, 2] = _H[3, 3] = 1.0
_Q = np.diag([1e-4, 1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3, 1e-3]).astype(np.float32)
_R = np.diag([5e-4, 5e-4, 5e-4, 5e-4]).astype(np.float32)


def _bbox_to_z(bbox):
    x1, y1, x2, y2 = bbox
    return np.array([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1], dtype=np.float32)

def _z_to_bbox(z):
    cx, cy, w, h = z[:4]
    w, h = max(w, 1e-4), max(h, 1e-4)
    return (cx-w/2, cy-h/2, cx+w/2, cy+h/2)

def _center(b):
    return (b[0]+b[2])*0.5, (b[1]+b[3])*0.5

def _iou(a, b):
    ix1, iy1 = max(a[0],b[0]), max(a[1],b[1])
    ix2, iy2 = min(a[2],b[2]), min(a[3],b[3])
    inter = max(0., ix2-ix1)*max(0., iy2-iy1)
    if inter <= 0: return 0.
    return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)

def _hungarian_match(cost):
    n, m = cost.shape
    if n == 0 or m == 0:
        return []
    if n <= 8 and m <= 8:
        used_r = set()
        used_c = set()
        pairs = []
        for idx in np.argsort(cost, axis=None):
            r, c = divmod(int(idx), m)
            if r not in used_r and c not in used_c:
                pairs.append((r, c))
                used_r.add(r); used_c.add(c)
                if len(pairs) == min(n, m):
                    break
        return pairs
    from scipy.optimize import linear_sum_assignment
    r, c = linear_sum_assignment(cost)
    return list(zip(r.tolist(), c.tolist()))


class _KalmanBox:
    def __init__(self, bbox):
        self.x = np.zeros(8, dtype=np.float32)
        self.x[:4] = _bbox_to_z(bbox)
        self.P = np.eye(8, dtype=np.float32) * 1e-2

    def predict(self):
        self.x = _F @ self.x
        self.P = _F @ self.P @ _F.T + _Q
        return _z_to_bbox(self.x)

    def update(self, bbox):
        z = _bbox_to_z(bbox)
        S = _H @ self.P @ _H.T + _R
        K = self.P @ _H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - _H @ self.x)
        self.P = (np.eye(8, dtype=np.float32) - K @ _H) @ self.P
        return _z_to_bbox(self.x)


@dataclass
class _State:
    kf: _KalmanBox
    bbox: Tuple[float, float, float, float]
    misses: int
    hits: int = 0          # consecutive confirmed hits
    confirmed: bool = False
    traj: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class _DeadRecord:
    tid: int
    bbox: Tuple[float, float, float, float]
    traj: List[Tuple[float, float]]
    died_at_frame: int
    gmc_dx: float = 0.0
    gmc_dy: float = 0.0


class BikeTracker:
    """
    Kalman IoU tracker with:
    - Two-pass matching (IoU + center-distance fallback)
    - Tentative/confirmed states: new tracks must be seen for `confirm_hits`
      consecutive frames before being treated as real  filters out noise detections
    - Ghost frames during occlusion
    - Re-ID with Global Motion Compensation
    """

    def __init__(
        self,
        iou_thresh: float = 0.25,
        max_misses: int = 60,
        max_center_step: float = 0.35,
        max_area_ratio: float = 6.0,   # relaxed: partial occlusion changes bbox size a lot
        confirm_hits: int = 3,         # frames a new track must be seen before confirmed
        reid_frames: int = 90,
        reid_center_thresh: float = 0.25,
    ) -> None:
        self.iou_thresh = iou_thresh
        self.max_misses = max_misses
        self.max_center_step = max_center_step
        self.max_area_ratio = max_area_ratio
        self.confirm_hits = confirm_hits
        self.reid_frames = reid_frames
        self.reid_center_thresh = reid_center_thresh
        self.next_id = 1
        self.states: Dict[int, _State] = {}
        self.seen_ids: Set[int] = set()
        self._frame_idx = 0
        self._dead: List[_DeadRecord] = []

    def update(self, detections: List[Detection]) -> List[Track]:
        self._frame_idx += 1
        cutoff = self._frame_idx - self.reid_frames
        self._dead = [d for d in self._dead if d.died_at_frame >= cutoff]

        prev_centers = {tid: _center(s.bbox) for tid, s in self.states.items()}

        for state in self.states.values():
            state.bbox = state.kf.predict()

        tids = list(self.states.keys())
        dets = list(detections)
        assigned: Dict[int, Detection] = {}
        matched_det_oids: Set[int] = set()
        matched_tids: Set[int] = set()

        # Pass 1  IoU Hungarian
        if tids and dets:
            n, m = len(tids), len(dets)
            cost = np.ones((n, m), dtype=np.float32)
            for i, tid in enumerate(tids):
                for j, det in enumerate(dets):
                    if not self._plausible(self.states[tid].bbox, det.bbox):
                        continue
                    cost[i, j] = 1.0 - _iou(self.states[tid].bbox, det.bbox)
            for i, j in _hungarian_match(cost):
                if cost[i, j] >= 1.0 - self.iou_thresh:
                    continue
                self._match(tids[i], dets[j], assigned, matched_det_oids, matched_tids)

        # Pass 2  center-distance fallback
        for tid in tids:
            if tid in matched_tids:
                continue
            pcx, pcy = _center(self.states[tid].bbox)
            best_j, best_dist = -1, self.max_center_step * 0.6
            for j, det in enumerate(dets):
                if id(det) in matched_det_oids:
                    continue
                ccx, ccy = _center(det.bbox)
                dist = ((pcx-ccx)**2 + (pcy-ccy)**2)**0.5
                if dist < best_dist and self._similar_size(self.states[tid].bbox, det.bbox):
                    best_dist, best_j = dist, j
            if best_j >= 0:
                self._match(tid, dets[best_j], assigned, matched_det_oids, matched_tids)

        # GMC estimation
        gmc_dx, gmc_dy = self._estimate_gmc(prev_centers, matched_tids)
        for rec in self._dead:
            rec.gmc_dx += gmc_dx
            rec.gmc_dy += gmc_dy

        # Unmatched tracks -> ghost or expire
        for tid in tids:
            if tid in matched_tids:
                continue
            state = self.states[tid]
            state.misses += 1
            # Tentative tracks that miss immediately are dropped right away
            if not state.confirmed and state.misses >= 1:
                self._dead.append(_DeadRecord(
                    tid=tid, bbox=state.bbox,
                    traj=list(state.traj), died_at_frame=self._frame_idx,
                ))
                del self.states[tid]
            elif state.misses > self.max_misses:
                self._dead.append(_DeadRecord(
                    tid=tid, bbox=state.bbox,
                    traj=list(state.traj), died_at_frame=self._frame_idx,
                ))
                del self.states[tid]
            else:
                ghost = Detection(bbox=state.bbox, confidence=0.0, class_id=-1)
                cx, cy = _center(state.bbox)
                state.traj.append((cx, cy))
                if len(state.traj) > 60:
                    state.traj = state.traj[-60:]
                assigned[tid] = ghost

        # Unmatched detections -> Re-ID or spawn
        for det in dets:
            if id(det) in matched_det_oids:
                continue
            reused = self._try_reid(det)
            if reused is not None:
                assigned[reused] = det
            else:
                self._spawn(det)

        # Only output confirmed tracks (or tentative tracks that have a live detection)
        result: List[Track] = []
        for tid, det in assigned.items():
            if tid not in self.states:
                continue
            state = self.states[tid]
            # Show tentative tracks only if they have a real detection this frame
            if not state.confirmed and det.confidence == 0.0:
                continue
            result.append(Track(
                track_id=tid,
                bbox=state.bbox,
                confidence=det.confidence,
                trajectory=state.traj,
            ))
        return result

    def _estimate_gmc(self, prev_centers, matched_tids):
        dxs, dys = [], []
        for tid in matched_tids:
            if tid not in prev_centers or tid not in self.states:
                continue
            px, py = prev_centers[tid]
            cx, cy = _center(self.states[tid].bbox)
            dxs.append(cx - px)
            dys.append(cy - py)
        if not dxs:
            return 0.0, 0.0
        return float(np.median(dxs)), float(np.median(dys))

    def _match(self, tid, det, assigned, matched_det_oids, matched_tids):
        state = self.states[tid]
        state.bbox = state.kf.update(det.bbox)
        state.misses = 0
        state.hits += 1
        if state.hits >= self.confirm_hits:
            state.confirmed = True
        cx, cy = _center(state.bbox)
        state.traj.append((cx, cy))
        if len(state.traj) > 60:
            state.traj = state.traj[-60:]
        assigned[tid] = det
        matched_det_oids.add(id(det))
        matched_tids.add(tid)

    def _try_reid(self, det: Detection) -> Optional[int]:
        ccx, ccy = _center(det.bbox)
        best: Optional[_DeadRecord] = None
        best_dist = self.reid_center_thresh
        for rec in self._dead:
            pcx = _center(rec.bbox)[0] + rec.gmc_dx
            pcy = _center(rec.bbox)[1] + rec.gmc_dy
            dist = ((pcx-ccx)**2 + (pcy-ccy)**2)**0.5
            if dist < best_dist and self._similar_size(rec.bbox, det.bbox):
                best_dist, best = dist, rec
        if best is None:
            return None
        self._dead = [d for d in self._dead if d.tid != best.tid]
        cx, cy = _center(det.bbox)
        traj = best.traj + [(cx, cy)]
        if len(traj) > 60:
            traj = traj[-60:]
        # Resurrect as confirmed (it was confirmed before it died)
        state = _State(kf=_KalmanBox(det.bbox), bbox=det.bbox, misses=0,
                       hits=self.confirm_hits, confirmed=True, traj=traj)
        self.states[best.tid] = state
        self.seen_ids.add(best.tid)
        return best.tid

    def _spawn(self, det: Detection) -> int:
        cx, cy = _center(det.bbox)
        # Drop if too close to an existing live track
        for state in self.states.values():
            pcx, pcy = _center(state.bbox)
            dist = ((pcx-cx)**2 + (pcy-cy)**2)**0.5
            if dist < self.max_center_step * 0.4 and self._similar_size(state.bbox, det.bbox):
                return -1
        tid = self.next_id
        self.next_id += 1
        self.states[tid] = _State(kf=_KalmanBox(det.bbox), bbox=det.bbox,
                                  misses=0, hits=1, confirmed=False, traj=[(cx, cy)])
        self.seen_ids.add(tid)
        return tid

    def _plausible(self, prev, cur) -> bool:
        pcx, pcy = _center(prev)
        ccx, ccy = _center(cur)
        if ((pcx-ccx)**2 + (pcy-ccy)**2)**0.5 > self.max_center_step:
            return False
        return self._similar_size(prev, cur)

    def _similar_size(self, a, b) -> bool:
        pa = max((a[2]-a[0])*(a[3]-a[1]), 1e-9)
        ca = max((b[2]-b[0])*(b[3]-b[1]), 1e-9)
        return max(pa/ca, ca/pa) <= self.max_area_ratio

    def total_unique(self) -> int:
        # Only count confirmed tracks to avoid noise IDs inflating the count
        return len(self.seen_ids)
