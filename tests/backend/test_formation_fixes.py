"""Regression tests for the tracking / sizing / formation-display fixes."""
from __future__ import annotations

import math

import numpy as np

from app.pipeline.cotracker_pipeline import add_support_points, tracked_points_to_frames
from app.pipeline.foot_assist import compute_foot_anchors, match_points_to_boxes
from app.pipeline.homography import (
    CANVAS_ASPECT,
    compute_rescue_affine,
    compute_stage_homography,
    estimate_floor_aspect,
    project_to_stage,
)
from app.pipeline.processor import (
    _segment_formations,
    _snap_formations_to_templates,
    smooth_trajectories,
)
from app.pipeline.templates import fit_template, generate_templates_for_count
from app.pipeline.types import Detection
from app.schemas.jobs import (
    DancerClick,
    DancerPosition,
    Formation,
    FormationDancerPosition,
    FramePositions,
    VideoMetadata,
)
from app.services.cotracker import TrackedPoint


def _meta(frame_count: int = 3, width: int = 1920, height: int = 1080) -> VideoMetadata:
    return VideoMetadata(
        filename="t.mp4", fps=24.0, frame_count=frame_count,
        width=width, height=height, duration_sec=frame_count / 24.0,
    )


def _box(x: int, y: int, w: int, h: int) -> Detection:
    return Detection(bbox=(x, y, w, h), anchor_px=(x + w // 2, y + h), confidence=0.9)


def _formation(points: list[tuple[float, float]]) -> Formation:
    return Formation(
        index=0, start_frame=0, end_frame=10, start_time_sec=0.0, end_time_sec=1.0,
        duration_sec=1.0,
        dancers=[FormationDancerPosition(id=i + 1, x=x, y=y) for i, (x, y) in enumerate(points)],
    )


# ---------------------------------------------------------------- templates


def test_v_template_is_symmetric_for_even_counts():
    v = next(t for t in generate_templates_for_count(6) if t.name == "V")
    xs = sorted(round(float(x), 6) for x in v.points[:, 0])
    assert xs == sorted(-x for x in xs)


def test_tilted_line_does_not_snap_to_line():
    # A line rotated ~35 degrees used to fit "Line" perfectly (rotation was
    # allowed) and was then drawn tilted under the Line label.
    pts = [(0.3 + 0.08 * i, 0.3 + 0.056 * i) for i in range(6)]
    f = _formation(pts)
    _snap_formations_to_templates([f], threshold=0.08)
    assert f.shape_name != "Line"


def test_v_and_inverted_v_are_distinguished():
    v = [(0.5, 0.2), (0.4, 0.35), (0.6, 0.35), (0.3, 0.5), (0.7, 0.5)]
    inv = [(x, 1.0 - y) for x, y in v]
    fv, finv = _formation(v), _formation(inv)
    _snap_formations_to_templates([fv, finv], threshold=0.08)
    assert fv.shape_name == "V"
    assert finv.shape_name == "Inverted V"


def test_flat_row_snaps_to_line_not_flattened_v():
    pts = [(0.2 + 0.12 * i, 0.5 + (0.004 if i in (0, 5) else -0.002)) for i in range(6)]
    f = _formation(pts)
    _snap_formations_to_templates([f], threshold=0.08)
    assert f.shape_name == "Line"


def test_tight_cluster_does_not_snap():
    rng = np.random.default_rng(0)
    pts = [(0.5 + float(a), 0.5 + float(b)) for a, b in rng.normal(0, 0.01, (6, 2))]
    f = _formation(pts)
    _snap_formations_to_templates([f], threshold=0.08)
    assert f.shape_name is None


def test_fit_template_preserves_wide_shallow_v():
    v = next(t for t in generate_templates_for_count(5) if t.name == "V")
    observed = v.points * np.array([0.3, 0.05]) + 0.5  # very wide, shallow
    snapped, rmse = fit_template(v, observed)
    assert rmse < 1e-6


# -------------------------------------------------------------- foot assist


def test_foot_gate_picks_box_whose_bottom_matches_predicted_foot():
    # Back dancer's torso (500, 300) also lies inside the FRONT dancer's big
    # box. With the known torso->foot offset only the back box fits.
    front = _box(440, 250, 120, 400)  # bottom at 650
    back = _box(470, 260, 60, 150)  # bottom at 410
    matched = match_points_to_boxes({"Back": (500, 300)}, [front, back], {"Back": (0, 110)})
    assert matched == {"Back": (500, 410)}


def test_foot_assist_never_returns_torso_as_foot(tmp_path):
    import cv2

    video = tmp_path / "v.avi"
    vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 24.0, (64, 48))
    for _ in range(6):
        vw.write(np.zeros((48, 64, 3), np.uint8))
    vw.release()

    class NoBoxes:
        def detect(self, frame):
            return []

    rep = {"A": [(30, 20)] * 6, "B": [(10, 20)] * 6}
    feet = compute_foot_anchors(
        video, rep, detector=NoBoxes(), sample_every=1, initial_offsets={"A": (0, 15)}
    )
    assert feet["A"][3] == (30, 35)  # seeded key-frame offset applied
    assert all(p is None for p in feet["B"])  # no offset known -> no fake foot


# ----------------------------------------------------- cotracker pipeline


def test_measured_click_foot_offset_overrides_zero_offset():
    clicks = [DancerClick(name="D", x=500, y=300)]
    tracks = [[TrackedPoint(0, 500, 300, True)]]
    frames = tracked_points_to_frames(tracks, clicks, _meta(1), click_foot_offsets=[200])
    assert frames[0].dancers[0].anchor_px == [500, 500]


def test_support_points_added_inside_matched_box():
    clicks = [DancerClick(name="D", x=500, y=270)]  # head click
    out = add_support_points(clicks, {"D": (450, 250, 100, 300)}, _meta())
    assert len(out) == 3
    assert all(c.name == "D" and 450 <= c.x <= 550 and 250 <= c.y <= 550 for c in out)


def test_support_point_not_duplicated_on_existing_click():
    clicks = [DancerClick(name="D", x=500, y=338)]  # already on the chest
    out = add_support_points(clicks, {"D": (450, 250, 100, 300)}, _meta())
    assert len(out) == 3  # an alternate spot replaces the duplicate
    ys = sorted(c.y for c in out)
    assert all(b - a >= 24 for a, b in zip(ys, ys[1:]))


def test_support_points_skipped_when_box_centre_far_from_click():
    clicks = [DancerClick(name="D", x=410, y=320)]  # box likely spans 2 people
    out = add_support_points(clicks, {"D": (400, 250, 200, 300)}, _meta())
    assert len(out) == 1


# --------------------------------------------------------------- geometry


def _camera_floor_corners(floor_w: float, floor_d: float, w: int = 1280, h: int = 720):
    f = 0.8 * w
    cam = np.array([0.0, 2.0, -7.0])
    c, s = math.cos(math.radians(7)), math.sin(math.radians(7))
    rot = np.array([[1, 0, 0], [0, c, s], [0, -s, c]])
    out = []
    for x, z in [(-floor_w / 2, 0), (floor_w / 2, 0), (floor_w / 2, floor_d), (-floor_w / 2, floor_d)]:
        p = rot @ (np.array([x, 0.0, z]) - cam)
        out.append([w / 2 + f * p[0] / p[2], h / 2 - f * p[1] / p[2]])
    return out


def test_floor_aspect_recovered_from_perspective():
    for fw, fd in [(8.0, 6.0), (10.0, 4.0), (6.0, 6.0)]:
        corners = _camera_floor_corners(fw, fd)
        hom = compute_stage_homography(corners)
        assert abs(estimate_floor_aspect(hom, (1280, 720)) - fw / fd) < 0.02 * fw / fd


def test_stage_drawn_with_true_floor_proportions():
    corners = _camera_floor_corners(6.0, 6.0)  # square floor
    hom = compute_stage_homography(corners, (1280, 720))
    bl = project_to_stage(hom, corners[3])
    br = project_to_stage(hom, corners[2])
    fl = project_to_stage(hom, corners[0])
    width_on_canvas = abs(br[0] - bl[0]) * CANVAS_ASPECT
    depth_on_canvas = abs(bl[1] - fl[1])
    assert abs(width_on_canvas / depth_on_canvas - 1.0) < 0.03


def test_rescue_ignores_a_few_outliers():
    pts = [(0.2 + 0.6 * (i % 10) / 9, 0.3 + 0.4 * (i // 10) / 19) for i in range(200)]
    pts += [(0.5, 1.4), (0.5, -0.3)]  # two glitch frames
    assert compute_rescue_affine(pts) == (1.0, 0.0, 0.0)


# ------------------------------------------------------ temporal handling


def _frames_from_series(series: dict[int, list[tuple[float, float]]]) -> list[FramePositions]:
    n = max(len(s) for s in series.values())
    frames = []
    for fi in range(n):
        dancers = [
            DancerPosition(id=did, bbox=[0, 0, 1, 1], anchor_px=[0, 0], x=s[fi][0], y=s[fi][1], confidence=1.0)
            for did, s in series.items()
            if fi < len(s)
        ]
        frames.append(FramePositions(frame=fi, timestamp_sec=fi / 24.0, dancers=dancers))
    return frames


def test_back_row_walk_splits_formations():
    # Two dancers hold, then BOTH walk 0.25 stage-depth units back over 1.5s,
    # then hold. The old image-pixel measure missed this kind of depth move.
    hold, move = 48, 36
    def path(x):
        p = [(x, 0.3)] * hold
        p += [(x, 0.3 + 0.25 * (k + 1) / move) for k in range(move)]
        p += [(x, 0.55)] * hold
        return p
    frames = _frames_from_series({1: path(0.3), 2: path(0.7)})
    forms = _segment_formations(frames, 24.0, movement_threshold=25.0, smoothing_window=9, min_duration_sec=0.5)
    assert len(forms) == 2
    assert forms[0].end_frame < hold + 6
    assert forms[1].start_frame > hold + move - 6


def test_smoothing_removes_jitter_and_spikes():
    rng = np.random.default_rng(1)
    clean = [(0.5, 0.5)] * 60
    noisy = [(x + float(rng.normal(0, 0.01)), y + float(rng.normal(0, 0.01))) for x, y in clean]
    noisy[30] = (0.9, 0.9)  # single-frame spike
    frames = _frames_from_series({1: noisy})
    smooth_trajectories(frames, fps=24.0)
    xs = [f.dancers[0].x for f in frames]
    assert max(abs(x - 0.5) for x in xs) < 0.02


def test_cotracker_frames_before_key_frame_are_invisible(tmp_path):
    torch = __import__("pytest").importorskip("torch")
    import cv2

    from app.services.cotracker import LocalClient

    video = tmp_path / "v.avi"
    vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 24.0, (640, 360))
    for _ in range(40):
        vw.write(np.zeros((360, 640, 3), np.uint8))
    vw.release()

    class FakeOnline:
        """Mimics CoTrackerOnlinePredictor: cumulative output, query coords
        echoed everywhere and 'visible' everywhere (as the real model's raw
        output can be before a query frame)."""

        step = 8

        def __call__(self, video_chunk, is_first_step=False, queries=None):
            if is_first_step:
                self.q = queries
                self.t = 0
                return None, None
            self.t += self.step
            T = self.t + self.step
            n = self.q.shape[1]
            tracks = self.q[:, None, :, 1:].expand(1, T, n, 2).clone()
            return tracks, torch.ones(1, T, n, dtype=torch.bool)

    LocalClient._model = FakeOnline()
    try:
        out = LocalClient(resize_width=320).track_video(
            video, [{"name": "A", "key_frame": 10, "x": 101, "y": 203}]
        )
    finally:
        LocalClient._model = None
    pts = out[0]
    assert len(pts) == 40
    assert not any(p.visible for p in pts[:10])
    assert all(p.visible for p in pts[10:])
    assert abs(pts[20].x - 101) <= 1 and abs(pts[20].y - 203) <= 1
