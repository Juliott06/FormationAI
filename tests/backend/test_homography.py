from __future__ import annotations

from app.pipeline.homography import compute_stage_homography, project_to_stage


def test_axis_aligned_rectangle_is_identity_like():
    # A floor that already fills the frame as an axis-aligned rectangle:
    # corners at the image edges of a 1000x500 frame.
    corners = [[0, 0], [1000, 0], [1000, 500], [0, 500]]
    h = compute_stage_homography(corners)
    assert h is not None
    # Back-left image corner (0,0) -> stage (0,1); front-left (0,500) -> (0,0)
    assert project_to_stage(h, (0, 0)) == (0.0, 1.0)
    assert project_to_stage(h, (1000, 0)) == (1.0, 1.0)
    assert project_to_stage(h, (1000, 500)) == (1.0, 0.0)
    assert project_to_stage(h, (0, 500)) == (0.0, 0.0)
    # Center of frame -> center of stage
    cx, cy = project_to_stage(h, (500, 250))
    assert abs(cx - 0.5) < 1e-6
    assert abs(cy - 0.5) < 1e-6


def test_corners_in_arbitrary_order_are_classified():
    # Same rectangle, but clicked in a scrambled order.
    scrambled = [[1000, 500], [0, 0], [0, 500], [1000, 0]]
    h = compute_stage_homography(scrambled)
    assert h is not None
    # Geometry-based classification still maps back-left image (0,0) -> (0,1)
    assert project_to_stage(h, (0, 0)) == (0.0, 1.0)
    assert project_to_stage(h, (0, 500)) == (0.0, 0.0)


def test_trapezoid_uncompresses_the_back():
    # Perspective floor: the BACK edge is narrow (dancers squished together in
    # the image), the FRONT edge is wide. A homography should spread the back
    # edge back out to the full width.
    # back edge y=100, from x=400..600 (narrow). front edge y=500, x=100..900 (wide).
    corners = [[400, 100], [600, 100], [900, 500], [100, 500]]
    h = compute_stage_homography(corners)
    assert h is not None
    # The two back corners are narrow in the image but map to the full stage width
    bl = project_to_stage(h, (400, 100))
    br = project_to_stage(h, (600, 100))
    assert abs(bl[0] - 0.0) < 1e-6
    assert abs(br[0] - 1.0) < 1e-6
    # A point at the image-center of the back edge maps to stage x≈0.5 (centered),
    # not compressed toward one side.
    mid_back = project_to_stage(h, (500, 100))
    assert abs(mid_back[0] - 0.5) < 1e-6
    # And it sits at the back of the stage (y≈1)
    assert mid_back[1] > 0.99


def test_two_dancers_symmetric_at_back_stay_symmetric():
    # Symmetric formation: two dancers equidistant from center along the narrow
    # back edge should land symmetric about x=0.5 in stage space.
    corners = [[400, 100], [600, 100], [900, 500], [100, 500]]
    h = compute_stage_homography(corners)
    left = project_to_stage(h, (450, 100))
    right = project_to_stage(h, (550, 100))
    assert abs((left[0] + right[0]) / 2 - 0.5) < 1e-6


def test_rejects_wrong_count():
    assert compute_stage_homography([[0, 0], [1, 1], [2, 2]]) is None
    assert compute_stage_homography([]) is None
    assert compute_stage_homography(None) is None


def test_rejects_degenerate_collinear():
    # All four points on a line -> zero floor area -> unusable
    assert compute_stage_homography([[0, 0], [10, 10], [20, 20], [30, 30]]) is None


def test_project_handles_point_at_infinity_gracefully():
    # A pathological H whose denominator vanishes returns the stage center
    # rather than exploding.
    h = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    assert project_to_stage(h, (0, 0)) == (0.5, 0.5)
