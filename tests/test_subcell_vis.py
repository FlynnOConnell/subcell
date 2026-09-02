"""
Offscreen tests for ``SubcellVis``.

The viewer renders through a real fastplotlib figure, so these need the
offscreen rendercanvas backend that ``tests/conftest.py`` selects. They pin
the lazy movie stacks, the session loader, the overlay and table state, that
every panel draws, and one end-to-end pipeline run from a synthetic movie
watched by the window.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pytest

from tests._synthetic import (
    DS_FACTOR,
    FULL_HZ,
    H,
    NUM_CHANNELS,
    SOURCE_COLS,
    SOURCE_ROWS,
    W,
    make_movie,
    make_store,
)

pytest.importorskip("fastplotlib")
pytest.importorskip("imgui_bundle")
pytest.importorskip("masknmf")

from subcell.config import ExtractionConfig, RegistrationConfig  # noqa: E402
from subcell.visualization._session import InterleavedMovie, load_session  # noqa: E402
from subcell.visualization._vis_utils import decimate_minmax, spans_from_mask, visible_slice  # noqa: E402
from subcell.visualization.subcell_vis import UNLABELED, SubcellVis  # noqa: E402

FIGURE_SIZE = (1400, 850)


def offscreen_selected() -> bool:
    from rendercanvas.auto import RenderCanvas

    return "offscreen" in RenderCanvas.__module__


pytestmark = pytest.mark.skipif(
    not offscreen_selected(),
    reason="offscreen rendercanvas backend not selected",
)


class _DrawErrors(logging.Handler):
    """Collects the exceptions rendercanvas swallows inside a draw."""

    def __init__(self):
        super().__init__(level=logging.ERROR)
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(self.format(record))


def draw_frames(vis: SubcellVis, n: int = 2) -> None:
    """
    Render n frames, which runs every imgui panel.

    A panel that raises does not fail the draw: rendercanvas logs the error
    and moves on. So the log is watched, and any error fails the test.
    """
    handler = _DrawErrors()
    handler.setFormatter(logging.Formatter("%(message)s"))
    log = logging.getLogger("rendercanvas")
    log.addHandler(handler)
    try:
        canvas = vis.figure.canvas
        canvas.request_draw(canvas._draw_frame)
        for _ in range(n):
            canvas.draw()
    finally:
        log.removeHandler(handler)
    assert not handler.records, "draw raised:\n" + "\n".join(handler.records)


@pytest.fixture(scope="module")
def store_path(tmp_path_factory):
    return make_store(tmp_path_factory.mktemp("store"))


@pytest.fixture
def vis(store_path):
    widget = SubcellVis(store_path, size=FIGURE_SIZE)
    widget.show()
    yield widget
    widget.close()


# ----------------------------------------------------------------------
# pure helpers
# ----------------------------------------------------------------------


def test_spans_from_mask_gives_start_and_exclusive_end():
    mask = np.array([0, 1, 1, 0, 0, 1, 0], bool)
    spans = spans_from_mask(mask, hz=10.0)
    assert spans.tolist() == [[0.1, 0.3], [0.5, 0.6]]
    assert spans_from_mask(np.zeros(5, bool), 10.0).shape == (0, 2)


def test_visible_slice_pads_around_the_window():
    x = np.arange(100, dtype=np.float32)
    window = visible_slice(x, 10.0, 20.0, pad=2)
    assert (window.start, window.stop) == (8, 23)
    assert visible_slice(x, -50.0, 500.0).start == 0


def test_decimate_minmax_keeps_extremes_and_gaps():
    x = np.arange(10_000, dtype=np.float32)
    y = np.sin(x / 50.0).astype(np.float32)
    y[5000] = 10.0
    y[7000:7100] = np.nan
    xs, ys = decimate_minmax(x, y, 400)
    assert xs.size <= 400
    assert np.nanmax(ys) == 10.0
    assert np.isnan(ys).any()
    short_x, short_y = decimate_minmax(x[:10], y[:10], 400)
    assert short_x.size == 10


# ----------------------------------------------------------------------
# lazy arrays and the session
# ----------------------------------------------------------------------


def test_interleaved_movie_reads_one_channel(store_path):
    import zarr

    root = zarr.open(str(store_path), mode="r")
    arr = root["trials/trial_001/registered_ds"]
    movie = InterleavedMovie(arr, NUM_CHANNELS, channel=1)
    n_ds = arr.shape[2] // NUM_CHANNELS
    assert movie.shape == (n_ds, H, W)
    frame = movie[3]
    assert frame.shape == (H, W)
    np.testing.assert_array_equal(frame, np.asarray(arr[:, :, 3 * NUM_CHANNELS + 1]))
    block = movie[2:6]
    assert block.shape == (4, H, W)
    np.testing.assert_array_equal(block[1], movie[3])
    assert np.isnan(movie[n_ds + 5]).all()


def test_trial_movie_stack_pads_short_and_missing_trials(store_path):
    session = load_session(store_path)
    stack = session.movie_stack("registered ds · ch1")
    assert stack.shape == (2, 30, *session.canvas)
    assert not np.isnan(stack[0, 0]).all()
    assert np.isnan(stack[1, 29]).all()  # trial 2 has 25 frames
    block = stack[1, 20:30]
    assert block.shape == (10, *session.canvas)
    assert np.isnan(block[5:]).all() and not np.isnan(block[:5]).all()
    raw = session.movie_stack("registered raw · ch1")
    assert raw.has(0) and not raw.has(1)
    assert np.isnan(raw[1, 0]).all()
    sub = stack[0:2, 0:3, ::2, ::3]
    assert sub.shape == (2, 3, session.canvas[0] // 2 + session.canvas[0] % 2, len(range(0, session.canvas[1], 3)))


def test_session_reads_every_product(store_path):
    session = load_session(store_path)
    assert session.n_trials == 2
    assert session.n_sources == 3
    assert session.canvas == (H, W)
    assert "registered ds · ch2" in session.movie_names()
    assert "registered raw · ch1" in session.trials[0].movies
    assert "registered raw · ch1" not in session.trials[1].movies
    assert {"mean · ch1", "activity", "avg 8bit · ch1"} <= set(session.image_names())
    assert "summary activity" in session.summary_images
    first = session.trials[0]
    assert first.full_hz == pytest.approx(FULL_HZ)
    assert first.ds_factor == DS_FACTOR
    assert first.discard_ds is not None and first.discard_ds.any()
    assert first.discard_full.size == first.n_raw_frames
    assert session.valid_trials.tolist() == [0, 1]
    assert session.trial_corr.shape == (2,)


def test_session_tolerates_a_registration_only_store(tmp_path):
    from subcell.io.zarr_store import AlignmentData, ExperimentStore

    store = ExperimentStore(tmp_path / "experiment.zarr")
    arr = store.create_registered_ds(1, (8, 8, 6), 1)
    arr[:] = 1.0
    store.save_alignment_data(
        1,
        AlignmentData(1, 0.01, 25.0, np.zeros(24), np.zeros(24), np.zeros(6), np.zeros(6), np.zeros(6)),
    )
    session = load_session(tmp_path)
    assert session.n_trials == 1 and session.extraction is None
    assert session.trials[0].n_raw_frames == 24


# ----------------------------------------------------------------------
# the viewer
# ----------------------------------------------------------------------


def test_layers_cover_movies_images_and_summary(vis):
    names = vis.layer_names
    assert vis.layer == "registered ds · ch1"
    assert {"registered raw · ch1", "mean · ch1", "activity", "summary activity"} <= set(names)
    for name in names:
        vis.set_layer(name)
        assert vis.layer == name
        active = vis._layers[name]
        assert active.graphic.visible and not active.pause
        others = [nd for other, nd in vis._layers.items() if other != name]
        assert all(nd.pause and not nd.graphic.visible for nd in others)
        draw_frames(vis, n=1)


def test_trial_slider_drives_columns_and_overlays(vis):
    assert vis.current_trial() == 0
    assert vis.session.trials[0].n_raw_frames == 120
    vis.set_trial(1)
    assert vis.current_trial() == 1
    assert vis._columns["area"].shape == (3,)
    vis.set_trial(5)
    assert vis.current_trial() == 1
    vis.set_trial(0)
    vis.set_frame(40)
    assert vis.current_frame() == 40
    assert vis.current_time() == pytest.approx(40 / FULL_HZ)


def test_selection_paints_the_footprint_and_marker(vis):
    vis.select(1)
    assert vis.selected == 1
    rgba = vis._fp_overlay.data.value
    r, c = int(SOURCE_ROWS[1]), int(SOURCE_COLS[1])
    assert rgba[r, c, 3] > 0
    assert vis._sel_marker.visible
    np.testing.assert_allclose(vis._sel_marker.data.value[0, :2], (SOURCE_COLS[1], SOURCE_ROWS[1]))
    vis.select(-1)
    assert vis.selected == -1 and not vis._sel_marker.visible
    assert vis._fp_overlay.data.value[r, c, 3] > 0  # unselected footprints still drawn


def test_click_picks_the_source_under_the_cursor(vis):
    vis._pick(int(SOURCE_ROWS[2]), int(SOURCE_COLS[2]))
    assert vis.selected == 2
    vis._pick(int(SOURCE_ROWS[0]), int(SOURCE_COLS[0]), frozenset({"Ctrl"}))
    assert vis.group == [0]
    assert vis.plotted_sources() == [2, 0]
    vis._pick(0, 0)
    assert vis.selected == -1 and vis.group == []


def test_labels_persist_beside_the_store(vis, store_path):
    vis.select(0)
    vis.assign_class(0, advance=True)
    assert vis.labels[0] == 0
    assert vis.selected == 1  # advanced to the next row
    vis.toggle_group(2)
    vis.assign_class(1, advance=False)
    assert vis.labels[1] == 1 and vis.labels[2] == 1
    path = store_path.parent / "experiment.labels.npz"
    assert path.exists()
    saved = np.load(path)
    assert saved["labels"].tolist() == [0, 1, 1]
    assert vis.add_label("bouton") and "bouton" in vis.label_names
    vis.assign_class(UNLABELED, advance=False)
    assert vis.labels[1] == UNLABELED

    reopened = SubcellVis(store_path, size=FIGURE_SIZE)
    try:
        assert reopened.labels.tolist() == [0, UNLABELED, UNLABELED]
        assert "bouton" in reopened.label_names
    finally:
        reopened.close()
    vis._labels.clear()
    vis.save_labels()


def test_table_order_filters_by_area_and_label(vis):
    vis._order.range_limits = (100, 100)
    vis._order.rebuild()
    assert len(vis._order.order) == 0
    vis._order.set_range_column("area")
    vis._order.rebuild()
    assert len(vis._order.order) == 3
    vis.select(2)
    vis.step(-1)
    assert vis.selected == 1
    vis.next_unlabeled()
    assert vis.selected in (0, 1, 2)


def test_every_panel_draws(vis):
    vis.select(0)
    vis.toggle_group(1)
    vis._signals["ls"] = True
    vis._signals["F0"] = True
    vis._per_f0 = True
    draw_frames(vis, n=3)
    for tab in ("motion", "trials", "log", "traces"):
        vis.select_tab(tab)
        draw_frames(vis, n=2)
    vis._show_disks = True
    vis._show_mask = True
    vis._refresh_overlays()
    assert vis._mask_overlay.data.value[..., 3].max() > 0
    vis._settings_open = True
    vis._keybinds_open = True
    vis._open_prompt.start()
    draw_frames(vis, n=2)
    vis.set_layer("summary activity")
    vis.auto_contrast()
    vis.toggle_play()
    draw_frames(vis, n=2)
    vis.toggle_play()


def test_open_swaps_the_store(store_path, tmp_path):
    other = make_store(tmp_path / "other", n_trials=3, seed=3)
    widget = SubcellVis(store_path, size=FIGURE_SIZE)
    try:
        widget.show()
        widget.open(other)
        assert widget.session.n_trials == 3
        assert widget.fov_widget.ranges["trial"].stop == 3
        widget.set_trial(2)
        draw_frames(widget, n=2)
        widget.reload()
        assert widget.session.n_trials == 3
    finally:
        widget.close()


def test_empty_window_builds_and_draws():
    widget = SubcellVis(size=FIGURE_SIZE)
    try:
        widget.show()
        assert widget.layer_names == ["empty"]
        assert widget.n_sources == 0
        draw_frames(widget, n=2)
        widget.select_tab("motion")
        draw_frames(widget, n=1)
    finally:
        widget.close()


def test_raw_layers_show_the_unregistered_movie(store_path):
    movie, frames_per_trial, _ = make_movie(n_trials=2, n_frames=32, size=24)
    from subcell.visualization._run import resolve_source

    resolved = resolve_source(movie, fs=FULL_HZ, frames_per_trial=frames_per_trial)
    widget = SubcellVis(store_path, raw=resolved.array_source.array, size=FIGURE_SIZE)
    try:
        widget.show()
        assert "raw · ch1" in widget.layer_names
        widget.set_layer("raw · ch1")
        draw_frames(widget, n=2)
        stack = widget._layers["raw · ch1"].data
        assert stack.shape[0] == 2
        frame = stack[1, 0]
        assert frame.shape == widget.session.canvas
        np.testing.assert_array_equal(frame[:24, :24], movie[32, 0, 0].astype(np.float32))
    finally:
        widget.close()


# ----------------------------------------------------------------------
# running the pipeline from the window
# ----------------------------------------------------------------------


@pytest.mark.slow
def test_pipeline_run_from_an_array_fills_the_window(tmp_path):
    movie, frames_per_trial, centers = make_movie(n_trials=2, n_frames=480, size=48)
    registration = RegistrationConfig(
        maxshift=6,
        clip_shift=3,
        ds_time=2,
        init_frames=60,
        min_cluster_size=5,
        template_min_count=10,
        save_full_resolution=True,
        n_workers=1,
    )
    extraction = ExtractionConfig(
        dXY=2,
        nmf_iter=1,
        max_synapse_density=0.02,
        n_parallel_workers=1,
        device="cpu",
        valid_trial_corr_min=0.5,
    )
    widget = SubcellVis(size=FIGURE_SIZE)
    try:
        widget.show()
        run = widget.run(
            movie,
            output=tmp_path / "out",
            registration=registration,
            extraction=extraction,
            device="cpu",
            fs=400.0,
            frames_per_trial=frames_per_trial,
        )
        assert run.running
        deadline = time.perf_counter() + 600
        while run.running and time.perf_counter() < deadline:
            widget._poll_run()
            draw_frames(widget, n=1)
            time.sleep(0.2)
        assert not run.running, "pipeline run did not finish"
        widget._last_reload = 0.0
        widget._poll_run()
        assert run.status == "done", f"{run.status}: {run.error}\n" + "\n".join(list(run.log)[-20:])
        session = widget.session
        assert session.n_trials == 2
        assert all("registered raw · ch1" in t.movies for t in session.trials)
        assert session.trials[0].max_shift > 0
        assert session.extraction is not None, "\n".join(list(run.log)[-30:])
        assert session.n_sources >= 1
        found = np.column_stack([session.extraction.source_rows, session.extraction.source_cols])
        nearest = np.min(np.linalg.norm(found[:, None, :] - centers[None, :, :], axis=2), axis=1)
        assert np.median(nearest) < 4.0, f"sources {found.tolist()} vs planted {centers.tolist()}"
        widget.select(0)
        widget.select_tab("traces")
        draw_frames(widget, n=2)
        widget.select_tab("motion")
        draw_frames(widget, n=2)
        assert len(run.log) > 0
    finally:
        widget.close()
