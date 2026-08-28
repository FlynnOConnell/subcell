"""Data loaders for NMF extraction results.

Supports:
- Python Zarr stores (from runner.py or manual save)
- MATLAB v7.3 .mat files (HDF5, from summarize_LoCo.m)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ._data_model import ExtractionData, TrialExtraction

logger = logging.getLogger(__name__)


def load_auto(path: str | Path) -> ExtractionData:
    """Auto-detect format and load extraction results.

    Dead sources (zero footprints across all trials) are pruned before
    returning so downstream code never sees them.

    Args:
        path: Path to a .zarr directory or .mat file.
    """
    path = Path(path)
    if path.suffix == ".mat":
        data = load_mat(path)
    elif path.suffix == ".zarr" or path.is_dir():
        data = load_zarr(path)
    else:
        raise ValueError(f"Unknown format: {path.suffix}. Expected .zarr or .mat")

    return _prune_dead_sources(data)


def _prune_dead_sources(data: ExtractionData) -> ExtractionData:
    """Remove sources whose footprint is zero in every trial."""
    n = data.n_sources
    if n == 0:
        return data

    # A source is alive if it has a positive footprint in at least one trial
    alive = np.zeros(n, dtype=bool)
    for trial in data.trials.values():
        fp_max = np.nanmax(trial.footprints, axis=0)  # (n_sources,)
        alive |= fp_max > 0

    if np.all(alive):
        return data  # nothing to prune

    keep = np.where(alive)[0]
    n_pruned = n - len(keep)
    logger.info("Pruned %d dead sources (%d → %d)", n_pruned, n, len(keep))

    # Subset per-trial arrays
    pruned_trials: dict[int, TrialExtraction] = {}
    for key, trial in data.trials.items():
        pruned_trials[key] = TrialExtraction(
            footprints=trial.footprints[:, keep],
            events=trial.events[keep, :],
            denoised=trial.denoised[keep, :],
            ls=trial.ls[keep, :],
            f0=trial.f0[keep, :],
            snr=trial.snr[keep],
            discard_frames=trial.discard_frames,
        )

    initial_fp = None
    if data.initial_footprints is not None:
        initial_fp = data.initial_footprints[:, :, keep]

    return ExtractionData(
        mean_image=data.mean_image,
        sel_pix=data.sel_pix,
        source_rows=data.source_rows[keep],
        source_cols=data.source_cols[keep],
        trials=pruned_trials,
        analyze_hz=data.analyze_hz,
        source_name=data.source_name,
        file_path=data.file_path,
        initial_footprints=initial_fp,
    )


# ---------------------------------------------------------------------------
# Zarr loader
# ---------------------------------------------------------------------------

def load_zarr(zarr_path: str | Path) -> ExtractionData:
    """Load extraction results from a Zarr store.

    Expected hierarchy (from runner.py / ExperimentStore):
        summary/
            mean_image          (H, W, C) float32
            activity_image      (H, W) float32
            sources/
                locations_r     (n_sources,) float64
                locations_c     (n_sources,) float64
                sel_pix         (H, W) bool
        trials/
            trial_XXX/
                extraction/
                    footprints      (n_sel_pix, n_sources) float32
                    events          (n_sources, n_time) float32
                    denoised        (n_sources, n_time) float32
                    ls              (n_sources, n_time) float32
                    F0              (n_sources, n_time) float32
                    SNR             (n_sources,) float32
                    discard_frames  (n_time,) bool
                alignment/
                    attrs: frame_time, align_hz
    """
    import zarr

    zarr_path = Path(zarr_path)
    root = zarr.open(str(zarr_path), mode="r")

    # Summary data
    summ = root["summary"]
    mean_image = np.array(summ["mean_image"])  # (H, W, C)
    sel_pix = np.array(summ["sources"]["sel_pix"]).astype(bool)  # (H, W)
    source_rows = np.array(summ["sources"]["locations_r"])
    source_cols = np.array(summ["sources"]["locations_c"])

    # Optional: per-source initial footprint masks
    initial_footprints = None
    if "initial_footprints" in summ["sources"]:
        initial_footprints = np.array(summ["sources"]["initial_footprints"]).astype(bool)

    # Per-trial extraction
    trials: dict[int, TrialExtraction] = {}
    analyze_hz = 0.0

    trials_grp = root["trials"]
    for key in sorted(trials_grp.keys()):
        trial_idx = int(key.split("_")[-1])
        trial_grp = trials_grp[key]

        if "extraction" not in trial_grp:
            logger.debug("Trial %d has no extraction results, skipping", trial_idx)
            continue

        ext = trial_grp["extraction"]
        trial = TrialExtraction(
            footprints=np.array(ext["footprints"]),
            events=np.array(ext["events"]),
            denoised=np.array(ext["denoised"]),
            ls=np.array(ext["ls"]),
            f0=np.array(ext["F0"]),
            snr=np.array(ext["SNR"]),
            discard_frames=np.array(ext["discard_frames"]).astype(bool),
        )
        trials[trial_idx] = trial

        # Get framerate from alignment metadata
        if analyze_hz == 0.0 and "alignment" in trial_grp:
            align_attrs = trial_grp["alignment"].attrs
            frame_time = float(align_attrs.get("frame_time", 0))
            if frame_time > 0:
                analyze_hz = 1.0 / frame_time
            else:
                analyze_hz = float(align_attrs.get("align_hz", 1.0))

    if not trials:
        raise ValueError(f"No extraction results found in {zarr_path}")

    logger.info(
        "Loaded Zarr: %d trials, %d sources, %.1f Hz",
        len(trials), len(source_rows), analyze_hz,
    )

    return ExtractionData(
        mean_image=mean_image,
        sel_pix=sel_pix,
        source_rows=source_rows,
        source_cols=source_cols,
        trials=trials,
        analyze_hz=analyze_hz,
        source_name="zarr",
        file_path=str(zarr_path),
        initial_footprints=initial_footprints,
    )


# ---------------------------------------------------------------------------
# MATLAB v7.3 loader
# ---------------------------------------------------------------------------

def _deref(f, ds):
    """Dereference an HDF5 dataset containing a MATLAB object reference.

    Recurses through chained references (e.g. peaks → ref → ref → Group).
    """
    import h5py

    if isinstance(ds, h5py.Dataset) and ds.dtype == object:
        if ds.shape == (1, 1):
            target = f[ds[0, 0]]
            return _deref(f, target)  # recurse in case of chained refs
        # Cell array: return list of derefs
        return [_deref(f, f[ref]) for ref in ds.flat]
    if isinstance(ds, h5py.Dataset):
        return np.array(ds)
    return ds


def _read_array(f, ds):
    """Read a dataset or dereference and read."""
    import h5py

    target = _deref(f, ds)
    if isinstance(target, (h5py.Dataset, np.ndarray)):
        return np.array(target)
    if isinstance(target, h5py.Group):
        raise TypeError(f"Expected dataset, got group: {target.name}")
    return np.array(target)


def load_mat(mat_path: str | Path) -> ExtractionData:
    """Load extraction results from a MATLAB SummaryLoCo-*.mat file.

    MATLAB v7.3 files are HDF5. Arrays are transposed relative to MATLAB
    (h5py reads in row-major, MATLAB stores column-major).
    """
    import h5py

    mat_path = Path(mat_path)
    f = h5py.File(str(mat_path), "r")

    try:
        es = f["exptSummary"]

        # Mean image: MATLAB (M,N,C) stored as HDF5 (C,N,M).
        # MATLAB's first dim is the fast-scan axis; Python convention puts
        # the slow axis first (H) so we use transpose(1,2,0) → (N,M,C)
        # which also makes column-major footprint order == row-major order.
        mean_im_raw = _read_array(f, es["meanIM"])
        if mean_im_raw.ndim == 3:
            mean_image = mean_im_raw.transpose(1, 2, 0).astype(np.float32)
        elif mean_im_raw.ndim == 2:
            mean_image = mean_im_raw.astype(np.float32)[:, :, np.newaxis]
        else:
            mean_image = mean_im_raw.astype(np.float32)

        # selPix: same axis swap — HDF5 (N,M) stays as-is → (N,M) = (H,W)
        sel_pix_raw = _read_array(f, es["selPix"])
        sel_pix = sel_pix_raw.astype(bool)

        # Peaks (source locations).
        # MATLAB row = fast-scan axis = our columns (W dim)
        # MATLAB col = slow-scan axis = our rows (H dim)
        peaks_target = _deref(f, es["peaks"])
        if isinstance(peaks_target, h5py.Group):
            mat_rows = _read_array(f, peaks_target["row"]).ravel() - 1  # 0-indexed
            mat_cols = _read_array(f, peaks_target["col"]).ravel() - 1
        elif isinstance(peaks_target, list):
            # Cell array of peaks per DMD - take first
            p0 = peaks_target[0]
            if isinstance(p0, h5py.Group):
                mat_rows = _read_array(f, p0["row"]).ravel() - 1
                mat_cols = _read_array(f, p0["col"]).ravel() - 1
            else:
                arr = _read_array(f, p0)
                mat_rows = arr[0, :] - 1
                mat_cols = arr[1, :] - 1
        else:
            arr = _read_array(f, peaks_target)
            mat_rows = arr[0, :] - 1
            mat_cols = arr[1, :] - 1

        # Swap: MATLAB row→our col, MATLAB col→our row
        source_rows = mat_cols
        source_cols = mat_rows

        # Framerate from params or aData
        analyze_hz = 1.0
        try:
            params_grp = es["params"]
            if isinstance(params_grp, h5py.Group):
                if "analyzeHz" in params_grp:
                    analyze_hz = float(np.array(params_grp["analyzeHz"]).ravel()[0])
            elif isinstance(params_grp, h5py.Dataset):
                params_deref = _deref(f, params_grp)
                if isinstance(params_deref, h5py.Group) and "analyzeHz" in params_deref:
                    analyze_hz = float(np.array(params_deref["analyzeHz"]).ravel()[0])
        except Exception:
            pass

        if analyze_hz <= 1.0:
            try:
                adata = _deref(f, es["aData"])
                if isinstance(adata, h5py.Group):
                    ft = float(np.array(adata["frametime"]).ravel()[0])
                    if ft > 0:
                        analyze_hz = 1.0 / ft
                elif isinstance(adata, list) and len(adata) > 0:
                    a0 = adata[0]
                    if isinstance(a0, h5py.Group) and "frametime" in a0:
                        ft = float(np.array(a0["frametime"]).ravel()[0])
                        if ft > 0:
                            analyze_hz = 1.0 / ft
            except Exception:
                pass

        # Load E struct (per-trial extraction results)
        E_ds = es["E"]
        trials: dict[int, TrialExtraction] = {}
        n_sel = int(np.sum(sel_pix))

        e_target = _deref(f, E_ds)
        if isinstance(e_target, h5py.Group):
            # Single trial struct
            trial = _load_mat_trial(f, e_target, n_sel, len(source_rows))
            if trial is not None:
                trials[0] = trial
        elif isinstance(e_target, list):
            # Cell array of trial structs
            for idx, item in enumerate(e_target):
                grp = _deref(f, item) if not isinstance(item, h5py.Group) else item
                if isinstance(grp, h5py.Group):
                    trial = _load_mat_trial(f, grp, n_sel, len(source_rows))
                    if trial is not None:
                        trials[idx] = trial

        if not trials:
            raise ValueError(
                f"No extraction results with footprints found in {mat_path}. "
                f"E resolved to: {type(e_target).__name__}"
            )

        # No footprint pixel reordering needed: MATLAB column-major order
        # of (M, N) == Python row-major order of (N, M) after axis swap.

        # If peaks count doesn't match footprints, derive centers from H maxima
        first_trial = next(iter(trials.values()))
        n_sources_from_h = first_trial.footprints.shape[1]
        if len(source_rows) != n_sources_from_h:
            logger.info(
                "Peaks count (%d) != footprint sources (%d), deriving centers from H maxima",
                len(source_rows), n_sources_from_h,
            )
            sel_coords = np.argwhere(sel_pix)  # (n_sel, 2) → (row, col)
            source_rows = np.zeros(n_sources_from_h, dtype=np.float64)
            source_cols = np.zeros(n_sources_from_h, dtype=np.float64)
            for i in range(n_sources_from_h):
                peak_idx = np.nanargmax(first_trial.footprints[:, i])
                source_rows[i] = sel_coords[peak_idx, 0]
                source_cols[i] = sel_coords[peak_idx, 1]

        logger.info(
            "Loaded MAT: %d trials, %d sources, %.1f Hz",
            len(trials), len(source_rows), analyze_hz,
        )

        return ExtractionData(
            mean_image=mean_image,
            sel_pix=sel_pix,
            source_rows=source_rows,
            source_cols=source_cols,
            trials=trials,
            analyze_hz=analyze_hz,
            source_name="mat",
            file_path=str(mat_path),
        )

    finally:
        f.close()


def _load_mat_trial(
    f, e_grp, n_sel: int, n_peaks: int
) -> TrialExtraction | None:
    """Load a single trial's extraction data from a MATLAB E struct group."""
    import h5py

    fields = list(e_grp.keys())
    if "footprints" not in fields:
        logger.debug("E group missing 'footprints', fields: %s", fields)
        return None

    # Footprints: MATLAB (n_sel, n_src) → HDF5 (n_src, n_sel) → .T
    mat_H = np.array(e_grp["footprints"])
    if mat_H.shape[0] == n_sel:
        pass  # already (n_sel, n_src)
    elif mat_H.shape[1] == n_sel:
        mat_H = mat_H.T
    # else: auto-orient by checking which dim matches
    footprints = mat_H.astype(np.float32)
    n_sources = footprints.shape[1]

    # Navigate dF sub-struct
    dF_target = _deref(f, e_grp["dF"]) if "dF" in e_grp else None
    if dF_target is None:
        logger.debug("E group missing 'dF'")
        return None

    def _read_signal(grp, name):
        arr = _read_array(f, grp[name])
        # Auto-orient to (n_sources, n_time)
        if arr.shape[0] != n_sources and arr.shape[1] == n_sources:
            arr = arr.T
        return arr.astype(np.float32)

    events = _read_signal(dF_target, "events")
    denoised = _read_signal(dF_target, "denoised")
    ls = _read_signal(dF_target, "ls")

    # F0
    if "F0" in fields:
        f0 = _read_array(f, e_grp["F0"])
        if f0.shape[0] != n_sources and len(f0.shape) > 1 and f0.shape[1] == n_sources:
            f0 = f0.T
        f0 = f0.astype(np.float32)
    else:
        f0 = np.zeros_like(events)

    # SNR
    if "SNR" in fields:
        snr = _read_array(f, e_grp["SNR"]).ravel().astype(np.float32)
    else:
        snr = np.zeros(n_sources, dtype=np.float32)

    # Discard frames
    if "discardFrames" in fields:
        discard = _read_array(f, e_grp["discardFrames"]).ravel().astype(bool)
    else:
        discard = np.zeros(events.shape[1], dtype=bool)

    return TrialExtraction(
        footprints=footprints,
        events=events,
        denoised=denoised,
        ls=ls,
        f0=f0,
        snr=snr,
        discard_frames=discard,
    )
