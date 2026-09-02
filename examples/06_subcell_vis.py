"""
SubcellVis: the spine-extraction pipeline in one NDWidget window.

Three ways in. Pick one and run this file, or paste a block into a notebook
(drop the ``fpl.loop.run()`` there; the cell's output is the canvas).

    python examples/06_subcell_vis.py path/to/experiment.zarr
    python examples/06_subcell_vis.py path/to/experiment.zarr --raw path/to/trials/
    python examples/06_subcell_vis.py --synthetic

The last one fabricates a movie with planted synapses, opens an empty window,
and runs registration + extraction on a background thread while the window
fills in: registered movies first, then the activity image and sources.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import fastplotlib as fpl

from subcell.config import ExtractionConfig, RegistrationConfig
from subcell.visualization import SubcellVis


def synthetic_demo() -> SubcellVis:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tests._synthetic import make_movie

    movie, frames_per_trial, _centers = make_movie(n_trials=3, n_frames=1200, size=64)
    vis = SubcellVis(raw=None, size=(1700, 950))
    vis.run(
        movie,
        output=Path(tempfile.mkdtemp(prefix="subcell_demo_")),
        registration=RegistrationConfig(
            maxshift=6, clip_shift=3, ds_time=2, init_frames=100,
            min_cluster_size=5, template_min_count=10, save_full_resolution=True,
        ),
        extraction=ExtractionConfig(dXY=2, max_synapse_density=0.02, n_parallel_workers=2),
        fs=400.0,
        frames_per_trial=frames_per_trial,
    )
    return vis


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("store", nargs="?", help="experiment.zarr, or the folder holding it")
    parser.add_argument("--raw", default=None, help="the unregistered movie, shown as the 'raw' layers")
    parser.add_argument("--synthetic", action="store_true", help="fabricate a movie and run the pipeline live")
    args = parser.parse_args()

    if args.synthetic:
        vis = synthetic_demo()
    else:
        vis = SubcellVis(args.store, raw=args.raw, label_names=("spine", "shaft", "junk"))
    vis.show()
    fpl.loop.run()


if __name__ == "__main__":
    main()
