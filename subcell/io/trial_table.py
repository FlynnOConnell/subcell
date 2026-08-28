"""Trial table management: build, save, and load trial manifests.

Ports buildTrialTableBergamo.m - organizes multi-trial recordings into a
structured manifest, excluding already-processed files.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

EXCLUDE_PATTERNS = ("REGISTERED", "FIGURE")


@dataclass
class TrialEntry:
    """A single trial in the experiment."""

    filename: str
    trial_index: int
    epoch: int


@dataclass
class TrialTable:
    """
    Manifest of all trials in an experiment.

    Tracks filenames and metadata through the processing pipeline.
    After registration, output paths are populated.
    """

    directory: str
    entries: list[TrialEntry] = field(default_factory=list)
    fn_reg_ds: dict[int, str] = field(default_factory=dict)
    fn_raw: dict[int, str] = field(default_factory=dict)
    fn_adata: dict[int, str] = field(default_factory=dict)
    align_params: dict = field(default_factory=dict)

    @classmethod
    def from_directory(
        cls,
        directory: Path,
        exclude_patterns: tuple[str, ...] = EXCLUDE_PATTERNS,
        epoch: int = 1,
    ) -> TrialTable:
        """
        Auto-build trial table from all .tif files in a directory.

        Ports buildTrialTableBergamo.m auto-mode (lines 26-36):
        - Finds all .tif files
        - Excludes files containing 'REGISTERED' or 'FIGURE'
        - Assigns all files to epoch=1

        Parameters
        ----------
        directory : Path
            Path to directory containing TIFF files.
        exclude_patterns : tuple[str, ...]
            Filename substrings to exclude.
        epoch : int
            Epoch to assign to all found files.
        """
        directory = Path(directory)
        tif_files = sorted(directory.glob("*.tif"))

        entries = []
        trial_ix = 0
        for tif_path in tif_files:
            name = tif_path.name
            if any(pat in name for pat in exclude_patterns):
                logger.debug("Excluding %s (matches exclusion pattern)", name)
                continue
            trial_ix += 1
            entries.append(TrialEntry(filename=name, trial_index=trial_ix, epoch=epoch))

        logger.info("Found %d trial files in %s", len(entries), directory)
        return cls(directory=str(directory), entries=entries)

    def save(self, path: Path) -> None:
        """Save trial table as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "directory": self.directory,
            "entries": [asdict(e) for e in self.entries],
            "fn_reg_ds": {str(k): v for k, v in self.fn_reg_ds.items()},
            "fn_raw": {str(k): v for k, v in self.fn_raw.items()},
            "fn_adata": {str(k): v for k, v in self.fn_adata.items()},
            "align_params": self.align_params,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Trial table saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> TrialTable:
        """Load trial table from JSON."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        entries = [TrialEntry(**e) for e in data["entries"]]
        return cls(
            directory=data["directory"],
            entries=entries,
            fn_reg_ds={int(k): v for k, v in data.get("fn_reg_ds", {}).items()},
            fn_raw={int(k): v for k, v in data.get("fn_raw", {}).items()},
            fn_adata={int(k): v for k, v in data.get("fn_adata", {}).items()},
            align_params=data.get("align_params", {}),
        )

    @property
    def n_trials(self) -> int:
        return len(self.entries)

    def get_filepath(self, trial_index: int) -> Path:
        """Get full path to a trial's source TIFF file."""
        for entry in self.entries:
            if entry.trial_index == trial_index:
                return Path(self.directory) / entry.filename
        raise KeyError(f"Trial index {trial_index} not found")
