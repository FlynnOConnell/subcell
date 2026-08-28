"""
Resolves the reference recording used by the MATLAB-equivalency scripts.

The scripts in this directory compare Python output against MATLAB intermediates
computed on one reference session. Point ``SPINE_TEST_DATA`` at the session
directory that holds the scan folders::

    set SPINE_TEST_DATA=D:\\iGluSnFR test data\\750098\\2024-09-24     (Windows)
    export SPINE_TEST_DATA="/data/iGluSnFR test data/750098/2024-09-24"  (POSIX)
"""

import os
from pathlib import Path

SCAN_DEFAULT = "test_scan_00001_20240924_110500"


def get_dir_session() -> Path:
    """
    Session directory holding the scan folders, from ``SPINE_TEST_DATA``.

    Returns
    -------
    Path
        Existing session directory.

    Raises
    ------
    RuntimeError
        If ``SPINE_TEST_DATA`` is unset or does not point at a directory.
    """
    dir_raw = os.environ.get("SPINE_TEST_DATA")
    if dir_raw is None:
        raise RuntimeError(
            "SPINE_TEST_DATA is not set. Point it at the session directory holding "
            "the scan folders, e.g. 'D:/iGluSnFR test data/750098/2024-09-24'."
        )
    dir_session = Path(dir_raw)
    if not dir_session.is_dir():
        raise RuntimeError(f"SPINE_TEST_DATA does not point at a directory: {dir_session}")
    return dir_session


def get_dir_scan(scan: str = SCAN_DEFAULT) -> Path:
    """
    Directory of a single scan within the reference session.

    Parameters
    ----------
    scan : str
        Scan folder name inside the session directory.

    Returns
    -------
    Path
        Existing scan directory.

    Raises
    ------
    RuntimeError
        If the scan folder does not exist.
    """
    dir_scan = get_dir_session() / scan
    if not dir_scan.is_dir():
        raise RuntimeError(f"Scan directory not found: {dir_scan}")
    return dir_scan
