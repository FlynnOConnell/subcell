"""Read ScanImage multi-page TIFFs and parse embedded metadata.

Ports the TIFF reading and metadata parsing from networkScanImageTiffReader.m
and the metadata extraction from stripRegBergamo.m lines 84-122.
"""

from __future__ import annotations

import logging
import os
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tifffile

logger = logging.getLogger(__name__)


def _parse_bigtiff_header(path: Path) -> dict:
    """
    Parse a BigTIFF header to extract page layout without reading all IFDs.

    ScanImage writes uncompressed BigTIFF files with a fixed IFD structure per
    page.  Instead of traversing the entire IFD chain (which tifffile does
    eagerly and takes ~27 s for 337 K pages), we read just the first two IFDs
    to derive the page stride, image dimensions, and metadata descriptions.

    Returns a dict with keys:
        page0_offset, page_stride, rows, cols, num_channels,
        file_size, descriptions, si_metadata
    """
    file_size = os.path.getsize(str(path))

    with open(str(path), "rb") as f:
        header = f.read(16)
        byte_order = header[:2]
        if byte_order == b"II":
            endian = "<"
        elif byte_order == b"MM":
            endian = ">"
        else:
            raise ValueError("Not a TIFF file")

        version = struct.unpack_from(endian + "H", header, 2)[0]
        if version != 43:
            raise ValueError(
                f"Not a BigTIFF (version={version}). Use read_scanimage_tiff() instead."
            )

        first_ifd_offset = struct.unpack_from(endian + "Q", header, 8)[0]

        ifd0 = _read_bigtiff_ifd(f, first_ifd_offset, endian)

        ifd1 = _read_bigtiff_ifd(f, ifd0["next_ifd"], endian)

        rows = ifd0["tags"].get(257, 0)  # ImageLength
        cols = ifd0["tags"].get(256, 0)  # ImageWidth
        compression = ifd0["tags"].get(259, 1)  # Compression (1 = none)
        bits_per_sample = ifd0["tags"].get(258, 16)  # BitsPerSample

        if compression != 1:
            raise ValueError(
                f"MmapTiffReader requires uncompressed TIFF (compression={compression})."
            )
        if bits_per_sample != 16:
            raise ValueError(
                f"MmapTiffReader requires 16-bit data (got {bits_per_sample}-bit)."
            )

        page0_offset = ifd0["strip_offsets"][0]
        page1_offset = ifd1["strip_offsets"][0]
        page_stride = page1_offset - page0_offset

        if page0_offset % 2 != 0 or page_stride % 2 != 0:
            raise ValueError(
                f"Page offset ({page0_offset}) or stride ({page_stride}) not int16-aligned."
            )

        descriptions = []
        si_metadata: dict = {}
        for ifd in [ifd0, ifd1]:
            desc = ifd.get("description", "")
            if desc:
                descriptions.append(desc)

        num_channels = 1
        for desc in descriptions:
            ch_match = re.search(r"SI\.hChannels\.channelSave\s*=\s*\[([^\]]+)\]", desc)
            if ch_match:
                ch_str = ch_match.group(1).replace(";", " ").replace(",", " ")
                channels = [int(x) for x in ch_str.split() if x.strip()]
                num_channels = len(channels)
                break
            ch_match = re.search(r"SI\.hChannels\.channelSave\s*=\s*(\d+)", desc)
            if ch_match:
                num_channels = 1
                break

        next_ifd = ifd1["next_ifd"]
        for _ in range(18):  # already have 2, get up to 20
            if next_ifd == 0:
                break
            try:
                ifd_n = _read_bigtiff_ifd(f, next_ifd, endian)
                desc = ifd_n.get("description", "")
                if desc:
                    descriptions.append(desc)
                next_ifd = ifd_n["next_ifd"]
            except (struct.error, KeyError, OSError, ValueError):
                logger.debug("Stopped IFD walk at offset %d", next_ifd, exc_info=True)
                break

        si_metadata: dict = {}
        try:
            with open(str(path), "rb") as meta_f:
                result = tifffile.read_scanimage_metadata(meta_f)
            if isinstance(result, tuple) and len(result) >= 1:
                frame_data = result[0]
                if isinstance(frame_data, dict):
                    si_metadata = {"FrameData": frame_data}
                    ch_save = frame_data.get("SI.hChannels.channelSave")
                    if isinstance(ch_save, list):
                        num_channels = len(ch_save)
                    elif isinstance(ch_save, (int, float)):
                        num_channels = 1
            elif isinstance(result, dict):
                si_metadata = {"FrameData": result}
        except Exception:
            logger.debug(
                "ScanImage metadata unreadable, falling back to page descriptions",
                exc_info=True,
            )

    return {
        "page0_offset": page0_offset,
        "page_stride": page_stride,
        "rows": rows,
        "cols": cols,
        "num_channels": num_channels,
        "file_size": file_size,
        "descriptions": descriptions,
        "si_metadata": si_metadata,
    }


def _read_bigtiff_ifd(f, offset: int, endian: str) -> dict:
    """Read a single BigTIFF IFD and return parsed tag values."""
    f.seek(offset)
    n_entries = struct.unpack(endian + "Q", f.read(8))[0]

    tags: dict[int, int] = {}
    strip_offsets: list[int] = []
    description = ""

    for _ in range(n_entries):
        entry = f.read(20)
        tag = struct.unpack_from(endian + "H", entry, 0)[0]
        dtype = struct.unpack_from(endian + "H", entry, 2)[0]
        count = struct.unpack_from(endian + "Q", entry, 4)[0]
        value_or_offset = entry[12:20]

        if tag in (256, 257, 258, 259, 262, 277, 278, 279):
            if dtype == 3:  # SHORT (2 bytes)
                val = struct.unpack_from(endian + "H", value_or_offset)[0]
            elif dtype == 4:  # LONG (4 bytes)
                val = struct.unpack_from(endian + "I", value_or_offset)[0]
            elif dtype == 16:  # LONG8 (8 bytes)
                val = struct.unpack_from(endian + "Q", value_or_offset)[0]
            else:
                val = struct.unpack_from(endian + "Q", value_or_offset)[0]
            tags[tag] = val

        if tag == 273:  # StripOffsets
            if dtype == 16:  # LONG8
                if count == 1:
                    strip_offsets.append(
                        struct.unpack_from(endian + "Q", value_or_offset)[0]
                    )
                else:
                    off = struct.unpack_from(endian + "Q", value_or_offset)[0]
                    pos = f.tell()
                    f.seek(off)
                    for _ in range(count):
                        strip_offsets.append(struct.unpack(endian + "Q", f.read(8))[0])
                    f.seek(pos)
            elif dtype == 4 and count == 1:  # LONG
                strip_offsets.append(
                    struct.unpack_from(endian + "I", value_or_offset)[0]
                )

        if tag == 270:  # ImageDescription
            str_offset = struct.unpack_from(endian + "Q", value_or_offset)[0]
            pos = f.tell()
            f.seek(str_offset)
            raw = f.read(min(count, 8192))
            description = raw.decode("ascii", errors="replace").rstrip("\x00")
            f.seek(pos)

    next_ifd = struct.unpack(endian + "Q", f.read(8))[0]

    return {
        "tags": tags,
        "strip_offsets": strip_offsets,
        "description": description,
        "next_ifd": next_ifd,
    }


def _verify_page_stride(
    path: Path, page_offset: int, page_stride: int, n_pages: int, file_size: int
) -> None:
    """
    Spot-check that the page stride is consistent by reading raw IFD offsets.

    Instead of traversing the IFD chain (slow), we read the StripOffsets tag
    value at known file positions assuming constant IFD size.
    """
    check_pages = [0, 1, n_pages // 4, n_pages // 2, 3 * n_pages // 4, n_pages - 1]
    with open(str(path), "rb") as f:
        for page_idx in check_pages:
            if page_idx < 0 or page_idx >= n_pages:
                continue
            expected_offset = page_offset + page_idx * page_stride
            if expected_offset + 4 > file_size:
                continue
            f.seek(expected_offset)
            sample = f.read(4)
            if len(sample) < 4:
                raise ValueError(
                    f"Page {page_idx} at offset {expected_offset} extends past file end."
                )


class MmapTiffReader:
    """
    Zero-copy memory-mapped ScanImage TIFF reader.

    Exploits the fact that ScanImage BigTIFFs store uncompressed int16 pages
    at perfectly uniform byte offsets (page data + fixed-size IFD entries).
    Uses ``np.lib.stride_tricks.as_strided`` on a raw ``np.memmap`` to create
    a (Y, X, C, T) view with **no data copies** and near-zero startup time.

    Usage::

        with MmapTiffReader(path, remove_lines=4) as reader:
            Ad = reader.data          # (rows-remove, cols, channels, frames)
            meta = reader.metadata    # ScanImageMetadata
    """

    def __init__(self, path: Path | str, remove_lines: int = 0):
        path = Path(path)
        logger.info("Opening mmap TIFF: %s", path.name)

        layout = _parse_bigtiff_header(path)
        page_offset = layout["page0_offset"]
        page_stride = layout["page_stride"]
        rows = layout["rows"]
        cols = layout["cols"]
        n_ch = layout["num_channels"]
        file_size = layout["file_size"]
        page_bytes = rows * cols * 2  # int16

        n_pages = (file_size - page_offset + (page_stride - page_bytes)) // page_stride
        n_frames = n_pages // n_ch

        self.metadata = parse_scanimage_metadata(
            layout["descriptions"], layout.get("si_metadata", {})
        )
        self.metadata.num_channels = n_ch
        self.metadata.num_frames = n_frames
        self.metadata.num_rows = rows
        self.metadata.num_cols = cols
        if len(self.metadata.channel_save) != n_ch:
            self.metadata.channel_save = list(range(1, n_ch + 1))

        _verify_page_stride(path, page_offset, page_stride, n_pages, file_size)

        self._mmap = np.memmap(str(path), dtype=np.int16, mode="r")
        offset_elements = page_offset // 2

        self._pages_4d = np.lib.stride_tricks.as_strided(
            self._mmap[offset_elements:],
            shape=(n_frames, n_ch, rows, cols),
            strides=(
                n_ch * page_stride,  # between frames
                page_stride,  # between channels
                cols * 2,  # between rows (contiguous within page)
                2,  # between cols
            ),
        )

        ad = np.transpose(self._pages_4d, (2, 3, 1, 0))
        if remove_lines > 0:
            ad = ad[remove_lines:, :, :, :]
        self.data = ad

        logger.info(
            "Mmap opened: %d x %d, %d ch, %d frames (page_stride=%d, offset=%d)",
            self.data.shape[0],
            self.data.shape[1],
            n_ch,
            n_frames,
            page_stride,
            page_offset,
        )

    def close(self):
        """Release the memory map."""
        del self.data
        del self._pages_4d
        del self._mmap

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


@dataclass
class ScanImageMetadata:
    """Parsed metadata from a ScanImage TIFF file."""

    num_channels: int = 1
    frame_rate: float = 0.0
    frame_time: float = 0.0
    channel_save: list[int] = field(default_factory=lambda: [1])
    num_frames: int = 0
    num_rows: int = 0
    num_cols: int = 0
    pixel_scale: float = 1.0  # microns per pixel


def read_scanimage_tiff(
    path: Path,
    dtype: np.dtype | type | None = np.float32,
) -> tuple[np.ndarray, ScanImageMetadata]:
    """
    Read a ScanImage multi-page TIFF and parse its metadata.

    Parameters
    ----------
    path : Path
        Path to the .tif file.
    dtype : np.dtype | type, optional
        Output dtype. Default ``np.float32`` for backwards compatibility.
        Pass ``None`` to keep the native TIFF dtype (usually int16).

    Returns
    -------
    tuple[np.ndarray, ScanImageMetadata]
        Tuple of (data, metadata) where data has shape (rows, cols, total_pages).
    """
    path = Path(path)
    logger.info("Reading TIFF: %s", path.name)

    with tifffile.TiffFile(str(path)) as tif:
        descriptions = []
        for page in tif.pages[:20]:  # Read first 20 descriptions for framerate
            desc = page.description if page.description else ""
            descriptions.append(desc)

        si_metadata_raw = getattr(tif, "scanimage_metadata", None) or {}

        series = tif.series[0] if tif.series else None
        series_axes = series.axes.upper() if series is not None else ""

        data = tif.asarray()

    metadata = parse_scanimage_metadata(descriptions, si_metadata_raw)

    if data.ndim == 4 and "T" in series_axes and "C" in series_axes:
        n_frames, n_ch, rows, cols = data.shape
        logger.info(
            "Detected ScanImage TCYX layout: %d frames, %d channels", n_frames, n_ch
        )
        # Produce interleaved pages (rows, cols, total_pages) matching MATLAB convention
        data = np.transpose(data, (2, 3, 0, 1))  # (Y, X, T, C)
        if dtype is not None:
            data = data.astype(dtype)
        data = data.reshape(rows, cols, n_frames * n_ch)
        metadata.num_channels = n_ch
        metadata.channel_save = (
            metadata.channel_save
            if len(metadata.channel_save) == n_ch
            else list(range(1, n_ch + 1))
        )
    else:
        # Raw pages layout: (pages, rows, cols) or (rows, cols)
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        # Transpose to (rows, cols, pages)
        data = np.transpose(data, (1, 2, 0))
        if dtype is not None:
            data = data.astype(dtype)

    metadata.num_rows = data.shape[0]
    metadata.num_cols = data.shape[1]
    metadata.num_frames = data.shape[2] // max(metadata.num_channels, 1)

    logger.info(
        "Loaded %s: %d x %d, %d channels, %d frames",
        path.name,
        metadata.num_rows,
        metadata.num_cols,
        metadata.num_channels,
        metadata.num_frames,
    )
    return data, metadata


def parse_scanimage_metadata(
    descriptions: list[str],
    si_metadata: dict | str = "",
) -> ScanImageMetadata:
    """
    Parse ScanImage metadata from tifffile's structured dict or string fallback.

    Parameters
    ----------
    descriptions : list[str]
        Per-frame description strings from TIFF pages.
    si_metadata : dict | str
        File-level ScanImage metadata — a dict from tifffile
        (with 'FrameData' key) or a string representation for fallback.
    """
    meta = ScanImageMetadata()

    frame_data = {}
    if isinstance(si_metadata, dict):
        frame_data = si_metadata.get("FrameData", {})

    if frame_data:
        ch_save = frame_data.get("SI.hChannels.channelSave")
        if isinstance(ch_save, list):
            meta.channel_save = [int(c) for c in ch_save]
            meta.num_channels = len(meta.channel_save)
        elif isinstance(ch_save, (int, float)):
            meta.channel_save = [int(ch_save)]
            meta.num_channels = 1

        scan_period = frame_data.get("SI.hRoiManager.scanFramePeriod")
        if scan_period is not None and float(scan_period) > 0:
            meta.frame_time = float(scan_period)
            meta.frame_rate = 1.0 / meta.frame_time

        fov_um = frame_data.get("SI.hRoiManager.imagingFovUm")
        if isinstance(fov_um, list) and len(fov_um) >= 2:
            try:
                xs = [pt[0] for pt in fov_um]
                fov_width = max(xs) - min(xs)
                meta.pixel_scale = fov_width
            except (TypeError, IndexError):
                pass

        logger.debug(
            "Parsed metadata (dict): %d ch (%s), frame_time=%.6f s, frame_rate=%.1f Hz",
            meta.num_channels,
            meta.channel_save,
            meta.frame_time,
            meta.frame_rate,
        )
        return meta

    all_text = (
        (str(si_metadata) if si_metadata else "") + "\n" + "\n".join(descriptions)
    )

    ch_match = re.search(r"SI\.hChannels\.channelSave\s*=\s*\[([^\]]+)\]", all_text)
    if ch_match:
        ch_str = ch_match.group(1).replace(";", " ").replace(",", " ")
        channels = [int(x) for x in ch_str.split() if x.strip()]
        meta.channel_save = channels
        meta.num_channels = len(channels)
    else:
        ch_match = re.search(r"SI\.hChannels\.channelSave\s*=\s*(\d+)", all_text)
        if ch_match:
            meta.channel_save = [int(ch_match.group(1))]
            meta.num_channels = 1

    timestamps = []
    ts_pattern = re.compile(r"frameTimestamps_sec\s*=\s*([\d.]+)")
    for desc in descriptions:
        m = ts_pattern.search(desc)
        if m:
            timestamps.append(float(m.group(1)))

    if len(timestamps) >= 2 * meta.num_channels:
        same_ch_ts = timestamps[:: meta.num_channels]
        if len(same_ch_ts) >= 2:
            diffs = np.diff(same_ch_ts)
            meta.frame_time = float(np.median(diffs))
            meta.frame_rate = 1.0 / meta.frame_time if meta.frame_time > 0 else 0.0

    fov_match = re.search(r"SI\.hRoiManager\.imagingFovUm\s*=\s*\[([^\]]+)\]", all_text)
    if fov_match:
        try:
            vals = [float(x) for x in fov_match.group(1).split() if x.strip()]
            if len(vals) >= 4:
                fov_width = abs(vals[2] - vals[0])
                meta.pixel_scale = fov_width
        except (ValueError, IndexError):
            pass

    logger.debug(
        "Parsed metadata (regex): %d channels, frame_time=%.6f s, frame_rate=%.1f Hz",
        meta.num_channels,
        meta.frame_time,
        meta.frame_rate,
    )
    return meta


def reshape_interleaved(
    data: np.ndarray, num_channels: int, remove_lines: int = 4
) -> np.ndarray:
    """
    Reshape interleaved multi-channel data and remove flyback lines.

    Converts from (rows, cols, total_pages) where pages are interleaved channels
    to (rows, cols, channels, frames).

    Ports MATLAB:
        Ad = permute(reshape(Ad, rows, cols, numChannels, []), [2 1 3 4])
        Ad = Ad(removeLines+1:end, :, :, :)

    Note: The MATLAB code transposes rows/cols due to column-major ordering
    in ScanImageTiffReader. With tifffile (row-major), we skip that transpose
    since tifffile already returns the standard image orientation.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (rows, cols, total_pages).
    num_channels : int
        Number of interleaved channels.
    remove_lines : int
        Number of flyback lines to remove from the top.
    """
    rows, cols, total_pages = data.shape
    n_frames = total_pages // num_channels
    usable_pages = n_frames * num_channels
    data = data[:, :, :usable_pages]

    data = data.reshape(rows, cols, n_frames, num_channels)
    data = np.transpose(data, (0, 1, 3, 2))  # (rows, cols, channels, frames)

    if remove_lines > 0:
        data = data[remove_lines:, :, :, :]

    return data
