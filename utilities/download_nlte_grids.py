#!/usr/bin/env python3
"""
Download NLTE grid binaries, auxiliary data and model atoms listed in
`nlte_grids_links.cfg`.

Usage
-----
python3 download_nlte_grids.py <output_dir> <grid_types> <elements>

Examples
--------
# Download 1D **and** 3D grids for Al, Ba, Ca into ./grids/
python3 download_nlte_grids.py ./grids 1D,3D Al Ba Ca

# Download only 1D grids for Fe & Mg (case‑insensitive, commas or spaces)
python3 download_nlte_grids.py ./grids 1d Fe,Mg

Accepted values for *grid_types*:
  • 1D        – download 1‑D grid files (…1d_bin & …1d_aux)
  • 3D        – download 3‑D grid files (…3d_bin & …3d_aux)
  • 1D,3D     – download both (comma separated list, order doesn’t matter)
  • all       – synonym for 1D,3D
"""
from __future__ import annotations

import argparse
import configparser
import os
import sys
import textwrap
from typing import Iterable, List, Set, Tuple

import requests
from zipfile import ZipFile

# Created by storm at 20.06.25

# -----------------------------------------------------------------------------
# Argument parsing helpers
# -----------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(__doc__),
    )
    parser.add_argument(
        "output_dir", help="Directory where downloaded files will be stored."
    )
    parser.add_argument(
        "grid_types",
        help="Grid types to fetch: 1D, 3D, 1D,3D or all (case‑insensitive)",
    )
    parser.add_argument(
        "elements",
        nargs="+",
        help="Element symbols (case‑insensitive, space or comma separated)",
    )
    return parser.parse_args()


def _normalise_grid_types(raw: str) -> Set[str]:
    raw = raw.strip().lower()
    if raw == "all":
        return {"1d", "3d"}
    parts = {p.strip() for p in raw.split(",") if p.strip()}
    valid = {"1d", "3d"}
    if not parts.issubset(valid):
        raise ValueError(
            f"Invalid grid_types '{raw}'. Allowed values: 1D, 3D, 1D,3D, all."
        )
    return parts


def _normalise_elements(tokens: List[str]) -> List[str]:
    # Accept either commas or spaces between element symbols.
    joined = " ".join(tokens)
    elems: List[str] = [tok.capitalize() for tok in joined.replace(",", " ").split() if tok]
    return elems


# -----------------------------------------------------------------------------
# Download helpers
# -----------------------------------------------------------------------------

def _download_file(url: str, dest_path: str) -> Tuple[bool, bool]:
    """Download *url* to *dest_path* (skip if already present).

    Returns True on success, False if an exception occurs.
    Returns second True if skipped because the file already exists.
    """
    if os.path.exists(dest_path):
        print(f"    ✔ Exists, skipping: {os.path.basename(dest_path)}")
        return True, True

    if os.path.exists(dest_path.replace(".zip", "")):
        print(f"    ✔ Unzipped exists, skipping: {os.path.basename(dest_path)}")
        return True, True

    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, "wb") as fh:
                for chunk in r.iter_content(chunk_size=8192):
                    fh.write(chunk)
        return True, False  # Downloaded successfully, not skipped
    except Exception as exc:
        print(f"    ✖ Failed: {exc}")
        return False, False


def _maybe_unzip(path: str, *, delete_zip: bool = True) -> bool:
    """Unzip *path* if it ends with .zip. Returns True on success."""
    if not path.lower().endswith(".zip"):
        return True  # nothing to do

    try:
        with ZipFile(path) as zf:
            zf.extractall(os.path.dirname(path))
        if delete_zip:
            os.remove(path)
        print(f"    ✔ Unpacked & removed ZIP: {os.path.basename(path)}")
        return True
    except Exception as exc:
        print(f"    ✖ Unzip failed: {exc}")
        return False


# -----------------------------------------------------------------------------
# Main logic
# -----------------------------------------------------------------------------

def _build_task_list(cfg: configparser.ConfigParser, elements: Iterable[str], grids: Set[str]) -> List[Tuple[str, str, str, str]]:
    """Return a list of (element, tag, url, filename) tuples to download."""
    tasks: List[Tuple[str, str, str, str]] = []
    # get all elements from the config
    all_elements = set(cfg.sections())
    if elements == ["All"]:
        elements = all_elements
    for elem in elements:
        if elem not in cfg:
            print(f"! Warning: '{elem}' not found in cfg; skipping.")
            continue
        sec = cfg[elem]
        tasks.append((elem, "model_atom", sec.get("model_atom_link"), sec.get("model_atom_name")))
        if "1d" in grids:
            tasks.append((elem, "1d_bin", sec.get("1d_bin_link"), sec.get("1d_bin_name")))
            tasks.append((elem, "1d_aux", sec.get("1d_aux_link"), sec.get("1d_aux_name")))
        if "3d" in grids:
            tasks.append((elem, "3d_bin", sec.get("3d_bin_link"), sec.get("3d_bin_name")))
            tasks.append((elem, "3d_aux", sec.get("3d_aux_link"), sec.get("3d_aux_name")))
    # Filter out any (url or filename) missing
    filtered = [t for t in tasks if t[2] and t[3]]
    return filtered


def main() -> None:
    args = _parse_args()

    out_dir = os.path.abspath(args.output_dir)
    grid_types = _normalise_grid_types(args.grid_types)
    elements = _normalise_elements(args.elements)

    cfg_file = os.path.join(os.path.dirname(__file__), "nlte_grids_links.cfg")
    if not os.path.isfile(cfg_file):
        sys.exit(f"Configuration file not found: {cfg_file}")

    cfg = configparser.ConfigParser(interpolation=None)
    cfg.read(cfg_file)

    tasks = _build_task_list(cfg, elements, grid_types)
    total = len(tasks)
    done = 0
    errors: List[str] = []

    print("Starting downloads…\n")
    for idx, (elem, tag, url, fname) in enumerate(tasks, start=1):
        dest = os.path.join(out_dir, elem, fname)
        print(f"[{idx:>3}/{total}] {elem:2s} {tag:<8s} → {fname}")
        ok, skipped = _download_file(url, dest)
        if ok and not skipped:
            ok = _maybe_unzip(dest)
        if ok:
            done += 1
        else:
            errors.append(f"{elem} {tag}")

    print("\nFinished:")
    print(f"  Downloaded {done}/{total} files successfully.")
    if errors:
        print(f"  Errors ({len(errors)}): {', '.join(errors)}")
    else:
        print("  No errors encountered. 🎉")


if __name__ == "__main__":
    main()
