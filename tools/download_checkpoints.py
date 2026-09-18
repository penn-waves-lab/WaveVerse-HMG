"""Download and install the WaveVerse-HMG weight-only checkpoint archive."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/checkpoints.json"
DEFAULT_URL = "https://drive.google.com/file/d/1CBIdN_3oH6LlRQZcksORe2NMNDZAqDZG/view?usp=sharing"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def destination(asset_root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError("Invalid checkpoint path: " + name)
    target = asset_root.joinpath(*relative.parts)
    try:
        target.resolve().relative_to(asset_root.resolve())
    except ValueError:
        raise ValueError("Checkpoint destination leaves the asset directory: " + str(target))
    return target


def install_archive(archive, asset_root, manifest, overwrite=False):
    """Verify every member before installing; preserve differing existing files."""
    if sha256(archive) != manifest["sha256"]:
        raise ValueError("Archive SHA-256 mismatch; use the published checkpoint ZIP.")
    expected = manifest["files"]
    asset_root = Path(asset_root).resolve()
    targets = {name: destination(asset_root, name) for name in expected}
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        if len(members) != len(expected) or {info.filename for info in members} != set(expected):
            raise ValueError("Archive members do not match the checkpoint manifest.")
        for info in members:
            if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("Expected a regular checkpoint file: " + info.filename)
            if info.file_size != expected[info.filename]["bytes"]:
                raise ValueError("Checkpoint size mismatch: " + info.filename)
        asset_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".hmg-install-", dir=str(asset_root)) as work:
            staging = Path(work)
            for name, metadata in expected.items():
                staged = staging / name
                staged.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(name) as source, staged.open("xb") as output:
                    shutil.copyfileobj(source, output)
                if sha256(staged) != metadata["sha256"]:
                    raise ValueError("Checkpoint SHA-256 mismatch: " + name)
            pending = []
            for name, target in targets.items():
                if target.exists():
                    if not target.is_file():
                        raise FileExistsError(
                            "Checkpoint destination is not a file: " + str(target)
                        )
                    if sha256(target) == expected[name]["sha256"]:
                        print("Already installed:", target)
                        continue
                    if not overwrite:
                        raise FileExistsError(
                            "A different checkpoint already exists: "
                            + str(target)
                            + ". Use a new --asset-root or --overwrite."
                        )
                pending.append((name, target))
            for name, target in pending:
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(str(staging / name), str(target))
                print("Installed:", target)
    return targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--url", default=DEFAULT_URL, help="Override the published Google Drive checkpoint URL"
    )
    source.add_argument("--archive", type=Path, help="Install a ZIP already downloaded locally")
    parser.add_argument("--asset-root", type=Path, default=ROOT / "assets")
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace differing checkpoint files"
    )
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    if args.archive:
        install_archive(args.archive.resolve(), args.asset_root, manifest, args.overwrite)
        return
    try:
        import gdown
    except ImportError:
        parser.error(
            "Missing gdown. Install the release dependencies: pip install -r requirements.txt"
        )
    with tempfile.TemporaryDirectory(prefix="waveverse-hmg-download-") as work:
        archive = Path(work) / manifest["filename"]
        downloaded = gdown.download(
            url=args.url, output=str(archive), fuzzy=True, use_cookies=False
        )
        if not downloaded:
            raise RuntimeError(
                "Download failed. Check that the Google Drive file is publicly shared."
            )
        install_archive(archive, args.asset_root, manifest, args.overwrite)


if __name__ == "__main__":
    main()
