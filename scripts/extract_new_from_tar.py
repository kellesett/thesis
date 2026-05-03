#!/usr/bin/env python3
"""Extract only missing files from a gzip tar stream.

Used by ``make pull-results SYNC_CONFLICT=skip``.  Existing files under the
destination directory are left untouched; missing files are created.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
from pathlib import Path, PurePosixPath


def safe_member_path(dest: Path, name: str) -> Path | None:
    posix = PurePosixPath(name)
    if posix.is_absolute() or ".." in posix.parts:
        return None
    parts = [part for part in posix.parts if part not in ("", ".")]
    if not parts:
        return dest
    return dest.joinpath(*parts)


def extract_new(dest: Path) -> tuple[int, int, int]:
    extracted = 0
    skipped_existing = 0
    skipped_unsupported = 0
    dest.mkdir(parents=True, exist_ok=True)

    with tarfile.open(fileobj=sys.stdin.buffer, mode="r|gz") as archive:
        for member in archive:
            target = safe_member_path(dest, member.name)
            if target is None:
                skipped_unsupported += 1
                continue

            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue

            if not member.isfile():
                skipped_unsupported += 1
                continue

            if target.exists():
                skipped_existing += 1
                src = archive.extractfile(member)
                if src is not None:
                    with open(os.devnull, "wb") as devnull:
                        shutil.copyfileobj(src, devnull)
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            src = archive.extractfile(member)
            if src is None:
                skipped_unsupported += 1
                continue

            try:
                with open(target, "xb") as dst:
                    shutil.copyfileobj(src, dst)
            except FileExistsError:
                skipped_existing += 1
                continue

            os.chmod(target, member.mode & 0o777)
            os.utime(target, (member.mtime, member.mtime))
            extracted += 1

    return extracted, skipped_existing, skipped_unsupported


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract only new files from a .tar.gz stream")
    parser.add_argument("--dest", required=True, type=Path)
    args = parser.parse_args()

    extracted, skipped_existing, skipped_unsupported = extract_new(args.dest)
    print(
        "extract-new: "
        f"extracted={extracted} "
        f"skipped_existing={skipped_existing} "
        f"skipped_unsupported={skipped_unsupported}"
    )


if __name__ == "__main__":
    main()
