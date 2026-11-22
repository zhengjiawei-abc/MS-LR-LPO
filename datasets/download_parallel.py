"""Download a large public file with resumable HTTP range chunks."""
from __future__ import annotations

import argparse
import hashlib
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path


USER_AGENT = "Mozilla/5.0 PDEBench-downloader"


def request(url: str, start: int | None = None, end: int | None = None):
    headers = {"User-Agent": USER_AGENT}
    if start is not None:
        headers["Range"] = f"bytes={start}-{end}"
    return urllib.request.Request(url, headers=headers)


def content_length(url: str) -> int:
    with urllib.request.urlopen(request(url, 0, 0), timeout=60) as response:
        content_range = response.headers.get("Content-Range", "")
        if "/" in content_range:
            return int(content_range.rsplit("/", 1)[1])
        length = response.headers.get("Content-Length")
        if length and int(length) > 1:
            return int(length)
    raise RuntimeError("Could not determine remote file size")


def download_chunk(
    url: str,
    path: Path,
    start: int,
    end: int,
    report,
) -> None:
    expected = end - start + 1
    if path.exists() and path.stat().st_size > expected:
        with path.open("r+b") as handle:
            handle.truncate(expected)
    for attempt in range(1, 11):
        offset = path.stat().st_size if path.exists() else 0
        if offset >= expected:
            report(path, expected - offset)
            return
        try:
            with urllib.request.urlopen(
                request(url, start + offset, end),
                timeout=120,
            ) as response:
                if response.status != 206:
                    raise RuntimeError(f"range request returned HTTP {response.status}")
                with path.open("ab") as handle:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        handle.write(block)
                        report(path, len(block))
            if path.stat().st_size != expected:
                raise RuntimeError(
                    f"chunk size {path.stat().st_size} != expected {expected}"
                )
            return
        except Exception as error:
            if attempt == 10:
                raise
            print(f"retry {attempt} {path.name}: {error}", flush=True)
            time.sleep(min(2 ** attempt, 30))


def md5(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--md5", required=True)
    parser.add_argument("--connections", type=int, default=32)
    parser.add_argument("--chunk-mb", type=int, default=256)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total = content_length(args.url)
    chunk_bytes = args.chunk_mb * 1024 * 1024
    chunks = []
    for index, start in enumerate(range(0, total, chunk_bytes)):
        end = min(start + chunk_bytes, total) - 1
        chunks.append((index, start, end, args.output.with_suffix(
            args.output.suffix + f".part{index:03d}"
        )))
    print(
        f"download {total / 1e9:.3f} GB in {len(chunks)} chunks, "
        f"{args.connections} connections",
        flush=True,
    )

    lock = threading.Lock()
    transferred = 0

    def report(path: Path, amount: int) -> None:
        nonlocal transferred
        with lock:
            transferred += max(0, amount)

    started = time.time()
    with ThreadPoolExecutor(max_workers=args.connections) as pool:
        futures = {
            pool.submit(download_chunk, args.url, path, start, end, report):
            (index, start, end, path)
            for index, start, end, path in chunks
            if not (
                path.exists()
                and path.stat().st_size == end - start + 1
            )
        }
        while futures:
            done, pending = wait(
                futures,
                timeout=15,
                return_when=FIRST_COMPLETED,
            )
            for future in done:
                future.result()
                futures.pop(future)
            on_disk = sum(
                path.stat().st_size
                for _, _, _, path in chunks
                if path.exists()
            )
            elapsed = max(time.time() - started, 1e-9)
            print(
                f"progress {on_disk / 1e9:.3f}/{total / 1e9:.3f} GB "
                f"({on_disk / total:.1%}) "
                f"speed {on_disk / 1e6 / elapsed:.2f} MB/s",
                flush=True,
            )

    with args.output.open("wb") as output:
        for _, _, _, path in chunks:
            with path.open("rb") as source:
                while True:
                    block = source.read(8 * 1024 * 1024)
                    if not block:
                        break
                    output.write(block)

    actual = md5(args.output)
    print(f"md5 {actual}", flush=True)
    if actual.lower() != args.md5.lower():
        raise RuntimeError(f"MD5 mismatch: expected {args.md5}, got {actual}")
    for _, _, _, path in chunks:
        path.unlink(missing_ok=True)
    print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
