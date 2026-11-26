"""Read HDF5 files through HTTP byte ranges without downloading the full file."""
from __future__ import annotations

import io
import subprocess
import time
from collections import OrderedDict


class HTTPRangeFile(io.RawIOBase):
    def __init__(
        self,
        url: str,
        block_size: int = 4 * 1024 * 1024,
        cache_blocks: int = 16,
    ) -> None:
        super().__init__()
        self.url = url
        self.block_size = block_size
        self.cache_blocks = cache_blocks
        self.cache: OrderedDict[int, bytes] = OrderedDict()
        self.position = 0
        self.size = self._content_size()

    def _fetch(self, start: int, end: int) -> bytes:
        for attempt in range(1, 6):
            result = subprocess.run(
                [
                    "curl.exe",
                    "-L",
                    "-sS",
                    "--fail",
                    "--retry",
                    "3",
                    "--retry-all-errors",
                    "-r",
                    f"{start}-{end}",
                    self.url,
                ],
                capture_output=True,
                timeout=180,
                check=False,
            )
            if result.returncode == 0:
                return result.stdout
            if attempt == 5:
                raise RuntimeError(
                    result.stderr.decode("utf-8", errors="replace")
                )
            time.sleep(2**attempt)
        raise RuntimeError("unreachable")

    def _content_size(self) -> int:
        result = subprocess.run(
            [
                "curl.exe",
                "-L",
                "-sS",
                "--fail",
                "-r",
                "0-0",
                "-D",
                "-",
                "-o",
                "NUL",
                self.url,
            ],
            capture_output=True,
            timeout=180,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
        headers = result.stdout.decode("latin-1", errors="replace")
        for line in headers.splitlines():
            if line.lower().startswith("content-range:"):
                return int(line.rsplit("/", 1)[1])
        raise RuntimeError("server did not return Content-Range")

    def _block(self, index: int) -> bytes:
        cached = self.cache.get(index)
        if cached is not None:
            self.cache.move_to_end(index)
            return cached
        start = index * self.block_size
        end = min(start + self.block_size, self.size) - 1
        data = self._fetch(start, end)
        self.cache[index] = data
        if len(self.cache) > self.cache_blocks:
            self.cache.popitem(last=False)
        return data

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        self.position = max(0, position)
        return self.position

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self.size - self.position
        stop = min(self.position + size, self.size)
        if stop <= self.position:
            return b""
        parts = []
        while self.position < stop:
            index = self.position // self.block_size
            block = self._block(index)
            offset = self.position - index * self.block_size
            take = min(stop - self.position, len(block) - offset)
            parts.append(block[offset : offset + take])
            self.position += take
        return b"".join(parts)

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def open_http_hdf5(url: str, block_size: int = 4 * 1024 * 1024):
    import h5py

    return h5py.File(HTTPRangeFile(url, block_size=block_size), "r")
