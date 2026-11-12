"""Video download (plan Part B step 1). A temporary copy is always deleted, even when the check fails.

Two sources, as for images:
  - https on an allowed host (the D1 rule, moderation.fetch.check_url): streamed to a temporary copy with a
    size cap and a whole-download time budget (VIDEO_FETCH_TIMEOUT_S). The copy is deleted on exit.
  - a path under IMAGE_ALLOWED_ROOT (a raw upload): read in place. It is the upload itself, so it is never deleted.

`download` is an async context manager: `async with fetcher.download(url) as path:`. The `finally` step
that deletes the copy runs whether the check worked, failed or was cancelled.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import BinaryIO, Final
from urllib.parse import urlsplit

import httpx

from moderation.config import Settings
from moderation.errors import (
    MEDIA_FORBIDDEN,
    MEDIA_NOT_FOUND,
    MEDIA_TOO_LARGE,
    PermanentError,
    RetryableError,
    translate,
)
from moderation.fetch import check_status, check_url

log = logging.getLogger("moderation.video.fetch")
_CHUNK_BYTES: Final = 64 * 1024


class VideoFetcher:
    """The caller owns `client` (follow_redirects=False, trust_env=False)."""

    def __init__(self, cfg: Settings, client: httpx.AsyncClient) -> None:
        self._cfg = cfg
        self._client = client
        self._root = Path(cfg.image_allowed_root).resolve() if cfg.image_allowed_root.strip() else None

    @asynccontextmanager
    async def download(self, url: str) -> AsyncIterator[Path]:
        if url.startswith(("http://", "https://")):
            path = await self._download_to_temp(url)
            try:
                yield path
            finally:
                _delete_copy(path)
        else:
            yield self._local_path(url)  # the upload itself: nothing to clean up

    async def _download_to_temp(self, url: str) -> Path:
        check_url(url, self._cfg.allowed_hosts, self._cfg.allow_http_localhost)
        handle, name = tempfile.mkstemp(prefix="video-", suffix=".bin", dir=_temp_folder(self._cfg))
        path = Path(name)
        try:
            with os.fdopen(handle, "wb") as out:
                await self._stream_into(url, out)
        except BaseException:
            _delete_copy(path)
            raise
        return path

    async def _stream_into(self, url: str, out: BinaryIO) -> None:
        size = 0
        with translate(RetryableError, MEDIA_NOT_FOUND, httpx.TransportError, TimeoutError):
            # The timeout is the budget for the whole download, so a slow server cannot hold a job.
            async with asyncio.timeout(self._cfg.video_fetch_timeout_s), self._client.stream("GET", url) as response:
                check_status(response.status_code)
                declared = response.headers.get("content-length")
                if declared is not None and declared.isdigit() and int(declared) > self._cfg.video_max_bytes:
                    raise PermanentError(MEDIA_TOO_LARGE, f"declared {declared} bytes exceeds {self._cfg.video_max_bytes}")
                async for chunk in response.aiter_bytes(_CHUNK_BYTES):
                    size += len(chunk)
                    if size > self._cfg.video_max_bytes:
                        raise PermanentError(MEDIA_TOO_LARGE, f"body exceeds {self._cfg.video_max_bytes} bytes")
                    await asyncio.to_thread(out.write, chunk)

    def _local_path(self, url: str) -> Path:
        """`url` is a path relative to IMAGE_ALLOWED_ROOT (an upload). Anything outside that folder is refused."""
        if self._root is None:
            raise PermanentError(MEDIA_FORBIDDEN, "local video paths are disabled (IMAGE_ALLOWED_ROOT is empty)")
        if urlsplit(url).scheme:
            raise PermanentError(MEDIA_FORBIDDEN, "only https links or uploads are accepted")
        path = (self._root / url).resolve()
        if not path.is_relative_to(self._root):
            raise PermanentError(MEDIA_FORBIDDEN, "path is outside IMAGE_ALLOWED_ROOT")
        with translate(RetryableError, MEDIA_NOT_FOUND, FileNotFoundError):
            if path.stat().st_size > self._cfg.video_max_bytes:
                raise PermanentError(MEDIA_TOO_LARGE, f"file exceeds {self._cfg.video_max_bytes} bytes")
        return path


def video_extension(data: bytes) -> str | None:
    """File extension by content for an uploaded video: MP4/MOV family or EBML (WebM, Matroska). None if neither.
    Lives here, not in probe.py, so the API process can use it without loading PyAV."""
    if data[4:8] in (b"ftyp", b"moov", b"mdat", b"wide", b"free"):
        return ".mp4"
    if data[:4] == b"\x1a\x45\xdf\xa3":
        return ".mkv"
    return None


def write_temp_copy(data: bytes, cfg: Settings) -> Path:
    """Write bytes to a temporary file (used for an animated image, which is checked as a video). Caller deletes it."""
    handle, name = tempfile.mkstemp(prefix="animated-", suffix=".bin", dir=_temp_folder(cfg))
    with os.fdopen(handle, "wb") as out:
        out.write(data)
    return Path(name)


def _temp_folder(cfg: Settings) -> str | None:
    """VIDEO_TEMP_DIR, created if missing (a fresh install has no such folder). Empty means the system temp folder."""
    if not cfg.video_temp_dir:
        return None
    Path(cfg.video_temp_dir).mkdir(parents=True, exist_ok=True)
    return cfg.video_temp_dir


def _delete_copy(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        log.exception("could not delete temporary video copy", extra={"path_name": path.name})
