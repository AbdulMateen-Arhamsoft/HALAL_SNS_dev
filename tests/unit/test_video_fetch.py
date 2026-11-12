"""D2 video download: the host rule, the size and time limits, the local upload path, and the temporary copy."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from moderation.config import Settings
from moderation.errors import (
    MEDIA_FORBIDDEN,
    MEDIA_NOT_FOUND,
    MEDIA_TOO_LARGE,
    PermanentError,
    RetryableError,
)
from moderation.video.fetch import VideoFetcher, video_extension, write_temp_copy

DB = "postgresql://u:p@localhost:5432/x"
HOST = "cdn.example"
BODY = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 4096


def cfg(tmp_path: Path, **overrides: object) -> Settings:
    base: dict[str, object] = {
        "image_allowed_hosts": HOST,
        "image_allowed_root": str(tmp_path / "root"),
        "video_temp_dir": str(tmp_path / "tmp"),
    }
    base.update(overrides)
    return Settings(_env_file=None, database_url=DB, **base)  # type: ignore[call-arg]


def fetcher_for(settings: Settings, handler: object) -> VideoFetcher:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False, trust_env=False)  # type: ignore[arg-type]
    return VideoFetcher(settings, client)


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "tmp"
    folder.mkdir()
    return folder


def ok(_: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=BODY)


async def test_https_download_is_a_temporary_copy_deleted_on_exit(tmp_path: Path, temp_dir: Path) -> None:
    fetcher = fetcher_for(cfg(tmp_path), ok)
    async with fetcher.download(f"https://{HOST}/v.mp4") as path:
        assert path.read_bytes() == BODY
        assert path.parent == temp_dir
    assert not path.exists()


async def test_temporary_copy_is_deleted_even_when_the_check_fails(tmp_path: Path, temp_dir: Path) -> None:
    fetcher = fetcher_for(cfg(tmp_path), ok)
    seen: list[Path] = []
    with pytest.raises(RuntimeError, match="check failed"):
        async with fetcher.download(f"https://{HOST}/v.mp4") as path:
            seen.append(path)
            raise RuntimeError("check failed")
    assert seen and not seen[0].exists()
    assert list(temp_dir.iterdir()) == []


async def test_host_not_on_the_allowlist_is_refused_before_any_download(tmp_path: Path, temp_dir: Path) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, content=BODY)

    with pytest.raises(PermanentError) as info:
        async with fetcher_for(cfg(tmp_path), handler).download("https://evil.example/v.mp4"):
            pass
    assert info.value.code == MEDIA_FORBIDDEN
    assert calls == []


async def test_plain_http_is_refused(tmp_path: Path) -> None:
    with pytest.raises(PermanentError) as info:
        async with fetcher_for(cfg(tmp_path), ok).download(f"http://{HOST}/v.mp4"):
            pass
    assert info.value.code == MEDIA_FORBIDDEN


async def test_declared_size_over_the_limit_is_too_large(tmp_path: Path, temp_dir: Path) -> None:
    def big(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=BODY, headers={"content-length": str(10**9)})

    with pytest.raises(PermanentError) as info:
        async with fetcher_for(cfg(tmp_path, video_max_bytes=1024), big).download(f"https://{HOST}/v.mp4"):
            pass
    assert info.value.code == MEDIA_TOO_LARGE
    assert list(temp_dir.iterdir()) == []  # nothing left behind


async def test_body_over_the_limit_is_too_large_and_the_copy_is_removed(tmp_path: Path, temp_dir: Path) -> None:
    with pytest.raises(PermanentError) as info:
        async with fetcher_for(cfg(tmp_path, video_max_bytes=1024), ok).download(f"https://{HOST}/v.mp4"):
            pass
    assert info.value.code == MEDIA_TOO_LARGE
    assert list(temp_dir.iterdir()) == []


async def test_not_found_is_retryable_as_media_not_found(tmp_path: Path) -> None:
    def missing(_: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    with pytest.raises(RetryableError) as info:
        async with fetcher_for(cfg(tmp_path), missing).download(f"https://{HOST}/v.mp4"):
            pass
    assert info.value.code == MEDIA_NOT_FOUND


async def test_redirect_is_refused_as_forbidden(tmp_path: Path) -> None:
    def redirect(_: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "https://evil.example/x.mp4"})

    with pytest.raises(PermanentError) as info:
        async with fetcher_for(cfg(tmp_path), redirect).download(f"https://{HOST}/v.mp4"):
            pass
    assert info.value.code == MEDIA_FORBIDDEN


async def test_an_upload_is_read_in_place_and_never_deleted(tmp_path: Path) -> None:
    settings = cfg(tmp_path)
    upload = Path(settings.image_allowed_root) / "uploads" / "abc.mp4"
    upload.parent.mkdir(parents=True)
    upload.write_bytes(BODY)
    async with fetcher_for(settings, ok).download("uploads/abc.mp4") as path:
        assert path == upload.resolve()
    assert upload.exists()  # the upload itself is kept


async def test_a_path_outside_the_upload_root_is_refused(tmp_path: Path) -> None:
    settings = cfg(tmp_path)
    Path(settings.image_allowed_root).mkdir(parents=True)
    (tmp_path / "secret.mp4").write_bytes(BODY)
    with pytest.raises(PermanentError) as info:
        async with fetcher_for(settings, ok).download("../secret.mp4"):
            pass
    assert info.value.code == MEDIA_FORBIDDEN


async def test_local_paths_are_refused_when_the_root_is_empty(tmp_path: Path) -> None:
    with pytest.raises(PermanentError) as info:
        async with fetcher_for(cfg(tmp_path, image_allowed_root=""), ok).download("uploads/abc.mp4"):
            pass
    assert info.value.code == MEDIA_FORBIDDEN


async def test_a_url_with_another_scheme_is_refused(tmp_path: Path) -> None:
    with pytest.raises(PermanentError) as info:
        async with fetcher_for(cfg(tmp_path), ok).download("file:///etc/passwd"):
            pass
    assert info.value.code == MEDIA_FORBIDDEN


async def test_missing_upload_is_retryable(tmp_path: Path) -> None:
    settings = cfg(tmp_path)
    Path(settings.image_allowed_root).mkdir(parents=True)
    with pytest.raises(RetryableError) as info:
        async with fetcher_for(settings, ok).download("uploads/not-yet.mp4"):
            pass
    assert info.value.code == MEDIA_NOT_FOUND


@pytest.mark.parametrize(
    "data,expected",
    [
        (b"\x00\x00\x00\x18ftypisom", ".mp4"),
        (b"\x00\x00\x00\x14moov", ".mp4"),
        (b"\x1a\x45\xdf\xa3\x01", ".mkv"),
        (b"GIF89a", None),
        (b"hello world", None),
        (b"", None),
    ],
)
def test_video_extension_by_content(data: bytes, expected: str | None) -> None:
    assert video_extension(data) == expected


def test_write_temp_copy_writes_the_bytes_into_the_temp_folder(tmp_path: Path, temp_dir: Path) -> None:
    path = write_temp_copy(b"GIF89a-animated", cfg(tmp_path))
    try:
        assert path.read_bytes() == b"GIF89a-animated"
        assert path.parent == temp_dir
    finally:
        path.unlink(missing_ok=True)
