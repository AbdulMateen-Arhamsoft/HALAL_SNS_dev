from __future__ import annotations

import re
from functools import lru_cache
from typing import Final, Literal, NamedTuple

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DeviceRequest = Literal["auto", "cuda", "cpu"]
_CATEGORY_CODE: Final = re.compile(r"G-0[1-7]")


class ModelPin(NamedTuple):
    """A Hugging Face repo at one commit. The worker only loads from disk; model_download.py downloads."""

    repo: str
    revision: str


# Pinned model revisions. Change one, run `python model_download.py pull`, restart the worker.
MODEL_PINS: Final = {
    "text": ModelPin("Qwen/Qwen3Guard-Gen-0.6B", "fada3b2f655b89601929198343c94cd2f64d93cc"),
    "image": ModelPin("google/siglip2-base-patch16-384", "f775b65a79762255128c981547af89addcfe0f88"),
}


class Settings(BaseSettings):
    """Every setting, read from the environment or .env. The only settings class."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Database ---
    database_url: str = Field(description="Postgres DSN, psycopg format. Required: there is no default")
    db_pool_max_size: int = Field(default=10, ge=1, description="Maximum open DB connections per process")

    # --- Models (worker only) ---
    text_model_dir: str = Field(default="models/qwen3guard-gen-0.6b", description="Qwen3Guard weights")
    image_model_dir: str = Field(default="models/siglip2-base-patch16-384", description="SigLIP2 weights")
    image_head_dir: str = Field(default="models/siglip2-head", description="Image head built by model_download.py build-head")
    ocr_model_dir: str = Field(default="models/rapidocr", description="RapidOCR PP-OCRv5 det + rec files")
    device: DeviceRequest = Field(
        default="auto",
        description="auto: GPU if usable, else CPU (degraded). cuda: GPU or refuse to start. cpu: CPU only",
    )
    ocr_threads: int = Field(default=2, ge=1, description="ONNX Runtime threads for OCR (always CPU)")
    model_max_batch: int = Field(default=8, ge=1, description="Largest batch one text-model call takes")

    # --- Worker ---
    fast_workers: int = Field(default=4, ge=1, description="Concurrent jobs")
    poll_s: float = Field(
        default=5.0,
        gt=0,
        description="Fallback poll. New jobs wake the worker by notification; this picks up retries",
    )
    lease_s: int = Field(default=20, ge=5, description="Seconds a claimed job stays with its worker")
    max_attempts: int = Field(default=5, ge=1, description="Tries before a job fails")

    # --- Callbacks ---
    callback_timeout_s: float = Field(default=5.0, gt=0, description="HTTP timeout for one callback")
    callback_max_attempts: int = Field(default=10, ge=1, description="Callback tries before the row is dead")

    # --- Images ---
    image_allowed_hosts: str = Field(default="", description="Comma-separated hostnames. Empty refuses every URL")
    allow_http_localhost: bool = Field(default=False, description="Allow plain http:// to localhost, 127.0.0.1 and host.docker.internal (dev only, local testing)")
    image_allowed_root: str = Field(
        default="",
        description="Folder local image paths and raw uploads (POST /v1/moderations/uploads) may come from. "
        "Empty refuses local paths and disables uploads. In Docker this must be /app/uploads, the volume "
        "docker-compose shares between api and worker",
    )
    image_max_bytes: int = Field(default=10 * 1024 * 1024, ge=1024, description="Largest image accepted")
    fetch_timeout_s: float = Field(default=5.0, gt=0, description="HTTP timeout for one image download")

    # --- Video (D2) ---
    # Limits from plan section 9. Placeholders are marked; they need confirming before production.
    video_max_duration_s: int = Field(default=300, ge=1, le=3600, description="Longest video accepted; longer is held as media_too_long")
    video_sample_fps: float = Field(default=1.0, gt=0, le=30, description="Frames kept per second of video (sampling step)")
    video_max_frames: int = Field(default=240, ge=1, le=2000, description="Most frames checked per video; scene-change frames are kept first")
    video_frame_side: int = Field(default=384, ge=32, le=1024, description="Side length of each frame given to the models, in pixels")
    video_scene_threshold: float = Field(default=30.0, gt=0, le=255, description="Mean grey difference that counts as a scene change (assumption, tune later)")
    video_dedup_distance: int = Field(default=5, ge=0, le=64, description="Hash distance (of 64 bits) at or below which a frame is a duplicate (assumption, tune later)")
    video_max_bytes: int = Field(default=100 * 1024 * 1024, ge=1024, description="PLACEHOLDER: largest video file accepted")
    video_max_pixels: int = Field(default=1920 * 1080, ge=1, description="PLACEHOLDER: largest frame size (width x height) accepted")
    video_fetch_timeout_s: float = Field(default=60.0, gt=0, description="HTTP timeout for one video download (whole download budget)")
    video_workers: int = Field(default=1, ge=1, le=1, description="Video jobs at once. Fixed at 1 by decision 11")
    video_max_pending_per_user: int = Field(default=3, ge=1, description="PLACEHOLDER: videos one author may have waiting before 429 rate_limited (only for posts that carry an author_id)")
    video_retention_days: int = Field(default=30, ge=1, description="PLACEHOLDER: days flagged-frame evidence is kept")
    video_temp_dir: str = Field(default="", description="Folder for temporary video copies. Empty uses the system temp folder")
    video_evidence_root: str = Field(default="evidence", description="Folder where flagged frames are saved as JPEG")
    video_hard_block_categories: str = Field(
        default="G-01,G-03,G-05",
        description="Categories where one flagged frame is enough (maximum). Others use the mean of the 3 worst frames",
    )
    video_text_unreadable_share: float = Field(
        default=0.5, gt=0, le=1, description="Hold the video when more than this share of text-bearing frames cannot be read"
    )

    # --- API ---
    api_max_videos: int = Field(default=1, ge=0, le=10, description="Videos per post. 1 to start (plan section 7)")
    api_max_body_bytes: int = Field(default=128 * 1024, ge=1024, description="Largest request body accepted, checked before it is read")
    api_max_images: int = Field(default=10, ge=0, description="Maximum images per request")
    api_max_text_chars: int = Field(default=10_000, ge=1, description="Maximum caption length")
    api_max_url_chars: int = Field(default=2048, ge=1, description="Maximum image URL length")
    api_max_callback_url_chars: int = Field(
        default=500, ge=1, le=2048, description="Maximum callback URL length (the database column allows 2048)"
    )
    api_wait_s: float = Field(
        default=10.0, ge=0, description="How long POST /v1/moderations waits for the decision before answering 202. 0: never wait"
    )
    api_wait_poll_s: float = Field(default=0.1, gt=0, description="How often a waiting request checks its job")
    api_keys: str = Field(
        default="",
        description="Comma-separated name:key pairs for the X-API-Key header, e.g. 'alice:abc123,bob:def456'. "
        "Empty disables auth entirely (dev only; never run exposed to the internet this way)",
    )

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO", description="Root log level")

    @property
    def allowed_hosts(self) -> frozenset[str]:
        return frozenset(h.strip().lower() for h in self.image_allowed_hosts.split(",") if h.strip())

    @property
    def video_hard_block_codes(self) -> frozenset[str]:
        """VIDEO_HARD_BLOCK_CATEGORIES parsed and checked. An unknown code is an error, never ignored."""
        codes = frozenset(c.strip().upper() for c in self.video_hard_block_categories.split(",") if c.strip())
        unknown = sorted(c for c in codes if not _CATEGORY_CODE.fullmatch(c))
        if unknown:
            raise ValueError(f"VIDEO_HARD_BLOCK_CATEGORIES has unknown codes {unknown}; use G-01 to G-07")
        return codes

    @property
    def api_key_names(self) -> dict[str, str]:
        """Key -> teammate name, parsed from `api_keys`. Empty means auth is disabled."""
        result = {}
        for pair in self.api_keys.split(","):
            if ":" in (pair := pair.strip()):
                name, key = pair.split(":", 1)
                result[key] = name
        return result


@lru_cache
def get_settings() -> Settings:
    return Settings()
