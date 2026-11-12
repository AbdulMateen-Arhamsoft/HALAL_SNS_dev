"""Shared vocabulary: labels, categories, the request, model outputs and results."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class Label(StrEnum):
    ALLOW = "allow"
    HOLD = "hold"
    BLOCK = "block"


class ItemSource(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    IMAGE_TEXT = "image_text"
    VIDEO = "video"  # D2: the combined frame scores of one video
    VIDEO_TEXT = "video_text"  # D2: on-screen text of a video's frames


# Harm categories G-01 to G-07. D3 adds the halal codes H-01 to H-06 here.
CATEGORIES: dict[str, str] = {
    "G-01": "Violence and gore",
    "G-02": "Weapons",
    "G-03": "Sexual content and nudity",
    "G-04": "Hate and harassment",
    "G-05": "Self-harm",
    "G-06": "Drugs and illegal goods",
    "G-07": "Crime and fraud",
}
UNCLASSIFIED = "unclassified"

TextTier = Literal["Safe", "Controversial", "Unsafe"]


@dataclass(frozen=True)
class TextScore:
    tier: TextTier
    category: str | None
    score: float


@dataclass(frozen=True)
class ImageScore:
    categories: dict[str, float]  # one score per category code
    embedding: tuple[float, ...]  # L2-normalised SigLIP2 vision embedding; D5 reuses it for the feed


@dataclass(frozen=True)
class OcrResult:
    text: str  # recognised lines in reading order, joined with newlines; empty when nothing was read
    mean_confidence: float | None  # None when no text boxes were found
    box_count: int  # text boxes found, including lines that could not be read


@dataclass(frozen=True)
class ItemResult:
    source: ItemSource
    media_id: str | None
    label: Label
    category: str | None
    score: float | None
    note: str | None = None  # why a held item without a category was held


@dataclass(frozen=True)
class RejectedImage:
    """An image that could not be checked. The backend gets it in the callback so the user can re-upload."""

    media_id: str
    error_code: str  # media_too_large, media_unsupported, media_undecodable or media_forbidden
    detail: str


@dataclass(frozen=True)
class RejectedVideo:
    """A video that could not be checked (D2). The backend gets it in the callback; the video is held."""

    media_id: str
    error_code: str  # media_too_large, media_too_long, media_unsupported, media_undecodable, media_forbidden, media_not_found
    detail: str


@dataclass(frozen=True)
class VideoFrameRecord:
    """One kept frame of a video, as stored in video_frame. `jpeg` is only set for flagged frames (evidence)."""

    media_id: str
    frame_no: int
    time_ms: int
    source: str  # "sample" or "scene"
    scores: dict[str, float]
    flagged: bool
    jpeg: bytes = b""
    evidence_path: str | None = None  # set by the worker once the evidence file is written


@dataclass(frozen=True)
class Decision:
    label: Label
    category: str | None
    reason: str
    items: tuple[ItemResult, ...] = ()
    policy_version: str = ""
    # Stored with the decision, never sent in the callback.
    embeddings: dict[str, tuple[float, ...]] = field(default_factory=dict)  # media_id -> image embedding
    model_versions: dict[str, str] = field(default_factory=dict)  # model role -> version
    rejected_images: tuple[RejectedImage, ...] = ()  # sent in the callback; each is also a held item
    rejected_videos: tuple[RejectedVideo, ...] = ()  # D2: sent in the callback; each is also a held item
    video_frames: tuple[VideoFrameRecord, ...] = ()  # D2: stored in video_frame; never sent in the callback


ShortId = Annotated[str, Field(min_length=1, max_length=128)]


class ImageRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    media_id: ShortId = Field(description="Your id for this image, unique within the request")
    url: str = Field(
        min_length=1,
        description="https URL on an allowed host, or the url returned by POST /v1/moderations/uploads",
    )


class VideoRef(BaseModel):
    """One video of a post (D2). `url` is https on an allowed host, or a path returned by POST /v1/moderations/uploads/video."""

    model_config = ConfigDict(extra="forbid")

    media_id: ShortId = Field(description="Your id for this video, unique within the request")
    url: str = Field(min_length=1, description="https URL on an allowed host, or the url returned by the video upload")


class ModerationRequest(BaseModel):
    """Body of POST /v1/moderations. Stored as the job payload and read back by the worker."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "request_id": "6f1c0c8e-3d52-4d0e-8c56-0a1f0c3b9e11",
                "surface": "post",
                "content_id": "post_123",
                "content_version": 1,
                "author_id": "u_456",
                "text": "Family picnic at the lake",
                "images": [{"media_id": "m1", "url": "https://media.example.com/m1.jpg"}],
                "callback_url": "https://backend.example.com/moderation-result",
            }
        },
    )

    request_id: ShortId = Field(description="Your id for this call, for tracing")
    surface: str = Field(min_length=1, max_length=32, description="What is checked. D1 accepts 'post' only")
    content_id: ShortId = Field(description="Id of the post")
    content_version: int = Field(ge=1, description="Starts at 1. An edit sends a higher number and is checked again")
    author_id: Annotated[str | None, Field(max_length=128)] = Field(default=None, description="Id of the author")
    text: str | None = Field(default=None, description="Caption. At least one of text or images is required")
    images: list[ImageRef] = Field(default_factory=list, description="Images to check (a limit applies, API_MAX_IMAGES)")
    videos: list[VideoRef] = Field(default_factory=list, description="Videos to check (D2; a limit applies, API_MAX_VIDEOS)")
    callback_url: HttpUrl | None = Field(
        default=None, description="Where the decision is POSTed. Leave out to poll GET /v1/moderations/{job_id}"
    )

    @model_validator(mode="after")
    def _needs_content(self) -> ModerationRequest:
        if not (self.text or self.images or self.videos):
            raise ValueError("at least one of text, images or videos is required")
        ids = [image.media_id for image in self.images] + [video.media_id for video in self.videos]
        if len(ids) != len(set(ids)):
            raise ValueError("media_id must be unique within the request")
        if self.videos and not self.author_id:
            raise ValueError("author_id is required when the post has videos (it sets the per-user video limit)")
        return self
