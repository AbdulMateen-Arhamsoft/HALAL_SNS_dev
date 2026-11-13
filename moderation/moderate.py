"""Check one request: caption and images through the models, then worst-wins across all pieces."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from moderation.config import Settings
from moderation.domain import (
    CATEGORIES,
    UNCLASSIFIED,
    Decision,
    ImageRef,
    ItemResult,
    ItemSource,
    Label,
    ModerationRequest,
    OcrResult,
    RejectedImage,
    RejectedVideo,
    TextScore,
    VideoFrameRecord,
    VideoRef,
)
from moderation.errors import (
    MEDIA_FORBIDDEN,
    MEDIA_NOT_FOUND,
    MEDIA_TOO_LARGE,
    MEDIA_TOO_LONG,
    MEDIA_UNDECODABLE,
    MEDIA_UNSUPPORTED,
    PermanentError,
)
from moderation.fetch import is_animated
from moderation.policy import Policy, worst_item
from moderation.text_clean import Segment, clean_text, has_template_text, split_segments
from moderation.video.fetch import VideoFetcher, write_temp_copy
from moderation.video.frames import select_frames
from moderation.video.probe import probe

if TYPE_CHECKING:
    from moderation.video.check import VideoChecker

ANIMATED_REASON = "Animated image: not checked, held"  # D1 behaviour, kept when no video pipeline is configured
TEMPLATE_NOTE = "Text imitates the moderation prompt"

# What the backend is told for an image we could not check (error_code -> detail).
REJECT_DETAILS = {
    MEDIA_TOO_LARGE: "file is too large",
    MEDIA_UNSUPPORTED: "unsupported file type: only JPEG, PNG, WebP and GIF are accepted",
    MEDIA_UNDECODABLE: "file is unreadable or corrupt",
    MEDIA_FORBIDDEN: "image could not be accessed",
}
# What the backend is told for a video we could not check (D2).
REJECT_VIDEO_DETAILS = {
    MEDIA_TOO_LARGE: "video file or frame size is too large",
    MEDIA_TOO_LONG: "video is longer than the allowed duration",
    MEDIA_UNSUPPORTED: "unsupported video format: only MP4, MOV, WebM and MKV are accepted",
    MEDIA_UNDECODABLE: "video file is unreadable or corrupt",
    MEDIA_FORBIDDEN: "video could not be accessed",
    MEDIA_NOT_FOUND: "video is not available yet",
}


class VideoGate:
    """Which videos may use the CPU and which may wait for the GPU, for the whole worker process.

    `cpu`: videos being prepared at once (download, decode, pick frames, read text).
    `waiting`: prepared videos that hold their frames while the GPU scores earlier ones. When it is full a
    prepared video keeps its CPU slot, so no further video is started: that is the memory limit.
    The counters show how much is held while videos wait (logged, and read by the tests)."""

    def __init__(self, parallel: int, max_waiting: int) -> None:
        self.cpu = asyncio.Semaphore(parallel)
        self.waiting = asyncio.Semaphore(max_waiting)
        self.waiting_videos = 0
        self.waiting_bytes = 0
        self.peak_waiting_videos = 0
        self.peak_waiting_bytes = 0

    def enter_waiting(self, nbytes: int) -> None:
        self.waiting_videos += 1
        self.waiting_bytes += nbytes
        self.peak_waiting_videos = max(self.peak_waiting_videos, self.waiting_videos)
        self.peak_waiting_bytes = max(self.peak_waiting_bytes, self.waiting_bytes)

    def leave_waiting(self, nbytes: int) -> None:
        self.waiting_videos -= 1
        self.waiting_bytes -= nbytes


@dataclass(frozen=True)
class VideoPipeline:
    """Everything a worker needs to check a video. None on a Moderator means videos are not checked (D1)."""

    cfg: Settings
    fetcher: VideoFetcher
    checker: VideoChecker
    gate: VideoGate = field(default=None)  # type: ignore[assignment]  # built from cfg in __post_init__

    def __post_init__(self) -> None:
        if self.gate is None:
            object.__setattr__(self, "gate", VideoGate(self.cfg.video_parallel, self.cfg.video_max_waiting))


@dataclass(frozen=True)
class PreparedVideo:
    """A video after its CPU steps: the frames to score and the text already read from them."""

    selection: Any  # moderation.video.frames.Selection
    ocr: list[OcrResult]

    @property
    def nbytes(self) -> int:
        return sum(len(frame.jpeg) for frame in self.selection.frames)


@dataclass
class VideoSink:
    """What the video checks produced for one request, collected next to the image results."""

    items: list[ItemResult] = field(default_factory=list)
    rejected: list[RejectedVideo] = field(default_factory=list)
    frames: list[VideoFrameRecord] = field(default_factory=list)


async def score_each[R](call: Callable[[list[bytes]], Awaitable[Sequence[R]]], blobs: list[bytes]) -> list[R | PermanentError]:
    """One batch call. If an image in it cannot be decoded, retry one by one so only that image is lost."""
    if not blobs:
        return []
    try:
        return list(await call(blobs))
    except PermanentError as exc:
        if exc.code != MEDIA_UNDECODABLE:
            raise
        if len(blobs) == 1:
            return [exc]
    return [(await score_each(call, [blob]))[0] for blob in blobs]


class Moderator:
    def __init__(
        self,
        *,
        text: Any,
        image: Any,
        ocr: Any,
        fetcher: Any,
        policy: Policy,
        model_versions: Mapping[str, str] | None = None,
        video: VideoPipeline | None = None,
    ) -> None:
        self._text = text
        self._image = image
        self._ocr = ocr
        self._fetcher = fetcher
        self._policy = policy
        self._model_versions = dict(model_versions or {})
        self._video = video

    def attach_video(self, pipeline: VideoPipeline) -> None:
        """Turn on video checks. The checker needs this moderator's text judge, so it is attached after both exist."""
        self._video = pipeline

    async def moderate(self, request: ModerationRequest) -> Decision:
        """Fetch errors propagate unchanged: PermanentError fails the job, RetryableError retries it."""
        if request.videos and self._video is None:
            # A worker without the video pipeline must not answer a post with videos: the job retries, then fails closed.
            raise RuntimeError("video checks are not configured on this worker")
        sink = VideoSink()
        # The text model runs in its own thread, so the caption is scored while the images and videos are read.
        caption, (images, embeddings, rejected), _ = await asyncio.gather(
            self._judge_caption(request.text),
            self.moderate_images(request.images, sink),
            self.moderate_videos(request.videos, sink),
        )
        items = [*caption, *images, *sink.items]
        winner = worst_item(items)
        return Decision(
            label=winner.label,
            category=winner.category,
            reason=build_reason(winner),
            items=tuple(items),
            policy_version=self._policy.version,
            embeddings=embeddings,
            model_versions=self._model_versions,
            rejected_images=tuple(rejected),
            rejected_videos=tuple(sink.rejected),
            video_frames=tuple(sink.frames),
        )

    async def moderate_videos(self, videos: Sequence[VideoRef], sink: VideoSink) -> None:
        """All the videos of a post are checked at the same time, within the worker's CPU and waiting limits.
        A video that cannot be checked is held and listed in rejected_videos; the other videos are still checked.
        Any other error stops the rest and propagates, so the whole job is retried."""
        if not videos:
            return
        assert self._video is not None  # moderate() refuses videos without the pipeline
        parts = [VideoSink() for _ in videos]
        tasks = [asyncio.create_task(self._moderate_downloaded(video, part)) for video, part in zip(videos, parts, strict=True)]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)  # let each one delete its temporary copy
            raise
        for part in parts:  # in the order sent, so the answer never depends on which video finished first
            sink.items.extend(part.items)
            sink.rejected.extend(part.rejected)
            sink.frames.extend(part.frames)

    async def _moderate_downloaded(self, video: VideoRef, sink: VideoSink) -> None:
        assert self._video is not None
        try:
            await self._prepare_then_score(video.media_id, self._video.fetcher.download(video.url), sink)
        except PermanentError as exc:
            self._reject_video(video.media_id, exc, sink)

    async def _prepare_then_score(
        self, media_id: str, source: AbstractAsyncContextManager[Path], sink: VideoSink
    ) -> tuple[ItemResult, ...]:
        """One video, in two stages.

        CPU stage (many videos at once, `gate.cpu`): the file is downloaded, probed, its frames picked and its text
        read. The file is deleted as soon as the frames are picked, so a waiting video holds only its JPEG frames.
        GPU stage (one video at a time, because the models take one call at a time): the frames are scored and the
        text judged. A prepared video waits in line for it, and `gate.waiting` limits how many may wait."""
        assert self._video is not None
        gate, checker = self._video.gate, self._video.checker
        holding = False
        entered = 0
        try:
            async with gate.cpu:
                async with source as path:
                    prepared = await self._prepare_frames(path)
                await gate.waiting.acquire()  # full: this video keeps its CPU slot, so no new video starts
                holding = True
            entered = prepared.nbytes
            gate.enter_waiting(entered)
            verdict = await checker.check(media_id, prepared.selection.frames, prepared.ocr)
        finally:
            if entered:
                gate.leave_waiting(entered)
            if holding:
                gate.waiting.release()
        sink.items.extend(verdict.items)
        for frame, result in zip(prepared.selection.frames, verdict.frames, strict=True):
            sink.frames.append(
                VideoFrameRecord(
                    media_id=media_id,
                    frame_no=frame.frame_no,
                    time_ms=frame.time_ms,
                    source=frame.source,
                    scores=result.scores,
                    flagged=result.flagged,
                    jpeg=frame.jpeg if result.flagged else b"",  # kept only as evidence for a flagged frame
                )
            )
        return verdict.items

    async def _prepare_frames(self, path: Path) -> PreparedVideo:
        """The CPU steps on one local file. Raises PermanentError when the file cannot be checked."""
        assert self._video is not None
        cfg = self._video.cfg
        info = await asyncio.to_thread(probe, path, cfg)
        selection = await asyncio.to_thread(select_frames, path, cfg, info)
        ocr = await self._video.checker.read_text(selection.frames)
        return PreparedVideo(selection, ocr)

    @staticmethod
    def _reject_video(media_id: str, exc: PermanentError, sink: VideoSink) -> None:
        detail = REJECT_VIDEO_DETAILS.get(exc.code, "video could not be checked")
        sink.rejected.append(RejectedVideo(media_id, exc.code, detail))
        sink.items.append(ItemResult(ItemSource.VIDEO, media_id, Label.HOLD, None, None, f"Video {media_id}: {detail}"))

    async def _judge_caption(self, text: str | None) -> list[ItemResult]:
        if not text:
            return []
        cleaned = clean_text(text)
        if not cleaned:  # only spaces or invisible characters: nothing to check
            return [ItemResult(ItemSource.TEXT, None, Label.ALLOW, None, None)]
        [items] = await self._judge_texts([cleaned])
        return items

    async def moderate_images(
        self, images: Sequence[ImageRef], sink: VideoSink | None = None
    ) -> tuple[list[ItemResult], dict[str, tuple[float, ...]], list[RejectedImage]]:
        """An image we cannot check is held and reported in `rejected`; the other images are still checked.
        An animated image (GIF, animated WebP or APNG) is checked frame by frame, as a video (D2)."""
        sink = sink if sink is not None else VideoSink()
        fetched = await asyncio.gather(*(self._fetch(image) for image in images))
        embeddings: dict[str, tuple[float, ...]] = {}
        rejected: list[RejectedImage] = []
        by_position: dict[int, ItemResult] = {}
        usable: list[tuple[int, ImageRef, bytes]] = []
        animated: list[tuple[int, ImageRef, bytes]] = []
        for position, (image, blob) in enumerate(zip(images, fetched, strict=True)):
            if isinstance(blob, PermanentError):
                by_position[position] = self._reject(image.media_id, blob.code, rejected)
            elif is_animated(blob):
                if self._video is None:
                    by_position[position] = ItemResult(
                        ItemSource.IMAGE, image.media_id, Label.HOLD, None, None, ANIMATED_REASON
                    )
                else:
                    animated.append((position, image, blob))
            else:
                usable.append((position, image, blob))
        data = [blob for _, _, blob in usable]
        scores, ocr_results = await asyncio.gather(score_each(self._image.score, data), score_each(self._ocr.read, data))

        ids: list[str] = []
        good_ocr: list[OcrResult] = []
        for (position, image, _), score, ocr in zip(usable, scores, ocr_results, strict=True):
            if isinstance(score, PermanentError) or isinstance(ocr, PermanentError):
                by_position[position] = self._reject(image.media_id, MEDIA_UNDECODABLE, rejected)
                continue
            ids.append(image.media_id)
            good_ocr.append(ocr)
            embeddings[image.media_id] = score.embedding
            by_position[position] = self._policy.judge_image(image.media_id, score.categories)
        for position, image, blob in animated:
            by_position[position] = await self._check_animated(image.media_id, blob, sink, rejected)
        items = [by_position[position] for position in range(len(images))]
        items.extend(await self._check_image_text(ids, good_ocr))
        return items, embeddings, rejected

    async def _check_animated(
        self, media_id: str, blob: bytes, sink: VideoSink, rejected: list[RejectedImage]
    ) -> ItemResult:
        """Run an animated image through the frame pipeline. Its own temporary copy is deleted in `finally`."""
        assert self._video is not None
        try:
            verdict_items = await self._prepare_then_score(media_id, self._temp_copy(blob), sink)
        except PermanentError as exc:
            return self._reject(media_id, exc.code, rejected)
        return worst_item(list(verdict_items))

    @contextlib.asynccontextmanager
    async def _temp_copy(self, blob: bytes) -> AsyncIterator[Path]:
        """The animated image written to a temporary file, deleted when the block ends."""
        assert self._video is not None
        path = await asyncio.to_thread(write_temp_copy, blob, self._video.cfg)
        try:
            yield path
        finally:
            path.unlink(missing_ok=True)

    async def _fetch(self, image: ImageRef) -> bytes | PermanentError:
        """A permanent fetch problem is about this image only. A retryable one still retries the whole job."""
        try:
            return await self._fetcher.fetch(image.url)
        except PermanentError as exc:
            return exc

    @staticmethod
    def _reject(media_id: str, code: str, rejected: list[RejectedImage]) -> ItemResult:
        detail = REJECT_DETAILS.get(code, "image could not be checked")
        rejected.append(RejectedImage(media_id, code, detail))
        return ItemResult(ItemSource.IMAGE, media_id, Label.HOLD, None, None, f"Image {media_id}: {detail}")

    async def judge_texts(self, texts: list[str]) -> list[list[ItemResult]]:
        """Public for the video checker: on-screen text is judged by exactly the caption rules."""
        return await self._judge_texts(texts)

    async def _judge_texts(self, texts: list[str]) -> list[list[ItemResult]]:
        """Each cleaned text: its worst segment verdict, plus a hold for prompt-like text.
        All segments of all texts go to the model in one call."""
        segments = [split_segments(text) for text in texts]
        scores = iter(await self._text.score([s.text for segs in segments for s in segs]))
        out = []
        for text, segs in zip(texts, segments, strict=True):
            row_scores = [next(scores) for _ in segs]
            # The whole text is only personal details (allowed): its intent clause decides, so it may block.
            decides = row_scores[0].category in self._policy.always_allow
            items = [self._judge_segment(s, sc, decides) for s, sc in zip(segs, row_scores, strict=True)]
            out.append(self._text_items(text, items))
        return out

    def _judge_segment(self, segment: Segment, score: TextScore, decides: bool) -> ItemResult:
        item = self._policy.judge_text(score)
        if segment.sentence and item.label is Label.BLOCK and not decides:
            return replace(item, label=Label.HOLD)  # one sentence out of context holds, never blocks
        return item

    def _text_items(self, text: str, segment_items: list[ItemResult]) -> list[ItemResult]:
        items = [worst_item(segment_items)]
        if has_template_text(text):
            items.append(ItemResult(ItemSource.TEXT, None, Label.HOLD, None, None, TEMPLATE_NOTE))
        return items

    async def _check_image_text(self, media_ids: list[str], ocr_results: list[OcrResult]) -> list[ItemResult]:
        """No text: nothing to check. Text we cannot read: hold. Readable text: judged like a caption."""
        items: list[ItemResult] = []
        readable: list[tuple[str, str]] = []
        for media_id, result in zip(media_ids, ocr_results, strict=True):
            if len("".join(result.text.split())) < self._policy.ocr_min_chars:  # letters only, not spaces or line breaks
                continue  # nothing read, or a stray mark: a photo without writing is not held for it
            if result.mean_confidence is None or result.mean_confidence < self._policy.ocr_min_confidence:
                note = f"Image {media_id}: text could not be read"
                items.append(ItemResult(ItemSource.IMAGE_TEXT, media_id, Label.HOLD, None, None, note))
            elif text := clean_text(result.text.replace("\n", " ")):
                readable.append((media_id, text))  # OCR lines are joined: one OCR line is not a sentence
        if readable:
            judged = await self._judge_texts([text for _, text in readable])
            for (media_id, _), text_items in zip(readable, judged, strict=True):
                items.extend(replace(item, source=ItemSource.IMAGE_TEXT, media_id=media_id) for item in text_items)
        return items


def build_reason(item: ItemResult) -> str:
    if item.label is Label.ALLOW:
        return "Allowed"
    if item.note:
        return item.note
    name = CATEGORIES[item.category] if item.category is not None else UNCLASSIFIED
    if item.source is ItemSource.TEXT:
        where = "Caption"
    elif item.source is ItemSource.IMAGE_TEXT:
        where = f"Text in image {item.media_id}"
    elif item.source is ItemSource.VIDEO_TEXT:
        where = f"Text in video {item.media_id}"
    elif item.source is ItemSource.VIDEO:
        where = f"Video {item.media_id}"
    else:
        where = f"Image {item.media_id}"
    return f"{where}: {name}"
