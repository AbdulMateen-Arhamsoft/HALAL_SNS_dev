"""D2 with the real models: the whole pipeline on a synthetic clip. Skipped unless the weights are downloaded.

These tests prove the pipeline runs end to end on the real SigLIP 2, RapidOCR and Qwen3Guard, and that a
clean clip is allowed. They do not measure accuracy: that is the D3 calibration (plan section 11).
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import video_fixtures as fx

from moderation.config import Settings
from moderation.domain import Label, ModerationRequest, VideoRef
from moderation.moderate import Moderator, VideoPipeline
from moderation.policy import D1_POLICY
from moderation.video.check import VideoChecker
from moderation.video.fetch import VideoFetcher
from moderation.video.frames import select_frames
from moderation.video.probe import probe
from tests.models.conftest import HEAD_DIR, IMAGE_DIR, OCR_DIR, TEXT_DIR


@pytest.fixture(scope="module")
def real_models() -> object:
    for module in ("torch", "transformers", "rapidocr"):
        pytest.importorskip(module)
    for folder in (TEXT_DIR, IMAGE_DIR, HEAD_DIR, OCR_DIR):
        if not folder.exists():
            pytest.skip(f"model folder missing: {folder} (run python model_download.py pull)")
    from moderation.worker import load_models

    cfg = Settings(_env_file=None, database_url="postgresql://unused@localhost/unused")
    text, image, ocr = load_models(cfg)
    yield cfg, text, image, ocr
    text.close()
    image.close()
    ocr.close()


async def test_a_clean_clip_goes_through_the_whole_pipeline(tmp_path: Path, real_models: object) -> None:
    cfg, text, image, ocr = real_models  # type: ignore[misc]
    settings = cfg.model_copy(update={"video_temp_dir": str(tmp_path), "video_evidence_root": str(tmp_path / "ev")})
    clip = fx.write_clip(tmp_path / "clean.mp4", fx.scene_frames([(1, 60), (2, 60)]))
    info = probe(clip, settings)
    selection = select_frames(clip, settings, info)
    moderator = Moderator(text=text, image=image, ocr=ocr, fetcher=None, policy=D1_POLICY)
    checker = VideoChecker(
        image=image,
        ocr=ocr,
        judge_texts=moderator.judge_texts,
        policy=D1_POLICY,
        hard_block=settings.video_hard_block_codes,
        text_unreadable_share=settings.video_text_unreadable_share,
    )
    verdict = await checker.check("clean.mp4", selection.frames)
    assert verdict.combined is not None
    assert len(verdict.frames) == len(selection.frames) >= 2
    assert set(verdict.timings) == {"score_s", "ocr_s", "text_s"}
    assert verdict.items[0].label is Label.ALLOW


async def test_the_pipeline_wires_into_the_moderator_as_the_worker_does(tmp_path: Path, real_models: object) -> None:
    cfg, text, image, ocr = real_models  # type: ignore[misc]
    settings = cfg.model_copy(update={"video_temp_dir": str(tmp_path), "video_evidence_root": str(tmp_path / "ev")})
    root = tmp_path / "root" / "uploads"
    root.mkdir(parents=True)
    fx.write_clip(root / "clip.mp4", fx.scene_frames([(5, 60)]))
    settings = settings.model_copy(update={"image_allowed_root": str(tmp_path / "root")})
    moderator = Moderator(text=text, image=image, ocr=ocr, fetcher=None, policy=D1_POLICY)
    checker = VideoChecker(
        image=image,
        ocr=ocr,
        judge_texts=moderator.judge_texts,
        policy=D1_POLICY,
        hard_block=settings.video_hard_block_codes,
        text_unreadable_share=settings.video_text_unreadable_share,
    )
    fetcher = VideoFetcher(settings, httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(404))))
    moderator.attach_video(VideoPipeline(cfg=settings, fetcher=fetcher, checker=checker))
    request = ModerationRequest(
        request_id="r1",
        surface="reel",
        content_id="reel-1",
        content_version=1,
        author_id="u1",
        videos=[VideoRef(media_id="v1", url="uploads/clip.mp4")],
    )
    decision = await moderator.moderate(request)
    assert decision.label is Label.ALLOW
    assert decision.rejected_videos == ()
    assert decision.video_frames
