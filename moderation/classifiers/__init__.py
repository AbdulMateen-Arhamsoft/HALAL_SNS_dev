"""The three models, loaded once by the worker: Qwen3Guard (text), SigLIP2 + head (images), RapidOCR (text in images)."""

from __future__ import annotations

import asyncio
import functools
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from moderation.errors import MODEL_OOM, RetryableError


class SingleThread:
    """One worker thread per model: blocking model calls never overlap and never block the event loop."""

    def __init__(self, name: str) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)

    async def run[**P, R](self, fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        return await asyncio.get_running_loop().run_in_executor(self._executor, functools.partial(fn, *args, **kwargs))

    def close(self) -> None:
        self._executor.shutdown(wait=True)


class OcrPool:
    """Several independent OCR engines (CPU), so that several videos can be read at the same time.

    One RapidOCR engine must not be called from two threads at once, so each engine keeps its own thread
    and a call borrows a free engine. With one engine this behaves like the engine itself. The engines
    are small ONNX models on the CPU: unlike the GPU models, a copy costs little memory."""

    def __init__(self, engines: Sequence[Any]) -> None:
        if not engines:
            raise ValueError("an OCR pool needs at least one engine")
        self._engines = list(engines)
        self._free: asyncio.Queue[Any] | None = None  # built on first use, inside the running event loop

    @property
    def size(self) -> int:
        return len(self._engines)

    async def read(self, images: Sequence[bytes]) -> list[Any]:
        if self._free is None:
            self._free = asyncio.Queue()
            for engine in self._engines:
                self._free.put_nowait(engine)
        engine = await self._free.get()
        try:
            return list(await engine.read(images))
        finally:
            self._free.put_nowait(engine)

    def close(self) -> None:
        for engine in self._engines:
            engine.close()


def split_on_oom[S, T, R](fn: Callable[[S, Sequence[T]], list[R]]) -> Callable[[S, Sequence[T]], list[R]]:
    """Decorator for a model's sync batch method (GPU out of memory): free the cache and
    run the two halves; a single item that still does not fit retries the job later."""

    @functools.wraps(fn)
    def wrapper(self: S, items: Sequence[T]) -> list[R]:
        import torch

        try:
            return fn(self, items)
        except torch.OutOfMemoryError as exc:
            if len(items) <= 1:
                raise RetryableError(MODEL_OOM, "one item does not fit in GPU memory") from exc
            torch.cuda.empty_cache()
            half = len(items) // 2
            return wrapper(self, items[:half]) + wrapper(self, items[half:])

    return wrapper
