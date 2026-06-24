"""Monkey-patch LangChain's Ollama backend to print streaming tokens in real-time.

The NER pipeline uses spaCy-LLM → langchain_community.llms.Ollama, which streams
tokens over HTTP but aggregates them internally before returning. This module
patches ``_create_generate_stream`` (sync) and ``_acreate_generate_stream``
(async) to print each token as it arrives, so the user can see the model is
actually generating and not stuck.

Usage::

    from ner.streaming_patch import patch_langchain_ollama_streaming
    patch_langchain_ollama_streaming()
    # … normal NER pipeline assembly and loop …
"""

import json
import os
import sys
from typing import Optional, TextIO

# ---------------------------------------------------------------------------
# Output stream selection
#
# The shell script (ner_and_eval.sh) pipes stderr through 2>&1 | tee which
# causes pipe buffering and would save streaming tokens to the log file.
# To avoid both problems we write directly to /dev/tty when available (real
# terminal), falling back to stderr for non-TTY environments (HPC/SLURM).
# ---------------------------------------------------------------------------
_STREAM_OUT: Optional[TextIO] = None


def _get_out() -> TextIO:
    global _STREAM_OUT
    if _STREAM_OUT is not None:
        return _STREAM_OUT
    try:
        _STREAM_OUT = open("/dev/tty", "w", buffering=1)
    except OSError:
        _STREAM_OUT = sys.stderr
    return _STREAM_OUT


# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------
def _dim(text: str) -> str:
    return f"\x1b[2;3m{text}\x1b[0m"


def _green(text: str) -> str:
    return f"\x1b[32m{text}\x1b[0m"


def _cyan(text: str) -> str:
    return f"\x1b[36m{text}\x1b[0m"


def _yellow(text: str) -> str:
    return f"\x1b[33m{text}\x1b[0m"


def _red(text: str) -> str:
    return f"\x1b[31m{text}\x1b[0m"


def _supports_ansi() -> bool:
    """Check whether the real output terminal supports ANSI escape codes."""
    for h in (sys.stdout, sys.stderr):
        if hasattr(h, "isatty") and h.isatty():
            return True
    try:
        fd = os.open("/dev/tty", os.O_RDONLY)
        is_tty = os.isatty(fd)
        os.close(fd)
        return is_tty
    except OSError:
        return False


_USE_ANSI = _supports_ansi()

# ---------------------------------------------------------------------------
# Thinking-loop guard (disabled)
#
# Some models get stuck repeating the same thinking token indefinitely.
# We *could* abort when thinking exceeds a threshold without any response
# token appearing, but in practice legitimate long-thinking models (Kimi,
# DeepSeek, etc.) regularly exceed 20 000 thinking tokens on difficult
# pages.  The guard was doing more harm than good — it was aborting valid
# thinking sessions.  The cloud API's own server-side timeout is the real
# backstop.
# ---------------------------------------------------------------------------
# _MAX_THINK_TOKENS = 20_000
# class _ThinkingLoopError(Exception): ...


# ---------------------------------------------------------------------------
# Patch target: _create_generate_stream  (sync)
# ---------------------------------------------------------------------------

def _patch_generate_stream(original):
    """Return a patched version of ``_create_generate_stream`` that prints
    tokens as they arrive, while still yielding raw response lines."""

    def patched(self, prompt, stop=None, images=None, **kwargs):
        think_count = 0
        resp_count = 0
        for raw_line in original(self, prompt, stop, images, **kwargs):
            if raw_line:
                try:
                    data = json.loads(raw_line)
                except json.JSONDecodeError:
                    pass
                else:
                    # --- Thinking tokens (model's internal reasoning) ---
                    think = data.get("thinking", "")
                    if think:
                        think_count += 1
                        if think_count == 1:
                            print(
                                _cyan("\n  ═══ think ═══") if _USE_ANSI
                                else "\n  --- think ---",
                                file=_get_out(), flush=True,
                            )
                        print(
                            (_dim(think) if _USE_ANSI else think),
                            end="", file=_get_out(), flush=True,
                        )
                    # --- Response tokens (normal output) ---
                    text = data.get("response", "")
                    if text:
                        resp_count += 1
                        if resp_count == 1:
                            print(
                                _cyan("\n  ═══ response ═══") if _USE_ANSI
                                else "\n  --- response ---",
                                file=_get_out(), flush=True,
                            )
                        print(text, end="", file=_get_out(), flush=True)
            yield raw_line

        if (think_count + resp_count) > 0:
            print(
                _green(f"\n  ✓ {think_count + resp_count} tokens") if _USE_ANSI
                else f"\n  -> {think_count + resp_count} tokens",
                file=_get_out(), flush=True,
            )

    return patched


def _patch_async_generate_stream(original):
    """Return a patched async version of ``_acreate_generate_stream``."""

    async def patched(self, prompt, stop=None, images=None, **kwargs):
        think_count = 0
        resp_count = 0
        async for raw_line in original(self, prompt, stop, images, **kwargs):
            if raw_line:
                try:
                    data = json.loads(raw_line)
                except json.JSONDecodeError:
                    pass
                else:
                    # --- Thinking tokens (model's internal reasoning) ---
                    think = data.get("thinking", "")
                    if think:
                        think_count += 1
                        if think_count == 1:
                            print(
                                _cyan("\n  ═══ think ═══") if _USE_ANSI
                                else "\n  --- think ---",
                                file=_get_out(), flush=True,
                            )
                        print(
                            (_dim(think) if _USE_ANSI else think),
                            end="", file=_get_out(), flush=True,
                        )

                    # --- Response tokens (normal output) ---
                    text = data.get("response", "")
                    if text:
                        resp_count += 1
                        if resp_count == 1:
                            print(
                                _cyan("\n  ═══ response ═══") if _USE_ANSI
                                else "\n  --- response ---",
                                file=_get_out(), flush=True,
                            )
                        print(text, end="", file=_get_out(), flush=True)
            yield raw_line

        if (think_count + resp_count) > 0:
            print(
                _green(f"\n  ✓ {think_count + resp_count} tokens") if _USE_ANSI
                else f"\n  -> {think_count + resp_count} tokens",
                file=_get_out(), flush=True,
            )

    return patched


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
_PATCHED: bool = False


def patch_langchain_ollama_streaming() -> None:
    """Monkey-patch ``langchain_community.llms.ollama.Ollama`` to stream tokens.

    Safe to call multiple times — only the first call applies the patch.
    Prints to ``/dev/tty`` when available (bypasses redirects), falling back
    to *stderr* for non-TTY environments (HPC/SLURM).
    """
    global _PATCHED
    if _PATCHED:
        return

    try:
        from langchain_community.llms import ollama as ollama_module
    except ImportError:
        print(
            _yellow("[streaming_patch] langchain_community not available, skipping")
            if _USE_ANSI else
            "[streaming_patch] langchain_community not available, skipping",
            file=sys.stderr,
        )
        return

    # Patch sync path
    ollama_module.Ollama._create_generate_stream = _patch_generate_stream(
        ollama_module.Ollama._create_generate_stream,
    )
    # Patch async path (future-proofing, currently unused by nlp())
    ollama_module.Ollama._acreate_generate_stream = _patch_async_generate_stream(
        ollama_module.Ollama._acreate_generate_stream,
    )

    _PATCHED = True
    print(
        _yellow("[streaming_patch] active — tokens will appear on stderr")
        if _USE_ANSI else
        "[streaming_patch] active",
        file=sys.stderr,
    )
