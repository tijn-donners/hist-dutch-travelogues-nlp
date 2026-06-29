"""Shared Ollama streaming utility for real-time LLM feedback.

Used by all modules that call Ollama to display thinking tokens and
response content as they arrive, so the user knows the LLM is working.
"""

import os
import time

import httpx
import ollama
from ollama import Client


def _ns_to_ms(value):
    """Convert Ollama duration fields (nanoseconds) to milliseconds, or None."""
    if value is None:
        return None
    return round(value / 1_000_000, 3)


def resolve_ollama_host(host=None, api_key=None):
    """Resolve an Ollama host URL and API key for reproducible local/cloud runs.

    Centralises the host-selection logic so every pipeline stage (NER, EL, RE)
    interprets the ``--host`` CLI flag and the ``OLLAMA_HOST`` env var the same
    way. Accepts:

    - ``None`` / ``""`` / ``"default"``: auto-switch — cloud
      (``https://ollama.com``) when ``OLLAMA_API_KEY`` is set, else
      ``http://localhost:11434``.
    - ``"cloud"``: alias for ``https://ollama.com`` (auth from env/``api_key``).
    - ``"localhost"``: alias for ``http://localhost:11434`` (no auth).
    - any other string: treated as a verbatim URL, e.g.
      ``http://localhost:1344`` or ``http://10.0.0.2:11434``.

    Args:
        host: Value from the ``--host`` CLI arg (or the ``OLLAMA_HOST`` env
            var). ``None`` triggers the auto-switch.
        api_key: Explicit API key, or ``None`` to read ``OLLAMA_API_KEY`` from
            the environment.

    Returns:
        ``(url, api_key)`` where ``api_key`` is ``None`` for local targets
        (localhost / 127.0.0.1), so callers can skip the Authorization header.
    """
    if api_key is None:
        api_key = os.environ.get("OLLAMA_API_KEY")

    if host is None or host == "" or host == "default":
        if api_key:
            url = "https://ollama.com"
        else:
            url = "http://localhost:11434"
    elif host == "cloud":
        url = "https://ollama.com"
    elif host == "localhost":
        url = "http://localhost:11434"
        api_key = None
    else:
        url = host  # verbatim URL (e.g. http://localhost:1344)
        # Local servers don't auth; drop any inherited key to avoid sending it.
        if "localhost" in url or "127.0.0.1" in url or "0.0.0.0" in url:
            api_key = None
    return url, api_key


def stream_ollama_chat(model, prompt, host, api_key, timeout=600.0,
                       temperature=0.0, think=None, think_log_path=None,
                       stats: dict | None = None, **options):
    """Stream an Ollama chat response, printing thinking and content in real time.

    Retries up to 5 times on transient network errors with exponential backoff
    (5, 10, 20, 40, 80 seconds) so that brief DNS or connectivity hiccups do
    not cause the pipeline stage to fail.

    Reasoning models (e.g. deepseek-v4-pro) sometimes emit their entire answer
    in the ``thinking`` channel and produce no ``content``. To avoid losing an
    expensive long run, thinking tokens are captured alongside content. If the
    final ``content`` is empty but ``thinking`` is not, the thinking text is
    returned instead (with a warning) so the caller can still attempt to parse
    it. When ``think_log_path`` is given, the full thinking trace is also
    written to that file regardless of outcome.

    Args:
        model: Ollama model name (e.g. "gemma4:31b", "deepseek-v4-pro").
        prompt: The full prompt text to send as a user message.
        host: Ollama host URL (e.g. "https://ollama.com" or "http://localhost:11434").
        api_key: API key for cloud access, or None for localhost.
        timeout: HTTP read timeout in seconds.
        temperature: Model temperature.
        think: Thinking mode (True, False, "low", "medium", "high").
               None uses the model's default. Passed as a top-level API
               parameter, NOT nested inside options.
        think_log_path: Optional path; if set, the full thinking trace is
               written here so long reasoning-model runs can be salvaged even
               when content is empty or truncated.
        stats: Optional mutable dict. When provided, it is populated with
               diagnostic fields for this call: ``retries`` (internal
               network-retry count), ``wall_seconds`` (client wall time),
               ``thinking_chars``, ``content_chars``, ``fallback_used``
               (bool: returned the thinking trace because content was empty),
               and the final chunk's server-side stats — ``eval_count``,
               ``eval_duration_ms``, ``total_duration_ms``,
               ``load_duration_ms``, ``prompt_eval_count``,
               ``prompt_eval_duration_ms``, ``done_reason`` (ns converted to
               ms). Non-breaking: callers that omit ``stats`` are unaffected.
        **options: Additional options passed to the chat API's options dict.

    Returns:
        The full response text as a string (content, or thinking as a
        fallback when content is empty).

    Raises:
        ollama.ResponseError: The API returned an error (non-retriable).
        ollama.RequestError: The request could not be sent (retried).
        httpx.TimeoutException: The request timed out (retried).
    """
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    client = Client(host=host, headers=headers, timeout=timeout)

    # Retry on transient network errors: 5, 10, 20, 40, 80 seconds
    retry_delays = [5, 10, 20, 40, 80]

    t0 = time.time()
    retry_count = 0
    final_stats = {}

    for attempt in range(len(retry_delays) + 1):
        try:
            print("  Sending request...", flush=True)
            stream = client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": temperature, **options},
                stream=True,
                think=think,
            )
            chunks = []
            think_chunks = []
            in_think = False
            for chunk in stream:
                msg = chunk["message"]
                if msg.get("thinking"):
                    if not in_think:
                        print("\n  [thinking] ", end="", flush=True)
                        in_think = True
                    print(msg["thinking"], end="", flush=True)
                    think_chunks.append(msg["thinking"])
                content = msg.get("content")
                if content:
                    if in_think:
                        print()
                        in_think = False
                    print(content, end="", flush=True)
                    chunks.append(content)
                # The final streaming chunk carries done=True and the
                # server-side timing/token stats — capture them for diagnostics.
                if getattr(chunk, "done", None):
                    final_stats = {
                        "eval_count": getattr(chunk, "eval_count", None),
                        "eval_duration_ms": _ns_to_ms(getattr(chunk, "eval_duration", None)),
                        "total_duration_ms": _ns_to_ms(getattr(chunk, "total_duration", None)),
                        "load_duration_ms": _ns_to_ms(getattr(chunk, "load_duration", None)),
                        "prompt_eval_count": getattr(chunk, "prompt_eval_count", None),
                        "prompt_eval_duration_ms": _ns_to_ms(getattr(chunk, "prompt_eval_duration", None)),
                        "done_reason": getattr(chunk, "done_reason", None),
                    }
            if in_think:
                print()
            print()

            thinking = "".join(think_chunks)
            # Persist the thinking trace so long reasoning runs can be salvaged.
            if think_log_path and thinking:
                from pathlib import Path
                Path(think_log_path).parent.mkdir(parents=True, exist_ok=True)
                with open(think_log_path, "w", encoding="utf-8") as f:
                    f.write(thinking)

            content_text = "".join(chunks)
            fallback_used = False
            if not content_text and thinking:
                # The model emitted its answer as thinking only (no content).
                # Fall back to the thinking text so the caller can still parse it.
                print("  [warning] model returned no content; falling back to "
                      "thinking trace", flush=True)
                fallback_used = True
                result = thinking
            else:
                result = content_text

            if stats is not None:
                stats["retries"] = retry_count
                stats["wall_seconds"] = round(time.time() - t0, 3)
                stats["thinking_chars"] = len(thinking)
                stats["content_chars"] = len(content_text)
                stats["fallback_used"] = fallback_used
                stats.update(final_stats)
            return result

        except (ollama.RequestError, httpx.TimeoutException, httpx.ConnectError) as e:
            retry_count += 1
            if attempt < len(retry_delays):
                delay = retry_delays[attempt]
                print(f"\n  [retry {attempt + 1}/{len(retry_delays)}] "
                      f"Network error: {e}. Sleeping {delay}s...", flush=True)
                time.sleep(delay)
            else:
                print(f"\n  [retry exhausted] All {len(retry_delays)} retries failed: {e}")
                raise
