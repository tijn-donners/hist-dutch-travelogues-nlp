"""Shared Ollama streaming utility for real-time LLM feedback.

Used by all modules that call Ollama to display thinking tokens and
response content as they arrive, so the user knows the LLM is working.
"""

import time

import httpx
import ollama
from ollama import Client


def stream_ollama_chat(model, prompt, host, api_key, timeout=600.0,
                       temperature=0.0, think=None, think_log_path=None, **options):
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
            if not content_text and thinking:
                # The model emitted its answer as thinking only (no content).
                # Fall back to the thinking text so the caller can still parse it.
                print("  [warning] model returned no content; falling back to "
                      "thinking trace", flush=True)
                return thinking
            return content_text

        except (ollama.RequestError, httpx.TimeoutException, httpx.ConnectError) as e:
            if attempt < len(retry_delays):
                delay = retry_delays[attempt]
                print(f"\n  [retry {attempt + 1}/{len(retry_delays)}] "
                      f"Network error: {e}. Sleeping {delay}s...", flush=True)
                time.sleep(delay)
            else:
                print(f"\n  [retry exhausted] All {len(retry_delays)} retries failed: {e}")
                raise
