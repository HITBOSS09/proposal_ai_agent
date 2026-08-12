#!/usr/bin/env python3
"""Reproduce the proposal Ollama HTTP request without proposal-pipeline code."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from urllib.request import Request, urlopen

import requests


ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = ROOT / "debug" / "prompt.txt"
URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:3b"
TIMEOUT_SECONDS = 60
HEADERS = {"Content-Type": "application/json"}
OPTIONS = {"temperature": 0.1}


def _prompt_parts() -> tuple[str, str]:
    complete_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    prefix = "[SYSTEM]\n"
    separator = "\n\n[USER]\n"
    if not complete_prompt.startswith(prefix) or separator not in complete_prompt:
        raise ValueError(f"{PROMPT_PATH} does not contain the expected system/user prompt format")
    return complete_prompt[len(prefix) :].split(separator, maxsplit=1)


def _payload() -> tuple[dict[str, object], bytes]:
    system_prompt, user_prompt = _prompt_parts()
    payload: dict[str, object] = {
        "model": MODEL,
        "system": system_prompt,
        "prompt": user_prompt,
        "stream": False,
        "options": OPTIONS,
    }
    return payload, json.dumps(payload).encode("utf-8")


def _report(name: str, started_at: float, status: int | None, response_body: bytes | None) -> None:
    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    print(f"{name} elapsed_ms={elapsed_ms:.1f}")
    print(f"{name} http_status={status}")
    print(f"{name} response_length={len(response_body) if response_body is not None else None}")


def run_urllib(body: bytes) -> bool:
    request = Request(URL, data=body, headers=HEADERS, method="POST")
    started_at = time.perf_counter()
    try:
        print("urllib ENTER urlopen")
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            print("urllib EXIT urlopen")
            status = response.getcode()
            print("urllib ENTER response.read")
            response_body = response.read()
            print("urllib EXIT response.read")
        _report("urllib", started_at, status, response_body)
        return True
    except Exception as error:
        _report("urllib", started_at, None, None)
        print(f"urllib exception_type={type(error).__name__}")
        print(f"urllib exception_repr={error!r}")
        print(f"urllib exception_args={error.args!r}")
        return False


def run_requests(payload: dict[str, object]) -> bool:
    started_at = time.perf_counter()
    try:
        print("requests ENTER post")
        response = requests.post(URL, json=payload, headers=HEADERS, timeout=TIMEOUT_SECONDS)
        print("requests EXIT post")
        print("requests ENTER response.content")
        response_body = response.content
        print("requests EXIT response.content")
        _report("requests", started_at, response.status_code, response_body)
        return True
    except Exception as error:
        _report("requests", started_at, None, None)
        print(f"requests exception_type={type(error).__name__}")
        print(f"requests exception_repr={error!r}")
        print(f"requests exception_args={error.args!r}")
        return False


def run_curl(body: bytes) -> bool:
    started_at = time.perf_counter()
    command = [
        "curl",
        "--silent",
        "--show-error",
        "--output",
        "-",
        "--write-out",
        "\nCURL_HTTP_STATUS=%{http_code}",
        "--max-time",
        str(TIMEOUT_SECONDS),
        "--header",
        "Content-Type: application/json",
        "--data-binary",
        "@-",
        URL,
    ]
    try:
        print("curl ENTER subprocess.run")
        result = subprocess.run(command, input=body, capture_output=True, check=False)
        print("curl EXIT subprocess.run")
        marker = b"\nCURL_HTTP_STATUS="
        response_body, _, status_text = result.stdout.rpartition(marker)
        status = int(status_text) if status_text.isdigit() else None
        _report("curl", started_at, status, response_body)
        print(f"curl return_code={result.returncode}")
        if result.stderr:
            print(f"curl stderr={result.stderr.decode('utf-8', errors='replace')!r}")
        return result.returncode == 0
    except Exception as error:
        _report("curl", started_at, None, None)
        print(f"curl exception_type={type(error).__name__}")
        print(f"curl exception_repr={error!r}")
        print(f"curl exception_args={error.args!r}")
        return False


def main() -> int:
    payload, body = _payload()
    print(f"url={URL}")
    print(f"model={MODEL}")
    print(f"timeout={TIMEOUT_SECONDS}")
    print(f"request_size_bytes={len(body)}")
    print(f"prompt_char_count={len(payload['system']) + len(payload['prompt'])}")
    results = {
        "urllib": run_urllib(body),
        "requests": run_requests(payload),
        "curl": run_curl(body),
    }
    print(f"comparison={results}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
