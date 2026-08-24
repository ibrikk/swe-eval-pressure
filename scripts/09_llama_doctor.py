#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request

from litellm_pool import parse_litellm_keys


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(2)


def gateway_model_id(model: str) -> str:
    # Mini-SWE uses `openai/<gateway-public-model-id>` so LiteLLM routes through
    # the OpenAI-compatible Scale gateway. Raw gateway requests use the public ID.
    return model[len("openai/") :] if model.startswith("openai/") else model


def request_json(url: str, key: str, payload: dict) -> tuple[int, dict | str]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8", "replace")
            status = resp.status
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        status = exc.code
    except Exception as exc:
        return 0, f"{type(exc).__name__}: {exc}"
    try:
        return status, json.loads(body)
    except json.JSONDecodeError:
        return status, body


def extract_content(body: dict | str) -> str:
    if not isinstance(body, dict):
        return str(body)
    try:
        return body["choices"][0]["message"].get("content") or ""
    except Exception:
        return json.dumps(body, ensure_ascii=False)


def main() -> None:
    keys = parse_litellm_keys(fallback_key=os.getenv("OPENAI_API_KEY", ""))
    base = os.getenv("LITE_LLM_URL") or os.getenv("OPENAI_BASE_URL")
    if not keys:
        die("LITE_LLM_KEY / LITE_LLM_KEYS / OPENAI_API_KEY is empty")
    if not base:
        die("LITE_LLM_URL / OPENAI_BASE_URL is empty")

    endpoint = base.rstrip("/") + "/chat/completions"
    candidates_raw = os.getenv(
        "LLAMA_MODEL_CANDIDATES",
        "llmengine/llama-3-3-70b-instruct,"
        "bedrock/us.meta.llama3-3-70b-instruct-v1:0,"
        "bedrock/meta.llama3-3-70b-instruct-v1:0",
    )
    candidates = [x.strip() for x in candidates_raw.split(",") if x.strip()]
    if os.getenv("LLAMA_MODEL"):
        current = gateway_model_id(os.environ["LLAMA_MODEL"].strip())
        if current and current not in candidates:
            candidates.insert(0, current)

    print(f"Gateway: {endpoint}")
    print(f"LiteLLM keys: {len(keys)}")
    print(f"Candidates: {len(candidates)}")
    print("Probe: ordinary chat completion + exact Mini-SWE text-command format\n")

    pattern = re.compile(r"```mswea_bash_command\s*\n(.*?)\n```", re.S)
    passed: list[str] = []

    for model in candidates:
        public_model = gateway_model_id(model)
        payload = {
            "model": public_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are being tested for command-format compliance. "
                        "Return exactly one fenced mswea_bash_command block and no other code block."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Return a command that prints LLAMA_ROUTE_OK. Use exactly this format:\n"
                        "```mswea_bash_command\n"
                        "printf 'LLAMA_ROUTE_OK\\n'\n"
                        "```"
                    ),
                },
            ],
            "temperature": 0,
            "max_tokens": 160,
        }
        status, body = 0, ""
        for key_index, key in enumerate(keys, start=1):
            status, body = request_json(endpoint, key, payload)
            if status != 429:
                break
            print(f"  key {key_index}/{len(keys)} rate-limited; trying next key")
        content = extract_content(body)
        match = pattern.search(content)
        command_ok = bool(match and "LLAMA_ROUTE_OK" in match.group(1))
        http_ok = 200 <= status < 300
        verdict = "PASS" if http_ok and command_ok else "FAIL"
        print(f"[{verdict}] {public_model}")
        print(f"  HTTP: {status}")
        if http_ok:
            shown = content.replace("\n", "\\n")[:300]
            print(f"  content: {shown}")
            if not command_ok:
                print("  reason: route answered, but did not follow the text-command format")
        else:
            if isinstance(body, dict):
                detail = body.get("error") or body.get("detail") or body.get("message") or body
                print("  error:", json.dumps(detail, ensure_ascii=False)[:500])
            else:
                print("  error:", str(body)[:500])
        if http_ok and command_ok:
            passed.append(public_model)
        print()

    if passed:
        winner = passed[0]
        print("Recommended first smoke route:")
        print(f"  export LLAMA_MODEL='openai/{winner}'")
        print("  ./lab.sh prepare pilot llama")
        print("  HARBOR_REPEATS=1 ./lab.sh run pilot llama")
        print("\nDo not launch the 210-run resource experiment until at least a small Harbor smoke finishes without provider/tool-format failures.")
    else:
        print("No candidate passed both route access and text-command formatting.")
        print("Run `./lab.sh models` to inspect the raw gateway response, then update LLAMA_MODEL_CANDIDATES with routes your key can access.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
