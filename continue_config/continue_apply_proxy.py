#!/usr/bin/env python3
import http.client
import json
import os
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse


OPEN_TAG = "<update_file>"
CLOSE_TAG = "</update_file>"

LISTEN_HOST = os.environ.get("APPLY_PROXY_HOST", "127.0.0.1")
LISTEN_PORT = int(os.environ.get("APPLY_PROXY_PORT", "14124"))
UPSTREAM_CHAT_URL = os.environ.get(
    "APPLY_PROXY_UPSTREAM_CHAT_URL",
    "http://127.0.0.1:12345/v1/chat/completions",
)
UPSTREAM_AUTH = os.environ.get("APPLY_PROXY_UPSTREAM_AUTH", "")
SHOW_UPSTREAM_IN_HEALTH = os.environ.get("APPLY_PROXY_HEALTH_SHOW_UPSTREAM", "").lower() in {
    "1",
    "true",
    "yes",
}


def make_request_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


class StreamingTagStripper:
    def __init__(self) -> None:
        self._buffer = ""

    def _drain(self, flush: bool) -> str:
        out = []
        while self._buffer:
            if self._buffer.startswith(OPEN_TAG):
                self._buffer = self._buffer[len(OPEN_TAG) :]
                continue
            if self._buffer.startswith(CLOSE_TAG):
                self._buffer = self._buffer[len(CLOSE_TAG) :]
                continue

            if not flush and (OPEN_TAG.startswith(self._buffer) or CLOSE_TAG.startswith(self._buffer)):
                break

            out.append(self._buffer[0])
            self._buffer = self._buffer[1:]
        return "".join(out)

    def feed(self, text: str) -> str:
        if not text:
            return ""
        self._buffer += text
        return self._drain(flush=False)

    def flush(self) -> str:
        return self._drain(flush=True)


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        message = "%s - - [%s] %s\n" % (
            self.client_address[0],
            self.log_date_time_string(),
            fmt % args,
        )
        print(message, end="")

    def _send_json(self, status: int, payload) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self._write_body(body)

    def _write_body(self, body: bytes, *, flush: bool = False) -> None:
        self.wfile.write(body)
        if flush:
            self.wfile.flush()

    def _write_sse_event(self, payload) -> None:
        body = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")
        self._write_body(body, flush=True)

    def _write_sse_done(self) -> None:
        self._write_body(b"data: [DONE]\n\n", flush=True)

    def do_GET(self) -> None:
        if self.path != "/health":
            self._send_json(404, {"error": "not_found"})
            return

        payload = {"ok": True, "listen": f"http://{LISTEN_HOST}:{LISTEN_PORT}"}
        if SHOW_UPSTREAM_IN_HEALTH:
            payload["upstream_chat_url"] = UPSTREAM_CHAT_URL
        self._send_json(200, payload)

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self._send_json(404, {"error": "unsupported_path", "path": self.path})
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length)

        try:
            request_payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            self._send_json(400, {"error": "invalid_json", "detail": str(exc)})
            return

        parsed = urlparse(UPSTREAM_CHAT_URL)
        connection_cls = (
            http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        )
        connection = connection_cls(
            parsed.hostname,
            parsed.port or (443 if parsed.scheme == "https" else 80),
            timeout=600,
        )

        upstream_headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if request_payload.get("stream") else "application/json",
        }
        if UPSTREAM_AUTH:
            upstream_headers["Authorization"] = UPSTREAM_AUTH

        try:
            connection.request(
                "POST",
                parsed.path or "/v1/chat/completions",
                body=raw_body,
                headers=upstream_headers,
            )
            upstream = connection.getresponse()
        except Exception as exc:
            self._send_json(502, {"error": "upstream_connect_failed", "detail": str(exc)})
            return

        if upstream.status >= 400:
            error_body = upstream.read()
            content_type = upstream.getheader("Content-Type", "application/json; charset=utf-8")
            self.send_response(upstream.status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(error_body)))
            self.end_headers()
            self._write_body(error_body)
            return

        if request_payload.get("stream"):
            request_id = make_request_id()
            self._handle_streaming_response(
                request_id=request_id,
                upstream=upstream,
                request_payload=request_payload,
            )
            return

        body_bytes = upstream.read()
        try:
            response_payload = json.loads(body_bytes.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_response(upstream.status)
            self.send_header(
                "Content-Type", upstream.getheader("Content-Type", "application/json; charset=utf-8")
            )
            self.send_header("Content-Length", str(len(body_bytes)))
            self.end_headers()
            self._write_body(body_bytes)
            return

        raw_text = ""
        clean_text = ""
        try:
            message = response_payload["choices"][0]["message"]
            raw_text = message.get("content", "")
            clean_text = raw_text.replace(OPEN_TAG, "").replace(CLOSE_TAG, "")
            message["content"] = clean_text
        except (KeyError, IndexError, TypeError):
            pass

        response_body = json.dumps(response_payload, ensure_ascii=False).encode("utf-8")
        self.send_response(upstream.status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self._write_body(response_body)

    def _handle_streaming_response(
        self,
        request_id,
        upstream,
        request_payload,
    ) -> None:
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        stripper = StreamingTagStripper()
        last_meta = {
            "id": f"chatcmpl-proxy-{request_id}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request_payload.get("model", "unknown"),
        }

        while True:
            line = upstream.readline()
            if not line:
                break

            decoded = line.decode("utf-8", errors="replace")
            if not decoded.startswith("data:"):
                continue

            data = decoded[5:].lstrip()
            if data.strip() == "[DONE]":
                flushed = stripper.flush()
                if flushed:
                    self._write_sse_event(
                        {
                            **last_meta,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": flushed},
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
                self._write_sse_done()
                return

            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue

            if isinstance(event, dict):
                for key in ("id", "object", "created", "model"):
                    if key in event:
                        last_meta[key] = event[key]

            choices = event.get("choices") or []
            if not choices:
                self._write_sse_event(event)
                continue

            choice = choices[0]
            delta = choice.get("delta") or {}
            content = delta.get("content")

            if isinstance(content, str):
                cleaned = stripper.feed(content)
                if cleaned:
                    delta["content"] = cleaned
                else:
                    delta["content"] = ""
                    if "role" not in delta and choice.get("finish_reason") is None:
                        continue

            self._write_sse_event(event)


def main() -> None:
    server = ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), ProxyHandler)
    print(
        f"continue-apply-proxy listening on http://{LISTEN_HOST}:{LISTEN_PORT} "
        f"-> {UPSTREAM_CHAT_URL}"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
