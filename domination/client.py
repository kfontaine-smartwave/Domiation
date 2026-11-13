"""Thin HTTP client for the DOM-IA-TION agent API (standard library only).

Authentication is the per-seat token shown in the game lobby, sent as a Bearer header.
Documentation of the API: GET <server>/api/agent
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

DEFAULT_URL = "https://ia-domination.yogui.ovh"


class ApiError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(f"HTTP {code}: {message}")
        self.code, self.message = code, message

    @property
    def fatal(self) -> bool:
        """Token refused / seat taken over / game gone: retrying cannot help."""
        return self.code in (401, 403, 404)


class Client:
    def __init__(self, url: str, token: str):
        self.url, self.token = url.rstrip("/"), token

    def call(self, method: str, path: str, body=None, timeout: float = 70):
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, method=method, headers={
            "Authorization": f"Bearer {self.token}", "Content-Type": "application/json",
            "User-Agent": "domination-npu-agent/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            try:
                message = json.loads(exc.read().decode("utf-8")).get("erreur", "")
            except (ValueError, AttributeError):
                message = ""
            raise ApiError(exc.code, message) from None

    def wait(self, timeout: int = 25) -> dict:
        """Long poll: returns as soon as it is my turn (or when `timeout` seconds have passed)."""
        return self.call("GET", f"/api/agent/attendre?timeout={timeout}", timeout=timeout + 45)

    def state(self) -> dict:
        return self.call("GET", "/api/agent/etat")

    def act(self, action: str, **fields) -> dict:
        return self.call("POST", "/api/agent/action", {"action": action, **fields})
