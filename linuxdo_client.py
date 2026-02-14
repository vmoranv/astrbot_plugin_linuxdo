from __future__ import annotations

import asyncio
import json
import time
import urllib.parse
from typing import Any

try:
    from curl_cffi import requests as curl_requests
except Exception:
    curl_requests = None


DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/142.0.0.0 Safari/537.36"
)


class LinuxDoApiError(RuntimeError):
    """Linux.do API 请求异常。"""


class LinuxDoAuthError(LinuxDoApiError):
    """Linux.do 登录态异常。"""


class LinuxDoClient:
    def __init__(
        self,
        base_url: str,
        cookie: str,
        user_agent: str,
        timeout_seconds: int = 20,
        proxy_url: str = "",
        proxy_use_env: bool = True,
        curl_impersonate: str = "chrome124",
        request_retry_on_429: bool = True,
        request_max_retries: int = 1,
        request_retry_backoff_seconds: float = 1.2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.cookie = self._clean_header_value(cookie)
        self.user_agent = self._clean_header_value(user_agent) or DEFAULT_USER_AGENT
        self.timeout_seconds = max(5, int(timeout_seconds))
        self.proxy_url = self._clean_header_value(proxy_url)
        self.proxy_use_env = bool(proxy_use_env)
        self.curl_impersonate = (curl_impersonate or "chrome124").strip()
        self.request_retry_on_429 = bool(request_retry_on_429)
        self.request_max_retries = max(0, int(request_max_retries))
        self.request_retry_backoff_seconds = max(0.1, float(request_retry_backoff_seconds))

    @staticmethod
    def _clean_header_value(value: str | None) -> str:
        if not value:
            return ""
        return " ".join(str(value).replace("\r", " ").replace("\n", " ").split()).strip()

    def is_configured(self) -> bool:
        return bool(self.cookie and self.user_agent)

    async def validate_auth(self) -> tuple[bool, str]:
        if not self.is_configured():
            return False, "未配置 Cookie 或 User-Agent"
        try:
            await self._request_json("/u/current.json")
            return True, "ok"
        except LinuxDoApiError as exc:
            # Mirror linuxdo-explorer: fallback to categories endpoint.
            try:
                await self._request_json("/categories.json")
                return True, "ok (fallback /categories.json)"
            except LinuxDoApiError:
                return False, str(exc)

    async def get_topic(self, topic_id: int) -> dict[str, Any]:
        return await self._request_json(f"/t/{topic_id}.json")

    async def get_current_user(self) -> dict[str, Any]:
        return await self._request_json("/u/current.json")

    async def get_categories(self) -> dict[str, Any]:
        return await self._request_json("/categories.json")

    async def get_binary(self, url_or_path: str) -> bytes:
        """Fetch raw bytes for an URL or site-relative path.

        This is mainly used for downloading topic images so we can forward them as base64 images
        (avoids Cloudflare blocking the downstream platform adapter).
        """
        return await asyncio.to_thread(self._request_bytes_sync, url_or_path)

    async def get_topic_posts_by_ids(
        self,
        topic_id: int,
        post_ids: list[int],
    ) -> list[dict[str, Any]]:
        if not post_ids:
            return []
        params = urllib.parse.urlencode([("post_ids[]", str(pid)) for pid in post_ids])
        payload = await self._request_json(f"/t/{topic_id}/posts.json?{params}")
        post_stream = payload.get("post_stream") or {}
        posts = post_stream.get("posts") or []
        if not isinstance(posts, list):
            return []
        return [p for p in posts if isinstance(p, dict)]

    async def get_latest_topics(self, page: int = 0) -> list[dict[str, Any]]:
        payload = await self._request_json(f"/latest.json?page={page}")
        return self._extract_topics(payload)

    async def get_new_topics(self, page: int = 0) -> list[dict[str, Any]]:
        payload = await self._request_json(f"/new.json?page={page}")
        return self._extract_topics(payload)

    async def get_top_topics(self, period: str = "daily") -> list[dict[str, Any]]:
        payload = await self._request_json(
            f"/top.json?period={urllib.parse.quote(period)}",
        )
        return self._extract_topics(payload)

    @staticmethod
    def _extract_topics(payload: dict[str, Any]) -> list[dict[str, Any]]:
        topic_list = payload.get("topic_list") or {}
        topics = topic_list.get("topics") or []
        if not isinstance(topics, list):
            return []
        return [item for item in topics if isinstance(item, dict)]

    async def _request_json(
        self,
        path: str,
        method: str = "GET",
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(self._request_json_sync, path, method, body)

    def _normalize_url(self, url_or_path: str) -> str:
        raw = (url_or_path or "").strip()
        if not raw:
            return ""
        if raw.startswith(("http://", "https://")):
            return raw
        if raw.startswith("//"):
            scheme = "https" if self.base_url.startswith("https://") else "http"
            return f"{scheme}:{raw}"
        if raw.startswith("/"):
            return f"{self.base_url}{raw}"
        return f"{self.base_url}/{raw}"

    def _request_bytes_sync(self, url_or_path: str) -> bytes:
        if curl_requests is None:
            raise LinuxDoApiError("缺少依赖 curl_cffi，请先安装后再使用插件")

        url = self._normalize_url(url_or_path)
        if not url:
            raise LinuxDoApiError("空的图片 URL")

        headers = self._build_headers(method="GET")
        headers["Accept"] = "image/avif,image/webp,image/apng,image/*,*/*;q=0.8"
        headers["Sec-Fetch-Dest"] = "image"

        attempts = 1 + (self.request_max_retries if self.request_retry_on_429 else 0)
        last_error: LinuxDoApiError | None = None

        for attempt in range(attempts):
            try:
                with curl_requests.Session(impersonate=self.curl_impersonate) as session:
                    session.trust_env = (not self.proxy_url) and self.proxy_use_env
                    if self.proxy_url:
                        session.proxies = {"http": self.proxy_url, "https": self.proxy_url}

                    response = session.request(
                        method="GET",
                        url=url,
                        headers=headers,
                        timeout=self.timeout_seconds,
                        allow_redirects=True,
                        default_headers=False,
                    )

                status_code = int(getattr(response, "status_code", 0))
                if status_code < 200 or status_code >= 300:
                    self._raise_for_status(
                        status_code,
                        str(getattr(response, "text", "") or ""),
                        headers=getattr(response, "headers", {}) or {},
                    )

                content = getattr(response, "content", b"") or b""
                if isinstance(content, bytearray):
                    return bytes(content)
                if isinstance(content, bytes):
                    return content
                return bytes(content)
            except LinuxDoApiError as exc:
                last_error = exc
                if (not self._is_429_error(exc)) or attempt >= attempts - 1:
                    raise
                sleep_seconds = self.request_retry_backoff_seconds * (2**attempt)
                time.sleep(sleep_seconds)
            except Exception as exc:
                raise LinuxDoApiError(f"Linux.do 网络请求失败: {exc}") from exc

        if last_error is not None:
            raise last_error
        raise LinuxDoApiError("Linux.do 请求失败")

    def _request_json_sync(
        self,
        path: str,
        method: str = "GET",
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if curl_requests is None:
            raise LinuxDoApiError("缺少依赖 curl_cffi，请先安装后再使用插件")

        endpoint = path if path.startswith("/") else f"/{path}"
        url = f"{self.base_url}{endpoint}"
        headers = self._build_headers(method=method)

        attempts = 1 + (self.request_max_retries if self.request_retry_on_429 else 0)
        last_error: LinuxDoApiError | None = None

        for attempt in range(attempts):
            try:
                with curl_requests.Session(impersonate=self.curl_impersonate) as session:
                    session.trust_env = (not self.proxy_url) and self.proxy_use_env
                    if self.proxy_url:
                        session.proxies = {"http": self.proxy_url, "https": self.proxy_url}

                    response = session.request(
                        method=method.upper(),
                        url=url,
                        headers=headers,
                        json=body,
                        timeout=self.timeout_seconds,
                        allow_redirects=True,
                        default_headers=False,
                    )

                status_code = int(getattr(response, "status_code", 0))
                text = str(getattr(response, "text", "") or "")
                if status_code < 200 or status_code >= 300:
                    self._raise_for_status(status_code, text, headers=getattr(response, "headers", {}) or {})

                return self._parse_json_text(text)
            except LinuxDoApiError as exc:
                last_error = exc
                if (not self._is_429_error(exc)) or attempt >= attempts - 1:
                    raise
                sleep_seconds = self.request_retry_backoff_seconds * (2**attempt)
                time.sleep(sleep_seconds)
            except Exception as exc:
                raise LinuxDoApiError(f"Linux.do 网络请求失败: {exc}") from exc

        if last_error is not None:
            raise last_error
        raise LinuxDoApiError("Linux.do 请求失败")

    @staticmethod
    def _is_429_error(exc: LinuxDoApiError) -> bool:
        message = str(exc)
        return ("HTTP 429" in message) or ("触发限流" in message)

    @staticmethod
    def _parse_json_text(text: str) -> dict[str, Any]:
        try:
            result = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LinuxDoApiError("Linux.do 返回了非 JSON 数据") from exc

        if not isinstance(result, dict):
            raise LinuxDoApiError("Linux.do JSON 结构异常")
        return result

    def _raise_for_status(self, status_code: int, body_text: str, headers: Any = None):
        body_preview = (body_text or "")[:120]
        if status_code == 429:
            retry_after = ""
            try:
                retry_after = str((headers or {}).get("Retry-After") or "").strip()
            except Exception:
                retry_after = ""
            hint = "Linux.do 触发限流(HTTP 429)，请稍后重试"
            if retry_after:
                hint += f"（Retry-After={retry_after}）"
            details = body_preview.strip()
            if details:
                hint += f": {details}"
            raise LinuxDoApiError(hint)
        if status_code in (401, 403):
            raise LinuxDoAuthError("登录状态无效或已过期，请更新 Cookie/User-Agent")
        raise LinuxDoApiError(f"Linux.do API 请求失败: HTTP {status_code} {body_preview}")

    def _build_headers(self, method: str) -> dict[str, str]:
        is_write = method.upper() in {"POST", "PUT", "DELETE", "PATCH"}
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
            "Accept-Encoding": "gzip, deflate, br",
            "Sec-CH-UA": '"Chromium";v="142", "Microsoft Edge";v="142", "Not_A Brand";v="99"',
            "Sec-CH-UA-Arch": '"x86"',
            "Sec-CH-UA-Bitness": '"64"',
            "Sec-CH-UA-Full-Version": '"142.0.3595.94"',
            "Sec-CH-UA-Full-Version-List": (
                '"Chromium";v="142.0.7444.176", '
                '"Microsoft Edge";v="142.0.3595.94", '
                '"Not_A Brand";v="99.0.0.0"'
            ),
            "Sec-CH-UA-Mobile": "?0",
            "Sec-CH-UA-Model": '""',
            "Sec-CH-UA-Platform": '"Windows"',
            "Sec-CH-UA-Platform-Version": '"19.0.0"',
        }
        if self.cookie:
            headers["Cookie"] = self.cookie

        if is_write:
            headers["Content-Type"] = "application/json; charset=UTF-8"
            headers["X-Requested-With"] = "XMLHttpRequest"
            headers["Origin"] = self.base_url
            headers["Referer"] = f"{self.base_url}/"
            headers["Sec-Fetch-Dest"] = "empty"
            headers["Sec-Fetch-Mode"] = "cors"
            headers["Sec-Fetch-Site"] = "same-origin"
        else:
            headers["Cache-Control"] = "max-age=0"
            headers["Sec-Fetch-Dest"] = "document"
            headers["Sec-Fetch-Mode"] = "navigate"
            headers["Sec-Fetch-Site"] = "same-origin"
            headers["Sec-Fetch-User"] = "?1"
            headers["Upgrade-Insecure-Requests"] = "1"

        return headers
