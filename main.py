from __future__ import annotations

import asyncio
import base64
import html
import json
import re
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.star import Context, Star, StarTools, register

try:
    from .linuxdo_client import LinuxDoApiError, LinuxDoAuthError, LinuxDoClient
except ImportError:
    from linuxdo_client import LinuxDoApiError, LinuxDoAuthError, LinuxDoClient


PLUGIN_ID = "astrbot_plugin_linuxdo"
TOPIC_URL_PATTERN = re.compile(r"/t/(?:[^/]+/)?(\d+)")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
IMG_TAG_PATTERN = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
IMG_ATTR_ORIG_PATTERN = re.compile(r"\bdata-orig-src\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)
IMG_ATTR_DATA_PATTERN = re.compile(r"\bdata-src\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)
IMG_ATTR_SRC_PATTERN = re.compile(r"\bsrc\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)
LIGHTBOX_FIGURE_PATTERN = re.compile(
    r"<figure\b[^>]*class=[\"'][^\"']*lightbox[^\"']*[\"'][^>]*>.*?</figure>",
    re.IGNORECASE | re.DOTALL,
)
FIGCAPTION_PATTERN = re.compile(r"<figcaption\b[^>]*>.*?</figcaption>", re.IGNORECASE | re.DOTALL)
MEDIA_TOKEN_PATTERN = re.compile(r"<a\b[^>]*>.*?</a>|<img\b[^>]*>", re.IGNORECASE | re.DOTALL)
A_HREF_PATTERN = re.compile(r"\bhref\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)
A_CLASS_PATTERN = re.compile(r"\bclass\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)
ANCHOR_INNER_PATTERN = re.compile(r"^<a\b[^>]*>(.*)</a>$", re.IGNORECASE | re.DOTALL)
BLOCK_TAG_PATTERN = re.compile(
    r"</?(?:p|div|li|ul|ol|br|blockquote|h[1-6]|figure|figcaption|hr)\b[^>]*>",
    re.IGNORECASE,
)
MARKDOWN_IMAGE_PATTERN = re.compile(r"!\[[^\]]*]\([^)]+\)")
IMAGE_META_PATTERN = re.compile(
    r"\b[\w.-]{4,}\s+\d{2,5}×\d{2,5}\s+\d+(?:\.\d+)?\s*(?:KB|MB|GB)\b",
    re.IGNORECASE,
)
IMAGE_SIZE_ONLY_PATTERN = re.compile(
    r"\b\d{2,5}×\d{2,5}\s+\d+(?:\.\d+)?\s*(?:KB|MB|GB)\b",
    re.IGNORECASE,
)


@dataclass
class GroupSubscription:
    tags: set[str] = field(default_factory=set)
    last_seen_topic_id: int = 0
    announced_topic_ids: set[int] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tags": sorted(self.tags),
            "last_seen_topic_id": self.last_seen_topic_id,
            "announced_topic_ids": sorted(self.announced_topic_ids),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GroupSubscription":
        tags = payload.get("tags") or []
        announced = payload.get("announced_topic_ids") or []
        return cls(
            tags={str(tag).strip() for tag in tags if str(tag).strip()},
            last_seen_topic_id=max(0, int(payload.get("last_seen_topic_id", 0))),
            announced_topic_ids={int(x) for x in announced if str(x).isdigit()},
        )

    def snapshot(self) -> "GroupSubscription":
        return GroupSubscription(
            tags=set(self.tags),
            last_seen_topic_id=self.last_seen_topic_id,
            announced_topic_ids=set(self.announced_topic_ids),
        )


@register(
    "astrbot_plugin_linuxdo",
    "vmoranv",
    "Linux.do 帖子查询与群聊订阅推送",
    "1.1.6",
    "https://github.com/vmoranv/astrbot_plugin_linuxdo",
)
class LinuxDoPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context, config)
        self.config = config or {}
        self._subscriptions: dict[str, GroupSubscription] = {}
        self._regex_issue_cache: set[str] = set()
        self._state_lock = asyncio.Lock()
        self._stop_event = asyncio.Event()
        self._poll_task: asyncio.Task[None] | None = None
        self._state_file = self._resolve_state_file()
        self._load_state()

    async def initialize(self):
        self._stop_event.clear()
        if self._poll_task is None or self._poll_task.done():
            self._poll_task = asyncio.create_task(self._subscription_loop())
        logger.info("[linuxdo] 插件已初始化")

    async def terminate(self):
        self._stop_event.set()
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        self._poll_task = None
        logger.info("[linuxdo] 插件已销毁")

    @filter.command("ldo")
    async def ldo(self, event: AstrMessageEvent, topic_input: str):
        """/ldo <url|帖子id> 返回帖子一楼"""
        topic_id = self._parse_topic_id(topic_input)
        if topic_id <= 0:
            yield event.plain_result("参数错误：请输入有效的帖子 URL 或帖子 ID。")
            return

        client = self._build_client()
        ok, msg = await self._ensure_auth(client)
        if not ok:
            yield event.plain_result(msg)
            return

        try:
            topic = await client.get_topic(topic_id)
            posts = await self._collect_topic_posts(topic_id, topic, required_count=1, client=client)
        except LinuxDoApiError as exc:
            yield event.plain_result(f"获取帖子失败：{exc}")
            return

        if not posts:
            yield event.plain_result("该帖子暂无可读内容。")
            return

        first_post = posts[0]
        title = str(topic.get("title") or "(无标题)")
        username = str(first_post.get("username") or "unknown")
        created_at = str(first_post.get("created_at") or "")
        header_lines = [
            f"【{title}】",
            f"链接：{self._topic_url(topic_id)}",
            f"一楼作者：@{username} {created_at}".rstrip(),
        ]

        max_images = self._conf_int("post_image_max_count", 6, 0, 20)
        max_bytes = self._conf_int("post_image_max_bytes_kb", 4096, 128, 20480) * 1024
        text_limit = self._conf_int("post_excerpt_length", 1200, 200, 5000)
        content_parts = self._extract_post_content_parts(first_post, text_limit, max_images)
        if not content_parts:
            content_parts = [("text", "(无正文)")]

        result = event.make_result().message("\n".join(header_lines) + "\n")
        failed_urls: list[str] = []
        sent_images = 0
        for part_type, payload in content_parts:
            if part_type == "text":
                text_piece = payload.strip()
                if not text_piece:
                    continue
                result.message("\n" + text_piece)
                continue

            image_url = payload
            if sent_images >= max_images:
                continue
            try:
                image_bytes = await client.get_binary(image_url)
            except LinuxDoApiError as exc:
                logger.warning(f"[linuxdo] 拉取帖子图片失败: {exc} url={image_url}")
                failed_urls.append(image_url)
                continue
            if not image_bytes:
                failed_urls.append(image_url)
                continue
            if len(image_bytes) > max_bytes:
                logger.warning(
                    f"[linuxdo] 跳过超大图片: size={len(image_bytes) // 1024}KB "
                    f"limit={max_bytes // 1024}KB url={image_url}",
                )
                failed_urls.append(image_url)
                continue

            encoded = base64.b64encode(image_bytes).decode("ascii")
            result.message("\n")
            result.base64_image(encoded)
            sent_images += 1

        if failed_urls:
            remain = failed_urls[: max(1, min(3, len(failed_urls)))]
            result.message("\n部分图片发送失败：\n" + "\n".join(remain))
        result.squash_plain()
        yield result

    @filter.command("ldo_trend")
    async def ldo_trend(self, event: AstrMessageEvent):
        """/ldo_trend 返回热门帖子"""
        client = self._build_client()
        ok, msg = await self._ensure_auth(client)
        if not ok:
            yield event.plain_result(msg)
            return

        trend_count = self._conf_int("trend_return_count", 10, 1, 30)
        period = self._conf_str("trend_period", "daily")

        try:
            topics = await client.get_top_topics(period=period)
            if not topics:
                topics = await client.get_latest_topics(page=0)
        except LinuxDoApiError as exc:
            yield event.plain_result(f"获取热门帖子失败：{exc}")
            return

        if not topics:
            yield event.plain_result("暂无热门帖子数据。")
            return

        lines = [f"Linux.do 热门帖子（period={period}，前 {trend_count} 条）"]
        for idx, topic in enumerate(topics[:trend_count], start=1):
            lines.append(self._format_topic_line(topic, idx))
        yield event.plain_result("\n".join(lines))

    @filter.command("ldo_auth_check")
    async def ldo_auth_check(self, event: AstrMessageEvent):
        """/ldo_auth_check 检查登录态与接口连通性"""
        client = self._build_client()
        if not client.is_configured():
            yield event.plain_result("未配置 Cookie/User-Agent。")
            return

        checks: list[tuple[str, str]] = []
        proxy_url = self._conf_str("proxy_url", "")
        proxy_use_env = self._conf_bool("proxy_use_env", True)
        curl_impersonate = self._conf_str("curl_impersonate", "chrome124")

        async def run_check(name: str, coro):
            try:
                await coro
                checks.append((name, "OK"))
            except LinuxDoApiError as exc:
                checks.append((name, f"FAIL: {exc}"))

        await run_check("u/current.json", client.get_current_user())
        await run_check("categories.json", client.get_categories())
        await run_check("top.json?period=daily", client.get_top_topics(period="daily"))

        check_map = {name: result for name, result in checks}
        current_result = check_map.get("u/current.json", "")
        categories_ok = check_map.get("categories.json", "").startswith("OK")
        top_ok = check_map.get("top.json?period=daily", "").startswith("OK")
        current_limited = ("HTTP 429" in current_result) or ("触发限流" in current_result)

        lines = ["Linux.do 鉴权诊断："]
        lines.append("- request_backend: curl_cffi (fixed)")
        lines.append(f"- curl_impersonate: {curl_impersonate}")
        lines.append(f"- proxy_url: {proxy_url or '(empty)'}")
        lines.append(f"- proxy_use_env: {proxy_use_env}")
        for name, result in checks:
            lines.append(f"- {name}: {result}")
        if current_limited and categories_ok and top_ok:
            lines.append("- conclusion: u/current.json 被限流，但读取接口可用（可继续使用 /ldo 与 /ldo_trend）。")
        elif categories_ok and top_ok:
            lines.append("- conclusion: 核心读取接口可用。")
        yield event.plain_result("\n".join(lines))

    @filter.command("ldo_comment")
    async def ldo_comment(self, event: AstrMessageEvent, topic_input: str, num: int = 0):
        """/ldo_comment <url|帖子id> <num> 返回帖子评论"""
        topic_id = self._parse_topic_id(topic_input)
        if topic_id <= 0:
            yield event.plain_result("参数错误：请输入有效的帖子 URL 或帖子 ID。")
            return

        target_count = num if num > 0 else self._conf_int("comment_return_count", 3, 1, 20)
        target_count = max(1, min(target_count, 20))

        client = self._build_client()
        ok, msg = await self._ensure_auth(client)
        if not ok:
            yield event.plain_result(msg)
            return

        try:
            topic = await client.get_topic(topic_id)
            posts = await self._collect_topic_posts(
                topic_id=topic_id,
                topic=topic,
                required_count=target_count + 1,
                client=client,
            )
        except LinuxDoApiError as exc:
            yield event.plain_result(f"获取评论失败：{exc}")
            return

        title = str(topic.get("title") or "(无标题)")
        comments = [post for post in posts if int(post.get("post_number", 0)) >= 2]
        if not comments:
            yield event.plain_result(f"【{title}】目前没有评论。")
            return

        comment_excerpt_length = self._conf_int("comment_excerpt_length", 260, 80, 2000)
        selected = comments[:target_count]
        lines = [f"【{title}】前 {len(selected)} 条评论：", f"链接：{self._topic_url(topic_id)}", ""]
        for post in selected:
            floor = int(post.get("post_number", 0))
            user = str(post.get("username") or "unknown")
            body = self._extract_post_text(post, comment_excerpt_length)
            lines.append(f"#{floor} @{user}")
            lines.append(body)
            lines.append("")
        yield event.plain_result("\n".join(lines).rstrip())

    @filter.command("ldo_sub")
    async def ldo_sub(self, event: AstrMessageEvent, tag: str):
        """/ldo_sub <tag> 在当前群订阅关键词"""
        if event.is_private_chat():
            yield event.plain_result("请在群聊里使用 `/ldo_sub <tag>`。")
            return

        normalized_tag = self._normalize_tag(tag)
        if not normalized_tag:
            yield event.plain_result("参数错误：tag 不能为空。")
            return

        client = self._build_client()
        ok, msg = await self._ensure_auth(client)
        if not ok:
            yield event.plain_result(msg)
            return

        baseline_topic_id = await self._get_latest_topic_id(client)
        session = event.unified_msg_origin

        async with self._state_lock:
            sub = self._subscriptions.get(session)
            if sub is None:
                sub = GroupSubscription(last_seen_topic_id=baseline_topic_id)
                self._subscriptions[session] = sub

            existed = normalized_tag in sub.tags
            sub.tags.add(normalized_tag)
            if sub.last_seen_topic_id <= 0:
                sub.last_seen_topic_id = baseline_topic_id
            self._save_state_locked()

            tags_text = ", ".join(sorted(sub.tags))
            interval = self._conf_int("ldo_update_interval_seconds", 300, 10, 86400)

        if existed:
            yield event.plain_result(
                f"tag `{normalized_tag}` 已在当前群订阅中。\n当前订阅：{tags_text}\n检查间隔：{interval}s",
            )
        else:
            yield event.plain_result(
                f"已订阅 tag `{normalized_tag}`。\n当前订阅：{tags_text}\n检查间隔：{interval}s",
            )

    @filter.command("ldo_unsub")
    async def ldo_unsub(self, event: AstrMessageEvent, tag: str = ""):
        """/ldo_unsub [tag] 取消当前群订阅"""
        session = event.unified_msg_origin
        normalized_tag = self._normalize_tag(tag)
        response_text = ""

        async with self._state_lock:
            sub = self._subscriptions.get(session)
            if not sub:
                response_text = "当前群没有任何订阅。"
            elif normalized_tag:
                if normalized_tag not in sub.tags:
                    response_text = f"当前群未订阅 `{normalized_tag}`。"
                else:
                    sub.tags.discard(normalized_tag)
                    if not sub.tags:
                        del self._subscriptions[session]
                        self._save_state_locked()
                        response_text = "已移除该 tag，当前群已无订阅。"
                    else:
                        self._save_state_locked()
                        response_text = (
                            f"已取消 `{normalized_tag}`。\n剩余订阅：{', '.join(sorted(sub.tags))}"
                        )
            else:
                del self._subscriptions[session]
                self._save_state_locked()
                response_text = "已清空当前群的全部订阅。"

        yield event.plain_result(response_text)

    @filter.command("ldo_sub_list")
    async def ldo_sub_list(self, event: AstrMessageEvent):
        """/ldo_sub_list 查看当前群订阅"""
        session = event.unified_msg_origin
        response_text = ""
        async with self._state_lock:
            sub = self._subscriptions.get(session)
            if not sub or not sub.tags:
                response_text = "当前群没有订阅 tag。"
            else:
                tags_text = ", ".join(sorted(sub.tags))
                response_text = (
                    "当前群订阅：\n"
                    f"- tags: {tags_text}\n"
                    f"- last_seen_topic_id: {sub.last_seen_topic_id}\n"
                    f"- check interval: {self._conf_int('ldo_update_interval_seconds', 300, 10, 86400)}s\n"
                    f"- match_title_for_sub: {self._conf_bool('match_title_for_sub', True)}\n"
                    f"- subscription_match_title_only: {self._conf_bool('subscription_match_title_only', False)}\n"
                    f"- subscription_enable_regex: {self._conf_bool('subscription_enable_regex', True)}"
                )

        yield event.plain_result(response_text)

    async def _subscription_loop(self):
        await asyncio.sleep(3)
        while not self._stop_event.is_set():
            try:
                await self._check_subscriptions_once()
            except Exception as exc:
                logger.error(f"[linuxdo] 订阅检查异常: {exc}")

            interval = self._conf_int("ldo_update_interval_seconds", 300, 10, 86400)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                continue

    async def _check_subscriptions_once(self):
        if not self._conf_bool("enable_subscription_push", True):
            return

        async with self._state_lock:
            if not self._subscriptions:
                return
            snapshot = {k: v.snapshot() for k, v in self._subscriptions.items()}
            min_last_seen = min((sub.last_seen_topic_id for sub in snapshot.values()), default=0)

        client = self._build_client()
        if not client.is_configured():
            logger.warning("[linuxdo] 未配置 Cookie/User-Agent，跳过订阅检查")
            return

        scan_count = self._conf_int("subscription_scan_count", 40, 5, 200)
        scan_pages = self._conf_int("subscription_scan_pages", 3, 1, 10)
        trend_period = self._conf_str("trend_period", "daily")

        topics: list[dict[str, Any]] = []
        page = 0
        used_single_page_fallback = False
        while page < scan_pages and len(topics) < scan_count:
            batch, source = await self._fetch_subscription_topic_batch(
                client=client,
                page=page,
                trend_period=trend_period,
            )
            if not batch:
                break
            topics.extend(batch)
            if source in {"top", "new"}:
                used_single_page_fallback = True

            # If we already fetched topics older-or-equal than the earliest last_seen among groups,
            # it's safe to stop paging.
            if min_last_seen > 0:
                batch_min_id = min(int(t.get("id", 0)) for t in batch)
                if batch_min_id <= min_last_seen:
                    break

            if used_single_page_fallback:
                break
            page += 1

        # Deduplicate by topic id while preserving the original order.
        seen: set[int] = set()
        deduped: list[dict[str, Any]] = []
        for t in topics:
            tid = int(t.get("id", 0))
            if tid <= 0 or tid in seen:
                continue
            seen.add(tid)
            deduped.append(t)
        topics = deduped[:scan_count]
        if not topics:
            return

        max_topic_id = max(int(topic.get("id", 0)) for topic in topics)
        min_topic_id = min(int(topic.get("id", 0)) for topic in topics)
        updates: dict[str, dict[str, Any]] = {}
        sorted_topics = sorted(topics, key=lambda item: int(item.get("id", 0)))
        match_title = self._conf_bool("match_title_for_sub", True)
        title_only = self._conf_bool("subscription_match_title_only", False)
        regex_enabled = self._conf_bool("subscription_enable_regex", True)
        if title_only:
            match_title = True

        for session, sub in snapshot.items():
            pushed_topic_ids: list[int] = []
            last_seen = sub.last_seen_topic_id
            for topic in sorted_topics:
                topic_id = int(topic.get("id", 0))
                if topic_id <= last_seen:
                    continue
                if topic_id in sub.announced_topic_ids:
                    continue

                matched_tags = self._match_topic(
                    topic,
                    sub.tags,
                    match_title=match_title,
                    title_only=title_only,
                    regex_enabled=regex_enabled,
                )
                if not matched_tags:
                    continue

                pushed = await self._send_subscription_hit(
                    session=session,
                    topic=topic,
                    matched_tags=matched_tags,
                )
                if pushed:
                    pushed_topic_ids.append(topic_id)

            # Only advance last_seen if we paged far enough back (otherwise we may miss older topics).
            can_advance = (min_topic_id <= last_seen) if last_seen > 0 else True
            updates[session] = {
                "last_seen_topic_id": (max(last_seen, max_topic_id) if can_advance else last_seen),
                "can_advance": can_advance,
                "pushed_topic_ids": pushed_topic_ids,
            }

        async with self._state_lock:
            changed = False
            for session, update in updates.items():
                sub = self._subscriptions.get(session)
                if not sub:
                    continue
                new_last_seen = int(update["last_seen_topic_id"])
                if new_last_seen > sub.last_seen_topic_id:
                    sub.last_seen_topic_id = new_last_seen
                    changed = True

                pushed_ids = update["pushed_topic_ids"]
                if pushed_ids:
                    before = len(sub.announced_topic_ids)
                    sub.announced_topic_ids.update(pushed_ids)
                    if len(sub.announced_topic_ids) > 500:
                        sub.announced_topic_ids = set(sorted(sub.announced_topic_ids)[-500:])
                    changed = changed or len(sub.announced_topic_ids) != before

            if changed:
                self._save_state_locked()

        # If we couldn't advance for some group, it means scan_count/pages is too small for the gap.
        if any(not u.get("can_advance", True) for u in updates.values()):
            logger.warning(
                "[linuxdo] 订阅扫描范围不足，可能会漏推送。"
                "请适当增大 subscription_scan_count/subscription_scan_pages，或缩短 ldo_update_interval_seconds。",
            )

    async def _send_subscription_hit(
        self,
        session: str,
        topic: dict[str, Any],
        matched_tags: list[str],
    ) -> bool:
        topic_id = int(topic.get("id", 0))
        title = str(topic.get("title") or "(无标题)")
        tags = ", ".join(str(t) for t in (topic.get("tags") or []))
        tags_info = tags if tags else "(无标签)"
        message = (
            "[Linux.do 订阅命中]\n"
            f"关键词：{', '.join(matched_tags)}\n"
            f"标题：{title}\n"
            f"标签：{tags_info}\n"
            f"链接：{self._topic_url(topic_id)}"
        )
        return await self.context.send_message(session, MessageChain().message(message))

    def _match_topic(
        self,
        topic: dict[str, Any],
        tags: set[str],
        match_title: bool,
        title_only: bool,
        regex_enabled: bool,
    ) -> list[str]:
        topic_tags_raw = [str(tag) for tag in (topic.get("tags") or [])]
        topic_tags_lower = [one.lower() for one in topic_tags_raw]
        title_raw = str(topic.get("title") or "")
        title_lower = title_raw.lower()
        matched: list[str] = []
        for tag in sorted(tags):
            if self._match_tag_once(
                tag=tag,
                topic_tags_raw=topic_tags_raw,
                topic_tags_lower=topic_tags_lower,
                title_raw=title_raw,
                title_lower=title_lower,
                match_title=match_title,
                title_only=title_only,
                regex_enabled=regex_enabled,
            ):
                matched.append(tag)
        return matched

    def _match_tag_once(
        self,
        tag: str,
        topic_tags_raw: list[str],
        topic_tags_lower: list[str],
        title_raw: str,
        title_lower: str,
        match_title: bool,
        title_only: bool,
        regex_enabled: bool,
    ) -> bool:
        is_regex, regex = self._build_regex_matcher(tag, regex_enabled)
        if is_regex:
            if regex is None:
                return False
            if title_only:
                return bool(regex.search(title_raw))
            if any(regex.search(one_tag) for one_tag in topic_tags_raw):
                return True
            return bool(match_title and regex.search(title_raw))

        needle = tag.lower()
        if title_only:
            return bool(match_title and needle in title_lower)
        if needle in topic_tags_lower or any(needle in one_tag for one_tag in topic_tags_lower):
            return True
        return bool(match_title and needle in title_lower)

    def _build_regex_matcher(
        self,
        tag: str,
        regex_enabled: bool,
    ) -> tuple[bool, re.Pattern[str] | None]:
        if not regex_enabled:
            return False, None

        prefix = self._conf_str("subscription_regex_prefix", "re:")
        if not prefix:
            prefix = "re:"
        raw = (tag or "").strip()
        if not raw.lower().startswith(prefix.lower()):
            return False, None

        pattern_text = raw[len(prefix):].strip()
        if not pattern_text:
            return True, None

        max_length = self._conf_int("subscription_regex_max_length", 120, 10, 1000)
        if len(pattern_text) > max_length:
            self._warn_regex_once(
                raw,
                f"[linuxdo] 订阅正则长度超限({len(pattern_text)}>{max_length})，已跳过: {raw}",
            )
            return True, None

        flags = re.IGNORECASE if self._conf_bool("subscription_regex_ignore_case", True) else 0
        try:
            return True, re.compile(pattern_text, flags)
        except re.error as exc:
            self._warn_regex_once(raw, f"[linuxdo] 无效订阅正则 `{raw}`: {exc}")
            return True, None

    def _warn_regex_once(self, tag: str, message: str):
        if tag in self._regex_issue_cache:
            return
        self._regex_issue_cache.add(tag)
        logger.warning(message)

    async def _collect_topic_posts(
        self,
        topic_id: int,
        topic: dict[str, Any],
        required_count: int,
        client: LinuxDoClient,
    ) -> list[dict[str, Any]]:
        post_stream = topic.get("post_stream") or {}
        posts = list(post_stream.get("posts") or [])
        if len(posts) >= required_count:
            return sorted(posts, key=lambda p: int(p.get("post_number", 0)))

        stream = post_stream.get("stream") or []
        stream_ids = [int(pid) for pid in stream if str(pid).isdigit()]
        exists = {int(post.get("id", 0)) for post in posts if post.get("id")}
        missing = [pid for pid in stream_ids if pid not in exists]
        if missing:
            remain = max(0, required_count - len(posts))
            extra_posts = await client.get_topic_posts_by_ids(topic_id=topic_id, post_ids=missing[:remain])
            posts.extend(extra_posts)

        unique: dict[int, dict[str, Any]] = {}
        for post in posts:
            post_id = int(post.get("id", 0))
            if post_id > 0:
                unique[post_id] = post
        ordered = sorted(unique.values(), key=lambda p: int(p.get("post_number", 0)))
        return ordered

    async def _ensure_auth(self, client: LinuxDoClient) -> tuple[bool, str]:
        if client.is_configured():
            return True, ""
        hint = "Linux.do 登录凭证未配置。请在插件配置中填写 `cookie` 与 `user_agent`。"
        return False, hint

    async def _fetch_subscription_topic_batch(
        self,
        client: LinuxDoClient,
        page: int,
        trend_period: str,
    ) -> tuple[list[dict[str, Any]], str]:
        """Fetch one subscription scan page.

        Preferred order:
        1) latest.json?page=n
        2) top.json?period=... (only page=0)
        3) new.json?page=0 (only page=0)
        """
        try:
            return await client.get_latest_topics(page=page), "latest"
        except LinuxDoApiError as exc_latest:
            if page > 0:
                logger.warning(f"[linuxdo] latest.json 第 {page} 页抓取失败: {exc_latest}")
                return [], "latest"

            try:
                logger.warning(f"[linuxdo] latest.json 抓取失败，回退 top.json: {exc_latest}")
                return await client.get_top_topics(period=trend_period), "top"
            except LinuxDoApiError as exc_top:
                logger.warning(f"[linuxdo] top.json 抓取失败，回退 new.json: {exc_top}")
                try:
                    return await client.get_new_topics(page=0), "new"
                except LinuxDoApiError as exc_new:
                    logger.warning(f"[linuxdo] new.json 也抓取失败，订阅扫描跳过: {exc_new}")
                    return [], "new"

    async def _get_latest_topic_id(self, client: LinuxDoClient) -> int:
        topics: list[dict[str, Any]] = []
        errors: list[str] = []
        for source in ("latest", "top", "new"):
            try:
                if source == "latest":
                    topics = await client.get_latest_topics(page=0)
                elif source == "top":
                    topics = await client.get_top_topics(period=self._conf_str("trend_period", "daily"))
                else:
                    topics = await client.get_new_topics(page=0)
                if topics:
                    break
            except LinuxDoApiError as exc:
                errors.append(f"{source}: {exc}")
                continue
        if not topics:
            if errors:
                logger.warning(f"[linuxdo] 获取最新帖子失败: {' | '.join(errors)}")
            return 0
        return max(int(topic.get("id", 0)) for topic in topics)

    def _build_client(self) -> LinuxDoClient:
        return LinuxDoClient(
            base_url=self._conf_str("base_url", "https://linux.do"),
            cookie=self._conf_str("cookie", ""),
            user_agent=self._conf_str("user_agent", ""),
            timeout_seconds=self._conf_int("request_timeout_seconds", 20, 5, 120),
            proxy_url=self._conf_str("proxy_url", ""),
            proxy_use_env=self._conf_bool("proxy_use_env", True),
            curl_impersonate=self._conf_str("curl_impersonate", "chrome124"),
            request_retry_on_429=self._conf_bool("request_retry_on_429", True),
            request_max_retries=self._conf_int("request_max_retries", 1, 0, 10),
            request_retry_backoff_seconds=float(
                self._conf_int("request_retry_backoff_seconds", 2, 1, 30),
            ),
        )

    def _resolve_state_file(self) -> Path:
        try:
            data_dir = StarTools.get_data_dir(PLUGIN_ID)
        except Exception:
            data_dir = Path(__file__).resolve().parent / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir / "subscriptions.json"

    def _load_state(self):
        self._subscriptions = {}
        if not self._state_file.exists():
            return
        try:
            payload = json.loads(self._state_file.read_text(encoding="utf-8"))
            raw_subscriptions = payload.get("subscriptions", {})
            if not isinstance(raw_subscriptions, dict):
                return
            for session, item in raw_subscriptions.items():
                if not isinstance(item, dict):
                    continue
                self._subscriptions[session] = GroupSubscription.from_dict(item)
        except Exception as exc:
            logger.warning(f"[linuxdo] 读取订阅状态失败: {exc}")

    def _save_state_locked(self):
        payload = {
            "subscriptions": {
                session: sub.to_dict() for session, sub in self._subscriptions.items()
            },
        }
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            self._state_file.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning(f"[linuxdo] 写入订阅状态失败: {exc}")

    def _parse_topic_id(self, topic_input: str) -> int:
        raw = (topic_input or "").strip()
        if not raw:
            return 0
        if raw.isdigit():
            return int(raw)

        parsed = urllib.parse.urlparse(raw)
        target = parsed.path or raw
        match = TOPIC_URL_PATTERN.search(target)
        if not match:
            match = TOPIC_URL_PATTERN.search(raw)
        if not match:
            return 0
        return int(match.group(1))

    def _topic_url(self, topic_id: int) -> str:
        base_url = self._conf_str("base_url", "https://linux.do").rstrip("/")
        return f"{base_url}/t/{topic_id}"

    def _normalize_tag(self, tag: str) -> str:
        raw = (tag or "").strip()
        if not raw:
            return ""

        prefix = self._conf_str("subscription_regex_prefix", "re:")
        if not prefix:
            prefix = "re:"
        if raw.lower().startswith(prefix.lower()):
            pattern = raw[len(prefix):].strip()
            if not pattern:
                return ""
            return f"{prefix}{pattern}"

        return raw.lower()

    def _extract_post_text(self, post: dict[str, Any], limit: int) -> str:
        text = ""
        cooked = str(post.get("cooked") or "")
        if cooked:
            cooked_slim = LIGHTBOX_FIGURE_PATTERN.sub(" ", cooked)
            cooked_slim = IMG_TAG_PATTERN.sub(" ", cooked_slim)
            cooked_slim = HTML_TAG_PATTERN.sub(" ", cooked_slim)
            text = html.unescape(" ".join(cooked_slim.split()))

        if not text:
            raw = str(post.get("raw") or "")
            if raw:
                raw = MARKDOWN_IMAGE_PATTERN.sub(" ", raw)
                text = " ".join(raw.split())

        if text:
            # Remove leftover image metadata strings like:
            # IMG_20260211_125649 1920×2560 1.47 MB
            text = IMAGE_META_PATTERN.sub(" ", text)
            text = IMAGE_SIZE_ONLY_PATTERN.sub(" ", text)
            text = " ".join(text.split())
        if not text:
            text = "(无正文)"
        if len(text) > limit:
            text = text[: max(1, limit - 3)] + "..."
        return text

    def _extract_post_content_parts(
        self,
        post: dict[str, Any],
        text_limit: int,
        max_images: int,
    ) -> list[tuple[str, str]]:
        """Return ordered (type, payload) parts for the first post.

        - type == "text": payload is plain text (may contain newlines)
        - type == "image": payload is normalized image url

        This uses `cooked` HTML so we can insert images in the correct position.
        """
        cooked = str(post.get("cooked") or "")
        if not cooked:
            text = self._extract_post_text(post, text_limit)
            return [("text", text)] if text else []

        if max_images <= 0:
            text = self._extract_post_text(post, text_limit)
            return [("text", text)] if text else []

        tokens: list[tuple[str, str]] = []
        pos = 0
        for match in MEDIA_TOKEN_PATTERN.finditer(cooked):
            if match.start() > pos:
                tokens.append(("html", cooked[pos:match.start()]))
            token_html = match.group(0)
            extracted = self._extract_media_url_from_token(token_html)
            if extracted:
                tokens.append(("image", extracted))
            pos = match.end()
        if pos < len(cooked):
            tokens.append(("html", cooked[pos:]))

        parts: list[tuple[str, str]] = []
        used_text = 0
        for kind, payload in tokens:
            if kind == "image":
                # Keep all image tokens; enforcement happens at send time.
                parts.append(("image", payload))
                continue

            text_piece = self._html_fragment_to_text(payload)
            if not text_piece:
                continue

            remaining = max(0, text_limit - used_text)
            if remaining <= 0:
                break
            if len(text_piece) > remaining:
                text_piece = text_piece[: max(1, remaining - 3)] + "..."
                parts.append(("text", text_piece))
                break

            parts.append(("text", text_piece))
            used_text += len(text_piece)

        # Merge adjacent text parts for cleaner chains.
        merged: list[tuple[str, str]] = []
        buffer: list[str] = []
        for kind, payload in parts:
            if kind == "text":
                buffer.append(payload)
                continue
            if buffer:
                merged.append(("text", "\n".join(buffer)))
                buffer = []
            merged.append((kind, payload))
        if buffer:
            merged.append(("text", "\n".join(buffer)))
        return merged

    def _extract_image_url_from_img_tag(self, tag_text: str) -> str:
        for pattern in (IMG_ATTR_ORIG_PATTERN, IMG_ATTR_DATA_PATTERN, IMG_ATTR_SRC_PATTERN):
            m = pattern.search(tag_text)
            if m:
                return m.group(1).strip()
        return ""

    def _extract_media_url_from_token(self, token_html: str) -> str:
        raw = (token_html or "").strip()
        if not raw:
            return ""
        lowered = raw.lower()
        if lowered.startswith("<img"):
            candidate = self._extract_image_url_from_img_tag(raw)
            if not candidate:
                return ""
            normalized = self._normalize_media_url(html.unescape(candidate))
            if normalized and self._should_keep_image_url(normalized):
                return normalized
            return ""

        if not lowered.startswith("<a"):
            return ""

        # Only treat lightbox/attachment anchors as images.
        class_match = A_CLASS_PATTERN.search(raw)
        class_text = (class_match.group(1) if class_match else "").lower()
        href_match = A_HREF_PATTERN.search(raw)
        href = (href_match.group(1).strip() if href_match else "")
        if not href:
            return ""

        normalized = self._normalize_media_url(html.unescape(href))
        if not normalized:
            return ""

        # 1) Explicit image-ish anchor classes.
        if ("lightbox" in class_text or "attachment" in class_text) and self._should_keep_image_url(normalized):
            return normalized

        # 2) Attachment-like text (e.g. "460×260 26.2 KB") with uploads URL.
        inner_text = self._anchor_inner_text(raw)
        looks_like_image_meta = bool(
            IMAGE_META_PATTERN.search(inner_text) or IMAGE_SIZE_ONLY_PATTERN.search(inner_text),
        )
        if looks_like_image_meta and "/uploads/" in normalized.lower():
            return normalized

        # 3) Plain anchor directly linking to image files.
        if self._should_keep_image_url(normalized):
            return normalized
        return ""

    def _anchor_inner_text(self, anchor_tag: str) -> str:
        m = ANCHOR_INNER_PATTERN.match(anchor_tag or "")
        if not m:
            return ""
        inner = m.group(1)
        inner = HTML_TAG_PATTERN.sub(" ", inner)
        inner = html.unescape(" ".join(inner.split()))
        return inner.strip()

    def _html_fragment_to_text(self, fragment: str) -> str:
        raw = (fragment or "").strip()
        if not raw:
            return ""

        # Drop captions which are typically just filenames/sizes for attachments.
        raw = FIGCAPTION_PATTERN.sub(" ", raw)
        # Replace attachment anchors with their inner text (often filename) so we can later strip meta lines.
        m = ANCHOR_INNER_PATTERN.match(raw)
        if m:
            raw = m.group(1)
        # Convert common block-level tags to newlines before stripping tags.
        raw = BLOCK_TAG_PATTERN.sub("\n", raw)
        # Remove remaining tags.
        raw = HTML_TAG_PATTERN.sub(" ", raw)
        raw = html.unescape(raw)
        raw = IMAGE_META_PATTERN.sub(" ", raw)
        raw = IMAGE_SIZE_ONLY_PATTERN.sub(" ", raw)

        raw = raw.replace("\r\n", "\n").replace("\r", "\n")
        raw = re.sub(r"[ \t\f\v]+", " ", raw)
        # Strip each line but keep newlines.
        lines = [line.strip() for line in raw.split("\n")]
        raw = "\n".join(lines)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()

    def _extract_post_image_urls(self, post: dict[str, Any]) -> list[str]:
        cooked = str(post.get("cooked") or "")
        if not cooked:
            return []
        urls: list[str] = []
        seen: set[str] = set()
        for tag_match in IMG_TAG_PATTERN.finditer(cooked):
            tag_text = tag_match.group(0)
            candidate = ""
            for pattern in (IMG_ATTR_ORIG_PATTERN, IMG_ATTR_DATA_PATTERN, IMG_ATTR_SRC_PATTERN):
                m = pattern.search(tag_text)
                if m:
                    candidate = m.group(1).strip()
                    break
            if not candidate:
                continue

            normalized = self._normalize_media_url(html.unescape(candidate))
            if not normalized:
                continue
            if not self._should_keep_image_url(normalized):
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            urls.append(normalized)
        return urls

    def _normalize_media_url(self, value: str) -> str:
        raw = (value or "").strip()
        if not raw:
            return ""
        if raw.startswith(("http://", "https://")):
            return raw
        if raw.startswith("//"):
            base_url = self._conf_str("base_url", "https://linux.do")
            scheme = "https" if base_url.startswith("https://") else "http"
            return f"{scheme}:{raw}"
        if raw.startswith("/"):
            return f"{self._conf_str('base_url', 'https://linux.do').rstrip('/')}{raw}"
        return f"{self._conf_str('base_url', 'https://linux.do').rstrip('/')}/{raw}"

    def _should_keep_image_url(self, url: str) -> bool:
        lowered = (url or "").strip().lower()
        if not lowered:
            return False
        if lowered.startswith("data:"):
            return False
        # Avoid avatars/emojis and other UI icons.
        if "user_avatar" in lowered:
            return False
        if ("/emoji/" in lowered or "/images/emoji/" in lowered) and ("/uploads/" not in lowered):
            return False
        image_suffixes = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".avif", ".bmp", ".heic")
        if lowered.endswith(image_suffixes):
            return True
        if "/uploads/" in lowered:
            # Discourse uploads may have transformed paths with query strings or no obvious suffix.
            path = lowered.split("?", 1)[0]
            if path.endswith(image_suffixes):
                return True
            if "/optimized/" in path or "/default/" in path:
                return True
            return False
        return False

    def _format_topic_line(self, topic: dict[str, Any], index: int) -> str:
        topic_id = int(topic.get("id", 0))
        title = str(topic.get("title") or "(无标题)")
        posts_count = int(topic.get("posts_count", 0))
        reply_count = int(topic.get("reply_count", 0))
        replies = max(reply_count, max(0, posts_count - 1))
        views = int(topic.get("views", 0))
        return (
            f"{index}. {title}\n"
            f"   回复 {replies} | 浏览 {views} | {self._topic_url(topic_id)}"
        )

    def _conf_str(self, key: str, default: str) -> str:
        value = self.config.get(key, default)
        if value is None:
            return default
        return str(value).strip()

    def _conf_int(
        self,
        key: str,
        default: int,
        min_value: int | None = None,
        max_value: int | None = None,
    ) -> int:
        value = self.config.get(key, default)
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        if min_value is not None:
            parsed = max(min_value, parsed)
        if max_value is not None:
            parsed = min(max_value, parsed)
        return parsed

    def _conf_bool(self, key: str, default: bool) -> bool:
        value = self.config.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
