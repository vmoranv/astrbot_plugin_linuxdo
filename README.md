# astrbot_plugin_linuxdo

Linux.do 查询与订阅插件（AstrBot）。

## 命令

- `/ldo <url|帖子id>`：返回该帖一楼内容。
- `/ldo_trend`：返回当前热门帖子列表（条数由配置控制）。
- `/ldo_auth_check`：检查登录态与接口连通性（`u/current.json` / `categories.json` / `top.json`）。
- `/ldo_comment <url|帖子id> <num>`：返回该帖前 `num` 条评论（从 2 楼开始）。当 `num` 缺省时使用配置值。
- `/ldo` 会尝试额外发送一楼图片（数量与单图大小可配置；默认最多 6 张、每张 4MB）。
- `/ldo_sub <tag>`：在当前群订阅 `tag`，后台按配置间隔轮询新帖并推送命中结果。
  - 订阅扫描优先使用 `latest.json`，失败时自动回退到 `top.json` / `new.json`，降低单端点限流导致的误报。
- `/ldo_sub <tag>` 支持两种写法：
  - 普通关键词：如 `docker`（按“包含”匹配）
  - 正则关键词：如 `re:^求助.*docker`（按正则匹配）
- `/ldo_unsub [tag]`：取消当前群订阅；不传 `tag` 则清空当前群全部订阅。
- `/ldo_sub_list`：查看当前群订阅状态。

## 配置

通过 `_conf_schema.json` 配置以下核心项：

- `cookie` / `user_agent`：Linux.do 登录凭证（参考 linuxdo-explorer 的抓取方式）。
- `proxy_url`：可选 HTTP 代理（例如 `http://127.0.0.1:10808`）。
- `proxy_use_env`：是否使用环境变量代理（`HTTP_PROXY/HTTPS_PROXY/ALL_PROXY`）。
- `curl_impersonate`：`curl_cffi` 的浏览器指纹（如 `chrome124`）。
- `request_retry_on_429` / `request_max_retries` / `request_retry_backoff_seconds`：429 限流自动重试策略。
- `trend_return_count`：`/ldo_trend` 返回帖子个数。
- `comment_return_count`：`/ldo_comment` 默认返回评论个数。
- `post_image_max_count` / `post_image_max_bytes_kb`：`/ldo` 图片发送限制。
- `ldo_update_interval_seconds`：订阅轮询间隔。
- `subscription_scan_count` / `subscription_scan_pages`：订阅扫描范围（用于避免漏推送）。
- `match_title_for_sub`：是否额外匹配标题（默认开启）。
- `subscription_match_title_only`：是否仅匹配标题（开启后不匹配 Linux.do 主题 tags）。
- `subscription_enable_regex`：是否启用 `re:` 正则关键词。
- `subscription_regex_prefix` / `subscription_regex_ignore_case` / `subscription_regex_max_length`：正则匹配行为控制。

其余项可用于调节超时、摘要长度、订阅扫描范围等。

## 说明

- 订阅推送目标为触发 `/ldo_sub` 的当前群聊（通过 `event.unified_msg_origin` 存储）。
- 订阅状态会持久化到插件数据目录下的 `subscriptions.json`。
- 如登录状态失效，请更新 `cookie` 和 `user_agent`。
- 若机器存在错误的全局代理环境变量（如 `HTTP_PROXY` 指向不可用端口），建议显式设置 `proxy_url` 或将 `proxy_use_env` 设为 `false`。
