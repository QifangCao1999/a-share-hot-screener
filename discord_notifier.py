"""Discord Bot 通知模块 — 通过 REST API 发送筛选结果到 Discord.

不依赖 discord.py，仅使用 requests 库。
支持：DM 私信 / 频道消息 / Embed 富文本格式。
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger("a_share_hot_screener.discord_notifier")

# ── 默认配置（可通过环境变量或 .env 覆盖）──────────────────
DEFAULT_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")
DEFAULT_CHANNEL_ID = os.getenv("DISCORD_CHANNEL_ID", "")
DEFAULT_USER_ID = os.getenv("DISCORD_USER_ID", "")

API_BASE = "https://discord.com/api/v10"


class DiscordNotifier:
    """Discord Bot REST API 通知器."""

    def __init__(
        self,
        bot_token: str = "",
        channel_id: str = "",
        user_id: str = "",
    ) -> None:
        self.bot_token = bot_token or DEFAULT_BOT_TOKEN
        self.channel_id = channel_id or DEFAULT_CHANNEL_ID
        self.user_id = user_id or DEFAULT_USER_ID
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bot {self.bot_token}",
            "Content-Type": "application/json",
        })
        self._dm_channel_id: Optional[str] = None

    # ── 低级 API ──────────────────────────────────────────

    def _request(self, method: str, path: str, **kwargs) -> Optional[Dict]:
        """发送 Discord API 请求，处理速率限制."""
        url = f"{API_BASE}{path}"
        resp = self._session.request(method, url, **kwargs)
        if resp.status_code == 429:
            retry_after = resp.json().get("retry_after", 5)
            logger.warning("Discord rate limited, retry after %.1fs", retry_after)
            import time
            time.sleep(retry_after + 0.5)
            resp = self._session.request(method, url, **kwargs)
        if resp.status_code >= 400:
            logger.error("Discord API error %d: %s", resp.status_code, resp.text[:500])
            return None
        return resp.json() if resp.text else {}

    def _get_dm_channel(self) -> Optional[str]:
        """获取或创建与用户的 DM 频道."""
        if self._dm_channel_id:
            return self._dm_channel_id
        if not self.user_id:
            return None
        data = self._request("POST", "/users/@me/channels", json={
            "recipient_id": self.user_id,
        })
        if data and "id" in data:
            self._dm_channel_id = data["id"]
            logger.info("DM channel created: %s", self._dm_channel_id)
            return self._dm_channel_id
        return None

    def _resolve_channel(self) -> Optional[str]:
        """确定发送目标频道：优先 channel_id，fallback 到 DM."""
        if self.channel_id:
            return self.channel_id
        return self._get_dm_channel()

    # ── 发送消息 ──────────────────────────────────────────

    def send_message(
        self,
        content: str = "",
        embeds: Optional[List[Dict]] = None,
        channel_id: Optional[str] = None,
    ) -> bool:
        """发送消息到指定频道或默认频道."""
        target = channel_id or self._resolve_channel()
        if not target:
            logger.error("无法确定发送目标: 需要 channel_id 或 user_id")
            return False

        payload: Dict[str, Any] = {}
        if content:
            payload["content"] = content[:2000]  # Discord 限制
        if embeds:
            payload["embeds"] = embeds[:10]  # 最多 10 个 embed

        result = self._request(
            "POST", f"/channels/{target}/messages", json=payload
        )
        if result:
            logger.info("消息已发送到频道 %s", target)
            return True
        # 如果 channel_id 失败，尝试 DM fallback
        if target == self.channel_id and self.user_id:
            logger.warning("频道发送失败，尝试 DM fallback")
            dm_channel = self._get_dm_channel()
            if dm_channel:
                result = self._request(
                    "POST", f"/channels/{dm_channel}/messages", json=payload
                )
                return result is not None
        return False

    def send_file(
        self,
        filepath: str,
        content: str = "",
        channel_id: Optional[str] = None,
    ) -> bool:
        """发送文件附件."""
        target = channel_id or self._resolve_channel()
        if not target:
            logger.error("无法确定发送目标")
            return False

        # 文件上传需要 multipart/form-data
        with open(filepath, "rb") as f:
            files = {"file": (os.path.basename(filepath), f)}
            data = {"content": content[:2000]} if content else {}
            # 临时移除 Content-Type header (requests 会自动设置 multipart)
            headers = {"Authorization": f"Bot {self.bot_token}"}
            resp = requests.post(
                f"{API_BASE}/channels/{target}/messages",
                headers=headers,
                data=data,
                files=files,
            )
        if resp.status_code >= 400:
            logger.error("文件上传失败 %d: %s", resp.status_code, resp.text[:500])
            return False
        logger.info("文件已上传: %s", filepath)
        return True


# ── 格式化工具 ──────────────────────────────────────────────


def format_screener_results(
    run_date: str,
    passed_stocks: List[Dict],
    total_input: int,
    passed_hard_filter: int,
    elapsed_seconds: float,
    extra_info: str = "",
) -> List[Dict]:
    """将筛选结果格式化为 Discord Embed 列表.

    Returns:
        List of embed dicts for Discord API.
    """
    # ── 主 Embed: 概览 ──
    pass_count = len(passed_stocks)
    pass_rate = f"{pass_count / total_input * 100:.1f}%" if total_input else "N/A"
    elapsed_min = elapsed_seconds / 60

    description_lines = [
        f"📊 **输入**: {total_input} 只",
        f"🔍 **通过硬筛**: {passed_hard_filter} 只",
        f"✅ **通过 Stage1**: **{pass_count}** 只 ({pass_rate})",
        f"⏱️ **耗时**: {elapsed_min:.1f} 分钟",
    ]
    if extra_info:
        description_lines.append(f"\n{extra_info}")

    main_embed: Dict[str, Any] = {
        "title": f"🔥 A股热点筛选日报 — {run_date}",
        "description": "\n".join(description_lines),
        "color": 0xFF6B35 if pass_count > 0 else 0x808080,
        "footer": {"text": "A-Share Hot Screener Stage 1"},
    }

    embeds = [main_embed]

    # ── 个股 Embed（按 total_score 降序，最多 10 只）──
    if passed_stocks:
        sorted_stocks = sorted(
            passed_stocks, key=lambda s: s.get("total_score", 0), reverse=True
        )

        stock_lines = []
        for i, s in enumerate(sorted_stocks[:25], 1):
            code = s.get("code", "?")
            name = s.get("name", "?")
            score = s.get("total_score", 0)
            ht = s.get("hot_theme_score", 0)
            tf = s.get("trend_flow_score", 0)
            le = s.get("liquidity_execution_score", 0)
            rc = s.get("risk_control_score", 0)
            industry = s.get("industry", "?")

            stock_lines.append(
                f"**{i}. {name}** (`{code}`) — ⭐ {score:.4f}\n"
                f"　　HT={ht:.2f} TF={tf:.2f} LE={le:.2f} RC={rc:.2f} | {industry}"
            )

        # Discord embed description 限制 4096 字符，分批
        chunk = []
        chunk_len = 0
        for line in stock_lines:
            if chunk_len + len(line) + 1 > 3800:
                embeds.append({
                    "description": "\n".join(chunk),
                    "color": 0x00C853,
                })
                chunk = []
                chunk_len = 0
            chunk.append(line)
            chunk_len += len(line) + 1

        if chunk:
            embeds.append({
                "title": f"📈 通过股票 Top {min(len(sorted_stocks), 25)}",
                "description": "\n".join(chunk),
                "color": 0x00C853,
            })

    return embeds
