"""Discord embed formatting for new signals and outcome notifications."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any

from .discord_client import DiscordClient

JST = timezone(timedelta(hours=9))

COLOR_LONG = 0x2ECC71   # green
COLOR_SHORT = 0xE74C3C  # red
COLOR_TP = 0x1ABC9C     # teal
COLOR_SL = 0x95A5A6     # grey
COLOR_EXPIRED = 0x7F8C8D


def _fmt_price(p: float) -> str:
    return f"${p:,.2f}"


def _fmt_pct(p: float, signed: bool = True) -> str:
    return f"{p:+.2f}%" if signed else f"{p:.2f}%"


def _fmt_jst(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=JST).strftime("%Y-%m-%d %H:%M JST")


def signal_embed(sig: dict[str, Any]) -> dict[str, Any]:
    """Discord embed for a newly fired signal."""
    f = sig["features"]
    is_long = sig["direction"] == "LONG"
    arrow = "🟢 LONG" if is_long else "🔴 SHORT"

    stop_pct = (sig["stop_price"] / sig["entry_price"] - 1) * 100
    target_pct = (sig["target_price"] / sig["entry_price"] - 1) * 100

    desc_lines = [
        f"**{arrow} {sig['coin']}** @ {_fmt_price(sig['entry_price'])} (HL perp)",
        f"SL {_fmt_price(sig['stop_price'])} ({_fmt_pct(stop_pct)}, {1.5:.1f}×ATR)",
        f"TP {_fmt_price(sig['target_price'])} ({_fmt_pct(target_pct)}, {2.5:.1f}×ATR)",
        f"R:R 1:{sig['rr_ratio']:.2f}  ·  expires {_fmt_jst(sig['expires_at_ms'])}",
        "",
        f"score **{sig['composite_score']:.2f}**  ·  "
        f"move/ATR {f['move_per_atr']:.2f}  ·  vol_z {f['vol_z_168']:.2f}",
        f"trend_4h {'↑' if f['trend_4h'] > 0 else '↓' if f['trend_4h'] < 0 else '→'}  ·  "
        f"funding_z {f['funding_z_24']:+.2f}  ·  ret_4h {_fmt_pct(f['ret_4h'])}",
    ]

    return {
        "title": f"signal #{sig['signal_id']} · {_fmt_jst(sig['generated_at_ms'])}",
        "description": "\n".join(desc_lines),
        "color": COLOR_LONG if is_long else COLOR_SHORT,
    }


def outcome_embed(notif: dict[str, Any]) -> dict[str, Any]:
    """Discord embed for outcome (TP/SL/EXPIRED)."""
    status = notif["status"]
    icon = {"HIT_TP": "✅", "HIT_SL": "❌", "EXPIRED": "⏱"}.get(status, "ℹ️")
    color = {"HIT_TP": COLOR_TP, "HIT_SL": COLOR_SL, "EXPIRED": COLOR_EXPIRED}.get(status, 0x95A5A6)
    is_long = notif["direction"] == "LONG"
    arrow = "LONG" if is_long else "SHORT"

    desc_lines = [
        f"**{icon} {status.replace('_', ' ')}**  ·  {arrow} signal #{notif['signal_id']}",
        f"entry {_fmt_price(notif['entry_price'])} → exit {_fmt_price(notif['close_price'])}",
        f"realized **{_fmt_pct(notif['realized_return'])}**",
        f"opened {_fmt_jst(notif['generated_at_ms'])}",
    ]
    return {
        "title": f"signal #{notif['signal_id']} {status}",
        "description": "\n".join(desc_lines),
        "color": color,
    }


def publish_signal(client: DiscordClient, sig: dict[str, Any]) -> None:
    client.send(embeds=[signal_embed(sig)])


def publish_outcome(client: DiscordClient, notif: dict[str, Any]) -> None:
    client.send(embeds=[outcome_embed(notif)])
