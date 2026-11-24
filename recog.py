"""Распознавание песни по аудио: сначала Shazam, потом (в bot.py) запасной вариант через Gemini."""
import asyncio
from shazamio import Shazam


async def shazam_recognize(path: str):
    """Возвращает 'Исполнитель — Название' или None."""
    try:
        out = await asyncio.wait_for(Shazam().recognize(path), timeout=60)
    except Exception:
        return None
    track = (out or {}).get("track")
    if not track:
        return None
    title, artist = track.get("title"), track.get("subtitle")
    if not title:
        return None
    return f"{artist} — {title}" if artist else title
