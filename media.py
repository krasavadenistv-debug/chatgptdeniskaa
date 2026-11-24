"""Поиск и скачивание аудио/видео через yt-dlp (YouTube, TikTok и др.)."""
import os, re, glob, shutil, tempfile
import yt_dlp
from yt_dlp.utils import match_filter_func

MAX_BYTES = 49 * 1024 * 1024          # лимит загрузки ботом в Telegram — 50 МБ
URL_RE = re.compile(r"https?://\S*(?:youtube\.com|youtu\.be|tiktok\.com)\S*", re.I)


def _base_opts(outdir):
    o = {
        "outtmpl": os.path.join(outdir, "%(title).80B [%(id)s].%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 30,
        "retries": 3,
    }
    cookies = os.getenv("YT_COOKIES_FILE")
    if cookies and os.path.exists(cookies):
        o["cookiefile"] = cookies
    return o


def _first_entry(info):
    if info and "entries" in info:
        entries = [e for e in info["entries"] if e]
        return entries[0] if entries else None
    return info


def download_video(target: str):
    """target — ссылка или текстовый запрос. Возвращает (tmpdir, путь, info)."""
    if not target.lower().startswith("http"):
        target = f"ytsearch1:{target}"
    tmp = tempfile.mkdtemp(prefix="dl_")
    try:
        for h in (720, 480, 360):                       # если файл > 50 МБ — пробуем меньше качество
            for f in glob.glob(os.path.join(tmp, "*")):
                os.remove(f)
            opts = _base_opts(tmp)
            opts.update({
                "format": (f"bv*[height<={h}][ext=mp4]+ba[ext=m4a]/"
                           f"b[height<={h}][ext=mp4]/b[height<={h}]/b"),
                "merge_output_format": "mp4",
                "match_filter": match_filter_func("duration < 1800"),   # не длиннее 30 минут
            })
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = _first_entry(ydl.extract_info(target, download=True))
            files = glob.glob(os.path.join(tmp, "*"))
            if not files:
                raise ValueError("Видео не найдено или длиннее 30 минут")
            path = max(files, key=os.path.getsize)
            if os.path.getsize(path) <= MAX_BYTES:
                return tmp, path, info
        raise ValueError("Видео больше 50 МБ даже в 360p (лимит Telegram для ботов)")
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def download_audio(target: str):
    """Находит песню (по названию/ссылке) и отдаёт mp3. Возвращает (tmpdir, путь, info)."""
    if not target.lower().startswith("http"):
        target = f"ytsearch1:{target} official audio"
    tmp = tempfile.mkdtemp(prefix="dl_")
    try:
        opts = _base_opts(tmp)
        opts.update({
            "format": "bestaudio/best",
            "match_filter": match_filter_func("duration < 900"),        # не длиннее 15 минут
            "postprocessors": [{"key": "FFmpegExtractAudio",
                                "preferredcodec": "mp3", "preferredquality": "192"}],
        })
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = _first_entry(ydl.extract_info(target, download=True))
        files = glob.glob(os.path.join(tmp, "*.mp3"))
        if not files:
            raise ValueError("Не нашёл трек или он длиннее 15 минут")
        path = files[0]
        if os.path.getsize(path) > MAX_BYTES:
            raise ValueError("Файл больше 50 МБ")
        return tmp, path, info
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
