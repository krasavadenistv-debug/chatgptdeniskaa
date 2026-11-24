import os, re, asyncio, logging, shutil, tempfile, itertools
from collections import deque
from google import genai
from google.genai import types
from openai import AsyncOpenAI
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ChatAction
from telegram.ext import (Application, CommandHandler, MessageHandler, CallbackQueryHandler,
                          ContextTypes, filters)
import media
import recog

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("gpt-bot")

OWNER_ID = int(os.environ["OWNER_ID"])
ALLOWED_CHATS = {int(x) for x in os.getenv("ALLOWED_CHATS", "").replace(" ", "").split(",") if x}

gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
nvidia = AsyncOpenAI(base_url="https://integrate.api.nvidia.com/v1",
                     api_key=os.environ["NVIDIA_API_KEY"])

GEMINI_SMART = os.getenv("GEMINI_SMART", "gemini-2.5-pro")
GEMINI_FAST = os.getenv("GEMINI_FAST", "gemini-2.5-flash")
NVIDIA_CODE = os.getenv("NVIDIA_CODE", "nvidia/llama-3.1-nemotron-ultra-253b-v1")
NVIDIA_GENERAL = os.getenv("NVIDIA_GENERAL", "meta/llama-3.3-70b-instruct")

ROUTES = {
    "code":    [("nvidia", NVIDIA_CODE), ("gemini", GEMINI_SMART), ("gemini", GEMINI_FAST)],
    "hard":    [("gemini", GEMINI_SMART), ("nvidia", NVIDIA_CODE), ("gemini", GEMINI_FAST)],
    "general": [("gemini", GEMINI_FAST), ("nvidia", NVIDIA_GENERAL), ("gemini", GEMINI_SMART)],
}

# фразы, на которые бот реагирует без команд
SONG_RE = re.compile(
    r"^\s*(?:пожалуйста[,\s]*)?(?:найди|найти|скачай|скинь|кинь|дай|включи|поставь|отправь|пришли)\s+"
    r"(?:мне\s+)?(?:трек|песню|песня|музыку|композицию|саундтрек)\s+(.+)$", re.I | re.S)
VIDEO_RE = re.compile(
    r"^\s*(?:пожалуйста[,\s]*)?(?:найди|найти|скачай|скинь|кинь|дай|отправь|пришли)\s+"
    r"(?:мне\s+)?(?:видео|ролик|клип)\s+(.+)$", re.I | re.S)
DESC_RE = re.compile(r"^(?:где|в которой|в котор\w+|которая|которую|у которой|про\s|там\s|со словами|слова)", re.I)
NEWTOPIC_RE = re.compile(r"^\s*(?:новая тема|начни новую тему|начнём новую тему)\s*[:\-—]?\s*(.*)$", re.I | re.S)

STATES = {}
SEM = asyncio.Semaphore(2)          # не больше 2 загрузок одновременно
PENDING = {}                        # id кнопки -> поисковый запрос (для кнопок «Скачать»)
_ids = itertools.count(1)


def state(key):
    if key not in STATES:
        STATES[key] = {"hist": deque(maxlen=40), "title": None, "talk_all": False,
                       "auto_links": True, "forced": None}
    return STATES[key]


def key_of(update: Update):
    m, c = update.effective_message, update.effective_chat
    return (c.id, m.message_thread_id if getattr(c, "is_forum", False) else None)


def is_allowed(update: Update) -> bool:
    u, c = update.effective_user, update.effective_chat
    if not u:
        return False
    if u.id == OWNER_ID:
        return True
    return c.type != "private" and c.id in ALLOWED_CHATS


def is_owner(update: Update) -> bool:
    return bool(update.effective_user and update.effective_user.id == OWNER_ID)


def system_prompt(st, group: bool) -> str:
    s = "Ты умный помощник. Отвечай на языке пользователя, по делу и понятно."
    if group:
        s += (" Это групповой чат, в нём пишут разные люди; их сообщения помечены именами "
              "(«Имя: текст»). Обращайся к тому, кто спросил, и учитывай контекст всей беседы.")
    if st["title"]:
        s += f" Текущая тема беседы: «{st['title']}». Держись этой темы."
    return s


def merged(msgs):
    out = []
    for m in msgs:
        if out and out[-1]["role"] == m["role"]:
            out[-1] = {"role": m["role"], "content": out[-1]["content"] + "\n" + m["content"]}
        else:
            out.append(dict(m))
    while out and out[0]["role"] != "user":
        out.pop(0)
    return out


# ---------- модели ----------
async def route(text: str):
    """Один быстрый вызов: что хочет пользователь. Возвращает (тип, запрос)."""
    prompt = (
        "Определи, что хочет пользователь. Ответь СТРОГО в одном из форматов:\n"
        "code — программирование, скрипты, баги\n"
        "hard — сложная логика, математика, анализ, длинные рассуждения\n"
        "general — обычный разговор, простые вопросы, тексты, перевод\n"
        "song|название — просит найти/скинуть конкретную песню (названа песня или исполнитель)\n"
        "song_desc|описание — ищет песню, не зная названия: по словам из текста, сюжету, описанию\n"
        "video|запрос — просит найти и прислать видео или клип по описанию\n"
        f"Сообщение: {text[:1500]}\nОтвет:")
    try:
        r = await gemini.aio.models.generate_content(model=GEMINI_FAST, contents=prompt)
        head, _, q = (r.text or "").strip().partition("|")
        head = head.strip().lower()
        if head in ("song", "song_desc", "video") and q.strip():
            return head, q.strip()
        return (head if head in ROUTES else "general"), None
    except Exception as e:
        log.warning("route failed: %s", e)
        return "general", None


async def ask_gemini(model, msgs, system):
    contents = [types.Content(role="user" if m["role"] == "user" else "model",
                              parts=[types.Part(text=m["content"])]) for m in msgs]
    r = await gemini.aio.models.generate_content(
        model=model, contents=contents,
        config=types.GenerateContentConfig(system_instruction=system))
    return r.text


async def ask_nvidia(model, msgs, system):
    r = await nvidia.chat.completions.create(
        model=model, max_tokens=4096,
        messages=[{"role": "system", "content": system}] + msgs)
    return r.choices[0].message.content


async def answer(st, cat: str, group: bool):
    cat = st["forced"] or cat
    system = system_prompt(st, group)
    last_err = None
    for provider, model in ROUTES[cat]:
        try:
            msgs = merged(list(st["hist"]))
            out = await (ask_gemini if provider == "gemini" else ask_nvidia)(model, msgs, system)
            if out:
                st["hist"].append({"role": "assistant", "content": out})
                return out, f"{model} · {cat}"
        except Exception as e:
            last_err = e
            log.warning("%s/%s failed: %s", provider, model, e)
    return f"Все модели недоступны: {last_err}", "error"


async def guess_songs(desc: str):
    """Поиск песни по описанию/словам через Gemini + поиск Google. До 3 вариантов."""
    prompt = (f"Пользователь ищет песню, не зная названия. Что он помнит: «{desc}».\n"
              "Найди в интернете и выведи до 3 наиболее вероятных вариантов, по одному на строку, "
              "строго в формате: Исполнитель — Название. Без пояснений и нумерации.")
    r = await gemini.aio.models.generate_content(
        model=GEMINI_FAST, contents=prompt,
        config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())]))
    res = []
    for line in (r.text or "").splitlines():
        line = re.sub(r"^[\s\-*•\d.)]+", "", line).strip().strip("*").strip()
        if len(line) > 3 and re.search(r"\s[—–-]\s", line):
            res.append(line[:100])
    return res[:3]


# ---------- загрузка ----------
def friendly_error(e: Exception) -> str:
    s = str(e)
    if "Sign in to confirm" in s or "not a bot" in s:
        return ("YouTube заблокировал запрос с этого сервера. Нужны cookies "
                "(YT_COOKIES_FILE) или запуск бота с домашнего IP.")
    if "Private" in s or "unavailable" in s:
        return "Видео недоступно или приватное."
    return s[:300]


def thread_of(msg, chat):
    return msg.message_thread_id if getattr(chat, "is_forum", False) else None


async def send_media(update: Update, ctx: ContextTypes.DEFAULT_TYPE, target: str, audio: bool):
    msg, chat = update.effective_message, update.effective_chat
    status = await msg.reply_text("🔎 Ищу и качаю…")
    await ctx.bot.send_chat_action(chat.id, ChatAction.UPLOAD_VOICE if audio else ChatAction.UPLOAD_VIDEO,
                                   message_thread_id=thread_of(msg, chat))
    tmp = None
    try:
        async with SEM:
            fn = media.download_audio if audio else media.download_video
            tmp, path, info = await asyncio.to_thread(fn, target)
        info = info or {}
        title = info.get("title") or "Без названия"
        with open(path, "rb") as f:
            if audio:
                await msg.reply_audio(f, title=title[:100],
                                      performer=(info.get("artist") or info.get("uploader") or "")[:100],
                                      duration=int(info.get("duration") or 0) or None,
                                      read_timeout=300, write_timeout=300)
            else:
                await msg.reply_video(f, caption=title[:200], supports_streaming=True,
                                      read_timeout=300, write_timeout=300)
        await status.delete()
    except Exception as e:
        log.exception("download failed")
        await status.edit_text(f"Не получилось: {friendly_error(e)}")
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


def download_buttons(names):
    rows = []
    for n in names:
        i = next(_ids)
        PENDING[i] = n
        if len(PENDING) > 300:
            PENDING.pop(next(iter(PENDING)))
        rows.append([InlineKeyboardButton(f"⬇️ {n}"[:60], callback_data=f"dl:{i}")])
    return InlineKeyboardMarkup(rows)


async def on_button(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    if not is_allowed(update):
        await q.answer()
        return
    await q.answer()
    name = PENDING.get(int((q.data or "dl:0").split(":")[1]))
    if not name:
        await q.message.reply_text("Кнопка устарела, попроси ещё раз.")
        return
    await send_media(update, ctx, name, audio=True)


# ---------- распознавание песни по аудио ----------
def pick_media(msg):
    return msg.voice or msg.audio or msg.video or msg.video_note


async def gemini_listen(path: str, mime: str):
    """Запасной вариант: Gemini слушает запись. Музыка -> гипотеза о треке, речь -> ответ на вопрос."""
    with open(path, "rb") as f:
        data = f.read()
    prompt = ("Послушай запись. Ответь СТРОГО в одном формате:\n"
              "SONG|Исполнитель — Название  — если это музыка и ты можешь предположить трек "
              "(по словам, мелодии, голосу). Если не уверен, всё равно дай лучшую догадку.\n"
              "UNKNOWN  — если это музыка, но догадок нет.\n"
              "SPEECH|ответ  — если это речь человека: коротко ответь на его вопрос или реплику "
              "на языке говорящего.")
    r = await gemini.aio.models.generate_content(
        model=GEMINI_FAST,
        contents=[types.Part.from_bytes(data=data, mime_type=mime), prompt])
    t = (r.text or "").strip()
    head, _, rest = t.partition("|")
    return head.strip().upper(), rest.strip()


async def recognize_media(update: Update, ctx: ContextTypes.DEFAULT_TYPE, src_msg):
    msg, chat = update.effective_message, update.effective_chat
    m = pick_media(src_msg)
    if not m:
        return
    if (m.file_size or 0) > 20 * 1024 * 1024:
        await msg.reply_text("Файл больше 20 МБ, пришли кусочек покороче.")
        return
    status = await msg.reply_text("🎧 Слушаю…")
    tmp = tempfile.mkdtemp(prefix="rec_")
    try:
        f = await m.get_file()
        path = os.path.join(tmp, "in" + os.path.splitext(f.file_path or "")[1])
        await f.download_to_drive(path)
        mime = getattr(m, "mime_type", None) or ("audio/ogg" if src_msg.voice else "video/mp4")

        name = await recog.shazam_recognize(path)
        if name:
            await status.edit_text(f"🎵 Это: {name}\nСкидываю трек…")
            await send_media(update, ctx, name, audio=True)
            await status.delete()
            return

        kind, rest = await gemini_listen(path, mime)
        if kind == "SONG" and rest:
            await status.edit_text(f"🤔 Не уверен, но похоже на: {rest}",
                                   reply_markup=download_buttons([rest]))
        elif kind == "SPEECH" and rest:
            await status.edit_text(rest[:4000])
        else:
            await status.edit_text("Не смог определить трек. Попробуй кусочек подлиннее и почище.")
    except Exception as e:
        log.exception("recognize failed")
        await status.edit_text(f"Не получилось: {str(e)[:200]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


async def on_media(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update):
        return
    msg, chat = update.effective_message, update.effective_chat
    if chat.type != "private":     # в группе — только если позвали или ответили боту
        uname = (ctx.bot.username or "").lower()
        cap = (msg.caption or "").lower()
        replied = bool(msg.reply_to_message and msg.reply_to_message.from_user
                       and msg.reply_to_message.from_user.id == ctx.bot.id)
        if not (replied or (uname and f"@{uname}" in cap)):
            return
    await recognize_media(update, ctx, msg)


# ---------- команды (необязательные) ----------
async def start(update, ctx):
    if not is_allowed(update):
        return
    await update.effective_message.reply_text(
        "Я ИИ-помощник. Команды не нужны, просто пиши:\n"
        "• «найди трек Imagine Dragons Believer» — пришлю песню\n"
        "• «найди песню, где поётся про …» — найду по описанию\n"
        "• ссылка на YouTube/TikTok — пришлю видео (добавь «mp3» — пришлю звук)\n"
        "• голосовое или кусочек песни — скажу, что за трек, и пришлю его\n"
        "• любой вопрос — отвечу лучшей моделью\n\n"
        "В группе: упомяни меня (@) или ответь на моё сообщение. Чтобы определить песню из чужого "
        "голосового, ответь на него и упомяни меня.\n"
        "Новая тема: «новая тема: название» или /new название.\n"
        "Владелец: /talk on|off, /links on|off, /id, /allow. Режим модели: /auto /code /hard.")


async def cmd_new(update, ctx):
    if not is_allowed(update):
        return
    await new_topic(update, " ".join(ctx.args).strip())


async def new_topic(update, title: str):
    st = state(key_of(update))
    st["hist"].clear()
    st["title"] = title or None
    await update.effective_message.reply_text(
        f"🆕 Новая тема: {title}" if title else "🆕 Начинаем с чистого листа.")


async def cmd_topic(update, ctx):
    if not is_allowed(update):
        return
    t = state(key_of(update))["title"]
    await update.effective_message.reply_text(f"Тема: {t}" if t else "Тема свободная. Задать: «новая тема: …»")


async def cmd_mode(update, ctx):
    if not is_allowed(update):
        return
    cmd = update.effective_message.text.split()[0].lstrip("/").split("@")[0]
    state(key_of(update))["forced"] = None if cmd == "auto" else cmd
    await update.effective_message.reply_text(f"Режим: {cmd}")


def _onoff(ctx):
    return bool(ctx.args) and ctx.args[0].lower() in ("on", "вкл", "1")


async def cmd_talk(update, ctx):
    if not is_owner(update):
        return
    on = _onoff(ctx)
    state(key_of(update))["talk_all"] = on
    await update.effective_message.reply_text(
        "Теперь отвечаю на все сообщения." if on else "Отвечаю только на упоминания и ответы мне.")


async def cmd_links(update, ctx):
    if not is_owner(update):
        return
    on = _onoff(ctx)
    state(key_of(update))["auto_links"] = on
    await update.effective_message.reply_text(
        "Ссылки на YouTube/TikTok скачиваю сразу." if on else "Ссылки скачиваю только если меня позвали.")


async def cmd_id(update, ctx):
    if not is_owner(update):
        return
    await update.effective_message.reply_text(f"ID чата: {update.effective_chat.id}")


async def cmd_allow(update, ctx):
    if not is_owner(update):
        return
    cid = update.effective_chat.id
    ALLOWED_CHATS.add(cid)
    await update.effective_message.reply_text(
        f"Группа разрешена до перезапуска. Чтобы навсегда — добавь {cid} в секрет ALLOWED_CHATS.")


# ---------- обычные сообщения ----------
async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update):
        return
    msg, chat, user = update.effective_message, update.effective_chat, update.effective_user
    group = chat.type != "private"
    st = state(key_of(update))

    uname = ctx.bot.username or ""
    raw = msg.text or ""
    text = re.sub(rf"@{re.escape(uname)}", "", raw, flags=re.I).strip() if uname else raw.strip()
    mentioned = bool(uname) and f"@{uname}".lower() in raw.lower()
    replied = bool(msg.reply_to_message and msg.reply_to_message.from_user
                   and msg.reply_to_message.from_user.id == ctx.bot.id)
    addressed = (not group) or mentioned or replied or st["talk_all"]

    # ссылка на ролик -> сразу видео (в группе — без упоминания, если включено /links)
    link = media.URL_RE.search(text)
    link_only = bool(link) and len(text) < len(link.group(0)) + 25
    if link_only and (addressed or st["auto_links"]):
        audio = bool(re.search(r"mp3|аудио|audio|песн|музык", text, re.I))
        await send_media(update, ctx, link.group(0), audio=audio)
        return

    # в группе запоминаем сообщения всех участников, чтобы бот понимал контекст беседы
    st["hist"].append({"role": "user", "content": f"{user.first_name}: {text}" if group else text})
    if not addressed:
        return

    # ответ на чужое голосовое/видео с упоминанием бота -> распознаём песню
    rt = msg.reply_to_message
    if rt and not replied and pick_media(rt):
        await recognize_media(update, ctx, rt)
        return

    m = NEWTOPIC_RE.match(text)
    if m:
        await new_topic(update, m.group(1).strip())
        return

    await ctx.bot.send_chat_action(chat.id, ChatAction.TYPING, message_thread_id=thread_of(msg, chat))

    kind, q = None, None
    m = SONG_RE.match(text)
    if m:
        q = m.group(1).strip()
        kind = "song_desc" if DESC_RE.match(q) else "song"    # «найди песню, где поётся про…» -> по описанию
    else:
        m = VIDEO_RE.match(text)
        if m:
            kind, q = "video", m.group(1).strip()
    if not kind:
        kind, q = await route(text)
    if kind == "song":
        await send_media(update, ctx, q, audio=True)
        return
    if kind == "video":
        await send_media(update, ctx, q, audio=False)
        return
    if kind == "song_desc":
        try:
            variants = await guess_songs(q)
        except Exception as e:
            log.warning("guess failed: %s", e)
            variants = []
        if variants:
            await msg.reply_text("Похоже на это. Нажми, какой скинуть:",
                                 reply_markup=download_buttons(variants))
        else:
            await msg.reply_text("Не смог найти такую песню. Опиши подробнее: слова из припева, язык, "
                                 "про что поётся.")
        return

    out, info = await answer(st, kind, group)
    for i in range(0, len(out), 4000):      # лимит Telegram — 4096 символов
        await msg.reply_text(out[i:i + 4000])
    await msg.reply_text(f"🤖 {info}")


def main():
    app = Application.builder().token(os.environ["TELEGRAM_BOT_TOKEN"]).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("new", cmd_new))
    app.add_handler(CommandHandler("topic", cmd_topic))
    app.add_handler(CommandHandler(["auto", "code", "hard"], cmd_mode))
    app.add_handler(CommandHandler("talk", cmd_talk))
    app.add_handler(CommandHandler("links", cmd_links))
    app.add_handler(CommandHandler("id", cmd_id))
    app.add_handler(CommandHandler("allow", cmd_allow))
    app.add_handler(CallbackQueryHandler(on_button, pattern=r"^dl:"))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO | filters.VIDEO | filters.VIDEO_NOTE, on_media))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & ~filters.VIA_BOT, on_text))
    app.run_polling()


if __name__ == "__main__":
    main()
