import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Tuple

import asyncpg
from aiohttp import web
from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message, Update
from openai import OpenAI

logging.basicConfig(level=logging.INFO)

# ---------- ENV ----------
BOT_TOKEN = os.environ["BOT_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DATABASE_URL = os.environ["DATABASE_URL"]
PUBLIC_URL = os.environ["PUBLIC_URL"].rstrip("/")
WEBHOOK_SECRET = os.environ["WEBHOOK_SECRET"]
PORT = int(os.getenv("PORT", 8080))
HISTORY_WINDOW = int(os.getenv("HISTORY_WINDOW", 12))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", 800))

# ---------- GLOBALS ----------
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
router = Router()
client = OpenAI(api_key=OPENAI_API_KEY)
pool: asyncpg.Pool | None = None

SYSTEM_PROMPTS = {
    "neutral": (
        "Ты — ‘Ирис‑лайт’: дружелюбный русскоязычный ассистент в Telegram."
        " Отвечай ясно и по делу. Избегай токсичности и небезопасных советов."
    ),
    "brief": (
        "Отвечай кратко, по пунктам, максимум фактов, минимум воды."
    ),
    "expert": (
        "Тон — экспертный и аккуратный, добавляй пояснения и caveats."
    ),
    "coach": (
        "Поддерживай, давай конкретные шаги и чек‑листы. Без сюсюканья."
    ),
    "coder": (
        "Отвечай как senior‑разработчик: примеры кода, лучшие практики, кратко."
    ),
}

# ---------- DB ----------
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
  user_id BIGINT PRIMARY KEY,
  mode TEXT NOT NULL DEFAULT 'neutral',
  last_seen TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS messages (
  id BIGSERIAL PRIMARY KEY,
  user_id BIGINT NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('user','assistant')),
  content TEXT NOT NULL,
  ts TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_messages_user_ts ON messages(user_id, ts);
"""

async def get_pool() -> asyncpg.Pool:
    global pool
    if pool is None:
        pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
        async with pool.acquire() as conn:
            await conn.execute(SCHEMA_SQL)
    return pool

async def set_mode(user_id: int, mode: str):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO users(user_id, mode) VALUES($1,$2) "
            "ON CONFLICT (user_id) DO UPDATE SET mode=EXCLUDED.mode, last_seen=NOW()",
            user_id, mode,
        )

async def get_mode(user_id: int) -> str:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT mode FROM users WHERE user_id=$1", user_id)
        return row[0] if row else "neutral"

async def append_msg(user_id: int, role: str, content: str):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO messages(user_id, role, content) VALUES($1,$2,$3)",
            user_id, role, content,
        )

async def get_history(user_id: int, limit: int) -> List[Tuple[str, str]]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT role, content FROM messages WHERE user_id=$1 ORDER BY ts DESC LIMIT $2",
            user_id, limit,
        )
    return list(reversed([(r[0], r[1]) for r in rows]))

async def clear_history(user_id: int):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM messages WHERE user_id=$1", user_id)

# ---------- Utils ----------

def chunk(text: str, limit: int = 4096) -> List[str]:
    parts = []
    while text:
        parts.append(text[:limit])
        text = text[limit:]
    return parts

_last_reply: dict[int, datetime] = {}

async def rate_limited(user_id: int, min_interval: float = 2.5) -> bool:
    now = datetime.utcnow()
    last = _last_reply.get(user_id)
    if last and (now - last) < timedelta(seconds=min_interval):
        return True
    _last_reply[user_id] = now
    return False

# ---------- LLM ----------
async def llm_reply(user_id: int, text: str) -> str:
    mode = await get_mode(user_id)
    sys_prompt = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["neutral"]) + (
        " Если просят код — давай рабочие примеры. Если запрос опасен — откажись"
        " и предложи безопасную альтернативу."
    )
    history = await get_history(user_id, HISTORY_WINDOW)

    messages = [{"role": "system", "content": sys_prompt}]
    for role, content in history:
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": text})

    def _call():
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.6,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        return resp.choices[0].message.content.strip()

    return await asyncio.to_thread(_call)

# ---------- Handlers ----------
@router.message(CommandStart())
async def on_start(m: Message):
    await set_mode(m.from_user.id, "neutral")
    await m.answer(
        "Привет! Я ‘Ирис‑лайт’. Пиши что нужно.\n\n"
        "Команды: /mode, /reset, /help"
    )

@router.message(Command("help"))
async def on_help(m: Message):
    await m.answer(
        "/mode — показать/сменить режим.\n"
        "/reset — очистить историю.\n"
        "Пиши любые вопросы — помню контекст."
    )

@router.message(Command("reset"))
async def on_reset(m: Message):
    await clear_history(m.from_user.id)
    await m.answer("Готово. Контекст очищен.")

@router.message(Command("mode"))
async def on_mode(m: Message):
    parts = m.text.strip().split()
    if len(parts) == 1:
        cur = await get_mode(m.from_user.id)
        await m.answer(
            f"Текущий режим: {cur}. Доступны: brief, expert, coach, coder, neutral.\n"
            "Пример: /mode expert"
        )
        return
    new_mode = parts[1].lower()
    if new_mode not in SYSTEM_PROMPTS:
        await m.answer("Не знаю такой режим. Попробуй: brief, expert, coach, coder, neutral.")
        return
    await set_mode(m.from_user.id, new_mode)
    await m.answer(f"Режим сменён на: {new_mode}")

@router.message(F.text.len() > 0)
async def on_text(m: Message):
    uid = m.from_user.id
    if await rate_limited(uid):
        await m.answer("Секунду...")
        return

    user_text = m.text.strip()
    await append_msg(uid, "user", user_text)

    try:
        reply = await llm_reply(uid, user_text)
    except Exception as e:
        logging.exception("LLM error: %s", e)
        await m.answer("Модель молчит. Попробуй переформулировать или позже.")
        return

    await append_msg(uid, "assistant", reply)
    for p in chunk(reply):
        await m.answer(p)

# ---------- Webhook app (aiohttp) ----------
app = web.Application()

async def handle_health(request: web.Request):
    return web.Response(text="ok")

async def handle_webhook(request: web.Request):
    # проверка секретного токена в заголовке
    secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
    if secret != WEBHOOK_SECRET:
        return web.Response(status=401, text="unauthorized")

    data = await request.json()
    update = Update.model_validate(data)
    # обрабатываем обновление асинхронно, чтобы сразу вернуть 200 OK
    asyncio.create_task(dp.feed_update(bot, update))
    return web.Response(text="ok")

app.add_routes([
    web.get("/health", handle_health),
    web.post(f"/webhook/{WEBHOOK_SECRET}", handle_webhook),
])

@dp.startup()
async def on_startup(dispatcher: Dispatcher):
    # Настроим вебхук у Telegram
    url = f"{PUBLIC_URL}/webhook/{WEBHOOK_SECRET}"
    await bot.set_webhook(url=url, secret_token=WEBHOOK_SECRET)
    logging.info("Webhook set to %s", url)

dp.include_router(router)

if __name__ == "__main__":
    import logging

    async def main():
        await dp.startup()
        logging.info("Starting web server...")
        # запуск aiohttp напрямую, БЕЗ asyncio.run()
        web.run_app(app, host="0.0.0.0", port=PORT)

    import asyncio
    asyncio.get_event_loop().run_until_complete(main())
