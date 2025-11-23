# pip install python-telegram-bot pandas apscheduler python-dotenv Flask
"""
Telegram bot that sends Ankara KYK ve Ankara �oniversitesi yemek menǬlerini CSV'lerden okur
ve gǬnlǬk olarak Telegram'a yollar. Tek dosya olarak tasarland��; dilersen dosyay�� `bot.py`
ad��yla ��al��Yt��rabilirsin.
"""

from __future__ import annotations

import asyncio
import calendar
import logging
import os
import threading  # <-- YEN��
import unicodedata
from datetime import date, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
from apscheduler.jobstores.base import ConflictingIdError
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# YEN��: Flask keep-alive i��in
from flask import Flask

try:  # Prefer stdlib zoneinfo (Py 3.9+)
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - fallback for older envs
    from pytz import timezone as ZoneInfo  # type: ignore

IST = ZoneInfo("Europe/Istanbul")

GENERIC_KEYWORDS = {
    "��orba",
    "corba",
    "pilav",
    "yemek",
    "salata",
    "tatl��",
    "tatli",
    "ana yemek",
    "k��fte",
    "kofte",
}

DATASETS = [
    {
        "name": "Ankara KYK",
        "paths": [Path("ankara_kyk_yemekleri.csv"), Path("kyk_aksam_yemekleri.csv")],
    },
    {
        "name": "Ankara �oniversitesi",
        "paths": [
            Path("ankara_universitesi_yemekleri.csv"),
            Path("ankara_universitesi_ogle_yemekleri.csv"),
            Path("ankara_uni_aksam_yemekleri.csv"),
        ],
    },
]

OGUN_ORDER = {"kahvalti": 0, "ogle": 1, "��Yle": 1, "aksam": 2, "ak�Yam": 2}
DAY_NAMES_TR = ["Pazartesi", "Sal��", "��ar�Yamba", "Per�Yembe", "Cuma", "Cumartesi", "Pazar"]


# -----------------------------
# FLASK KEEP-ALIVE SUNUCUSU
# -----------------------------
app = Flask(__name__)


@app.route("/")
def home():
    return "Bot ��al��Y��yor �o"��?"


def run_flask():
    # Render genelde PORT env veriyor, yoksa 3000
    port = int(os.getenv("PORT", 3000))
    # 0.0.0.0: d��Y dǬnyadan eri�Yilebilir olsun (Render i��in �Yart)
    app.run(host="0.0.0.0", port=port)


def start_flask_server():
    t = threading.Thread(target=run_flask, daemon=True)
    t.start()
    logging.info("Flask keep-alive server ba�Ylat��ld��.")


def load_env() -> Tuple[str, str]:
    """
    .env dosyas��n�� script ile ayn�� klas��rden otomatik yǬkler, eksikleri do�Yrular.
    - �-nce script klas��rǬndeki .env
    - Ard��ndan mevcut ��al��Yma dizini ve ebeveynlerini dola�Y��r (find_dotenv)
    """
    script_dir = Path(__file__).resolve().parent
    manual_candidates = [script_dir / ".env", Path.cwd() / ".env"]

    # Uniq ve s��ral�� liste
    seen = set()
    candidates: list[Path] = []
    for c in manual_candidates:
        if c not in seen:
            candidates.append(c)
            seen.add(c)

    dotenv_path: Optional[Path] = None
    for candidate in candidates:
        if candidate.exists():
            dotenv_path = candidate
            break

    # find_dotenv ile yukar�� do�Yru tarama (��r. OneDrive alt klas��rǬnde ��al��Y��rken)
    if not dotenv_path:
        try:
            from dotenv import find_dotenv

            found = find_dotenv(usecwd=True)
            if found:
                fp = Path(found)
                if fp.exists():
                    dotenv_path = fp
                    candidates.append(fp)
        except Exception:
            pass

    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path)
        logging.info("ENV yǬklendi: %s", dotenv_path)
    else:
        logging.warning(".env bulunamad��. �?u yollar denendi: %s", " | ".join(str(p) for p in candidates))
        load_dotenv()  # yoksay��labilir, default davran��Y

    token = os.getenv("TELEGRAM_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    missing_vars = []
    if not token:
        missing_vars.append("TELEGRAM_TOKEN")
    if not chat_id:
        missing_vars.append("TELEGRAM_CHAT_ID")
    if missing_vars:
        raise RuntimeError(
            "Eksik ortam de�Yi�Ykenleri: "
            + ", ".join(missing_vars)
            + ". .env dosyas��n��n script ile ayn�� klas��rde oldu�Yundan ve de�Yerlerin tan��ml�� oldu�Yundan emin olun."
        )
    return token, chat_id


def normalize_ogun(value: str) -> str:
    safe = (value or "").lower()
    return (
        safe.replace("�Y", "g")
        .replace("��", "i")
        .replace("�Y", "s")
        .replace("��", "o")
        .replace("��", "c")
        .replace("Ǭ", "u")
    )


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    df["tarih"] = pd.to_datetime(df["tarih"], errors="coerce")
    df = df.dropna(subset=["tarih"])
    df["tarih"] = df["tarih"].dt.tz_localize(None).dt.date
    return df


def load_sources() -> Dict[str, pd.DataFrame]:
    sources: Dict[str, pd.DataFrame] = {}
    for cfg in DATASETS:
        existing_paths = [p for p in cfg["paths"] if p.exists()]
        if not existing_paths:
            logging.warning("Dosya bulunamad��: %s", " | ".join(str(p) for p in cfg["paths"]))
            continue
        frames: list[pd.DataFrame] = []
        for path in existing_paths:
            df = load_csv(path)
            frames.append(df)
            logging.info("YǬklendi: %s (%d sat��r)", path, len(df))

        merged = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        merged = merged.drop_duplicates(subset=["tarih", "ogun", "yemekler"], keep="first").sort_values(
            ["tarih", "ogun"]
        )
        sources[cfg["name"]] = merged
    return sources


def resolve_day_name(target_date: date, day_value: str | None = None) -> str:
    cleaned = (day_value or "").strip()
    if cleaned:
        return cleaned
    return DAY_NAMES_TR[target_date.weekday()]


def format_dataset_block(name: str, df: pd.DataFrame, target_date: date) -> Tuple[str, Optional[str]]:
    day_df = df[df["tarih"] == target_date]
    if day_df.empty:
        return f"{name}: veri bulunamad��.", None

    day_name = resolve_day_name(target_date, str(day_df.iloc[0].get("gun", "")))
    ordered = (
        day_df.assign(_order=day_df["ogun"].apply(lambda x: OGUN_ORDER.get(normalize_ogun(str(x)), 99)))
        .sort_values(["_order", "ogun"])
        .drop(columns="_order")
    )

    lines = [f"{name}:"]
    for _, row in ordered.iterrows():
        ogun = str(row.get("ogun", "")).strip()
        yemekler = str(row.get("yemekler", "")).strip()
        kalem = row.get("kalem_sayisi", 0)
        try:
            kalem_int = int(kalem)
        except Exception:
            kalem_int = 0
        lines.append(f"- {ogun} ({kalem_int}): {yemekler}")

    source = str(ordered.iloc[0].get("kaynak", "")).strip()
    if source:
        lines.append(f"Kaynak: {source}")

    return "\n".join(lines), day_name


def build_message(target_date: date) -> str:
    sources = load_sources()
    if not sources:
        return "Hi�� veri yǬklenemedi; CSV dosyalar��n�� kontrol edin."

    header_day: Optional[str] = None
    blocks: list[str] = []
    for name, df in sources.items():
        block, day_name = format_dataset_block(name, df, target_date)
        header_day = header_day or day_name
        blocks.append(block)

    day_text = resolve_day_name(target_date, header_day)
    header = f"{target_date:%Y-%m-%d} ({day_text}) menǬsǬ"
    return f"{header}\n\n" + "\n\n".join(blocks)


def parse_user_date_arg(raw_date: str) -> Optional[date]:
    cleaned = (raw_date or "").strip()
    if not cleaned:
        return None
    try:
        parsed = datetime.strptime(cleaned, "%d/%m/%Y").replace(tzinfo=IST)
    except Exception:
        return None
    return parsed.date()


def normalize_text_for_search(value: str) -> str:
    transliterated = (
        unicodedata.normalize("NFKD", (value or "").lower())
        .replace("��", "c")
        .replace("�Y", "g")
        .replace("��", "i")
        .replace("��", "o")
        .replace("�Y", "s")
        .replace("Ǭ", "u")
        .replace("ǽ", "a")
        .replace("ǩ", "i")
        .replace("ǯ", "u")
    )
    return transliterated.strip()


def is_generic_query(query: str) -> bool:
    normalized = normalize_text_for_search(query)
    return normalized in GENERIC_KEYWORDS


def month_bounds(now: datetime) -> Tuple[date, date]:
    first_day = date(now.year, now.month, 1)
    last_day = date(now.year, now.month, calendar.monthrange(now.year, now.month)[1])
    return first_day, last_day


def search_meals_by_query(
    normalized_query: str, sources: Dict[str, pd.DataFrame], now: datetime
) -> Dict[str, list[Tuple[date, str, str]]]:
    start, end = month_bounds(now)
    results: Dict[str, list[Tuple[date, str, str]]] = {}

    for name, df in sources.items():
        month_df = df[(df["tarih"] >= start) & (df["tarih"] <= end)]
        matches: list[Tuple[date, str, str]] = []
        for _, row in month_df.iterrows():
            yemekler_text = normalize_text_for_search(str(row.get("yemekler", "")))
            if normalized_query in yemekler_text:
                target_date = row.get("tarih")
                if not isinstance(target_date, date):
                    continue
                ogun_value = str(row.get("ogun", "")).strip()
                day_name = resolve_day_name(target_date, str(row.get("gun", "")))
                matches.append((target_date, day_name, ogun_value))

        matches.sort(key=lambda item: (item[0], OGUN_ORDER.get(normalize_ogun(item[2]), 99), item[2]))
        if matches:
            results[name] = matches
    return results


async def send_menu(application: Application, chat_id: str, target_date: date) -> None:
    try:
        message = build_message(target_date)
    except Exception as exc:
        logging.exception("Mesaj olu�Yturulamad��: %s", exc)
        message = "MenǬ olu�Yturulurken bir hata olu�Ytu."
    await application.bot.send_message(chat_id=chat_id, text=message)
    logging.info("G��nderildi -> %s (%s)", chat_id, target_date.isoformat())


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    today = datetime.now(IST).date()
    lines = [
        "Merhaba! /menu veya /bugun ile bugǬnǬn menǬsǬnǬ, /yarin ile yar��n��n menǬsǬnǬ alabilirsin.",
        "GǬnlǬk otomatik g��nderim a����k.",
        f"BugǬnǬn menǬsǬnǬ istersen /bugun: {today:%Y-%m-%d}",
    ]
    await update.message.reply_text("\n".join(lines))


async def cmd_bugun(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    target = datetime.now(IST).date()
    await send_menu(context.application, str(update.effective_chat.id), target)


async def cmd_yarin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    target = datetime.now(IST).date() + timedelta(days=1)
    await send_menu(context.application, str(update.effective_chat.id), target)


async def cmd_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    target = datetime.now(IST).date()
    await send_menu(context.application, str(update.effective_chat.id), target)


async def cmd_tarih(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_chat:
        return

    date_arg = " ".join(context.args) if context.args else ""
    target_date = parse_user_date_arg(date_arg)
    if not target_date:
        await update.message.reply_text("LǬtfen tarihi GG/AA/YYYY bi��iminde gir: �-rne�Yin /tarih 02/11/2025")
        return

    await send_menu(context.application, str(update.effective_chat.id), target_date)


async def cmd_ara(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return

    if not context.args:
        await update.message.reply_text("LǬtfen aramak istedi�Yiniz yeme�Yi yaz��n. �-rnek: /ara Trile��e")
        return

    raw_query = " ".join(context.args).strip()
    if not raw_query:
        await update.message.reply_text("LǬtfen aramak istedi�Yiniz yeme�Yi yaz��n. �-rnek: /ara Trile��e")
        return

    if is_generic_query(raw_query):
        await update.message.reply_text(
            "LǬtfen daha spesifik bir yemek ad�� girin. �-rne�Yin: Trile��e, Et D��ner, Hamburger gibi."
        )
        return

    normalized_query = normalize_text_for_search(raw_query)
    now = datetime.now(IST)
    sources = load_sources()
    if not sources:
        await update.message.reply_text("Hi�� veri yǬklenemedi; CSV dosyalar��n�� kontrol edin.")
        return

    matches = search_meals_by_query(normalized_query, sources, now)
    if not any(matches.values()):
        await update.message.reply_text(f'"{raw_query}" bu ay��n menǬlerinde bulunamad��.')
        return

    lines = [f'"{raw_query}" i��in sonu��lar:']
    dataset_order = [cfg["name"] for cfg in DATASETS]
    for dataset_name in dataset_order:
        entries = matches.get(dataset_name)
        if not entries:
            continue
        lines.append(f"{dataset_name}:")
        for target_date, day_name, ogun_value in entries:
            lines.append(f"- {target_date:%Y-%m-%d} ({day_name}) - {ogun_value}")

    await update.message.reply_text("\n".join(lines))


async def configure_scheduler(application: Application, chat_id: str) -> None:
    scheduler = AsyncIOScheduler(timezone=IST, event_loop=asyncio.get_running_loop())

    async def daily_job() -> None:
        target = datetime.now(IST).date()
        await send_menu(application, chat_id, target)

    def runner() -> None:
        application.create_task(daily_job())

    trigger = CronTrigger(hour=8, minute=0, timezone=IST)  # 08:00'da g��nder
    try:
        scheduler.add_job(runner, trigger, id="daily_menu", replace_existing=True)
    except ConflictingIdError:
        scheduler.reschedule_job("daily_menu", trigger=trigger)

    scheduler.start()
    application.bot_data["scheduler"] = scheduler
    logging.info("GǬnlǬk g��nderim planland��: %s", trigger)


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    logging.info("��al��Yma dizini: %s", Path.cwd().resolve())

    # 1) Flask keep-alive server'�� ba�Ylat
    start_flask_server()

    # 2) Telegram + scheduler kur
    token, chat_id = load_env()
    application = (
        Application.builder()
        .token(token)
        .post_init(partial(configure_scheduler, chat_id=chat_id))
        .build()
    )

    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("bugun", cmd_bugun))
    application.add_handler(CommandHandler("yarin", cmd_yarin))
    application.add_handler(CommandHandler("menu", cmd_menu))
    application.add_handler(CommandHandler("tarih", cmd_tarih))
    application.add_handler(CommandHandler("ara", cmd_ara))

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
