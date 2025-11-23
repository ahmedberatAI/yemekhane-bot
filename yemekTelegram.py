# -*- coding: utf-8 -*-
"""
Telegram bot that sends Ankara KYK ve Ankara Üniversitesi yemek menülerini CSV'lerden okur
ve günlük olarak Telegram'a yollar.
"""

from __future__ import annotations

import asyncio
import calendar
import logging
import os
import threading
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
from flask import Flask
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    from pytz import timezone as ZoneInfo  # type: ignore

IST = ZoneInfo("Europe/Istanbul")

GENERIC_KEYWORDS: set[str] = {
    "corba",
    "pilav",
    "yemek",
    "salata",
    "tatli",
    "ana yemek",
    "kofte",
}

DATASETS = [
    {
        "name": "Ankara KYK",
        "paths": [Path("ankara_kyk_yemekleri.csv"), Path("kyk_aksam_yemekleri.csv")],
    },
    {
        "name": "Ankara Üniversitesi",
        "paths": [
            Path("ankara_universitesi_yemekleri.csv"),
            Path("ankara_universitesi_ogle_yemekleri.csv"),
            Path("ankara_uni_aksam_yemekleri.csv"),
        ],
    },
]

# kahvalti/ogle/aksam -> sıralama
OGUN_ORDER = {"kahvalti": 0, "ogle": 1, "aksam": 2}

DAY_NAMES_TR = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]


# -----------------------------
# FLASK KEEP-ALIVE SUNUCUSU
# -----------------------------
app = Flask(__name__)


@app.route("/")
def home():
    # Buraya botun canlı olduğunu gösteren basit bir metin yazıyoruz
    return "Bot çalışıyor 🚀"


def run_flask() -> None:
    # Render genelde PORT env veriyor, yoksa 3000
    port = int(os.getenv("PORT", 3000))
    app.run(host="0.0.0.0", port=port)


def start_flask_server() -> None:
    t = threading.Thread(target=run_flask, daemon=True)
    t.start()
    logging.info("Flask keep-alive server başlatıldı.")


# -----------------------------
# ENV YÜKLEME
# -----------------------------
def load_env() -> Tuple[str, str]:
    """
    .env dosyasını script ile aynı klasörden otomatik yükler, eksikleri doğrular.
    - Önce script klasöründeki .env
    - Ardından mevcut çalışma dizini ve ebeveynlerini dolaşır (find_dotenv).
    """
    script_dir = Path(__file__).resolve().parent
    manual_candidates = [script_dir / ".env", Path.cwd() / ".env"]

    seen: set[Path] = set()
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
        logging.info("ENV yüklendi: %s", dotenv_path)
    else:
        logging.warning(".env bulunamadı. Şu yollar denendi: %s", " | ".join(str(p) for p in candidates))
        load_dotenv()

    token = os.getenv("TELEGRAM_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    missing_vars: list[str] = []
    if not token:
        missing_vars.append("TELEGRAM_TOKEN")
    if not chat_id:
        missing_vars.append("TELEGRAM_CHAT_ID")

    if missing_vars:
        raise RuntimeError(
            "Eksik ortam değişkenleri: "
            + ", ".join(missing_vars)
            + ". .env dosyasının script ile aynı klasörde olduğundan ve değerlerin tanımlı olduğundan emin olun."
        )
    return token, chat_id


# -----------------------------
# NORMALİZASYON & YARDIMCI FONKSİYONLAR
# -----------------------------
def _turkish_to_ascii(text: str) -> str:
    mapping = str.maketrans(
        {
            "ç": "c",
            "ğ": "g",
            "ı": "i",
            "ö": "o",
            "ş": "s",
            "ü": "u",
        }
    )
    text = text.translate(mapping)
    # varsa kalan accent'leri de kaldır
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def normalize_ogun(value: str) -> str:
    safe = (value or "").lower().strip()
    safe = _turkish_to_ascii(safe)
    return safe.replace(" ", "")  # "öğle" -> "ogle", "akşam" -> "aksam"


def normalize_text_for_search(value: str) -> str:
    safe = (value or "").lower()
    safe = _turkish_to_ascii(safe)
    return safe.strip()


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
            logging.warning("Dosya bulunamadı: %s", " | ".join(str(p) for p in cfg["paths"]))
            continue

        frames: list[pd.DataFrame] = []
        for path in existing_paths:
            df = load_csv(path)
            frames.append(df)
            logging.info("Yüklendi: %s (%d satır)", path, len(df))

        merged = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        merged = (
            merged.drop_duplicates(subset=["tarih", "ogun", "yemekler"], keep="first")
            .sort_values(["tarih", "ogun"])
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
        return f"{name}: veri bulunamadı.", None

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
        return "Hiç veri yüklenemedi; CSV dosyalarını kontrol edin."

    header_day: Optional[str] = None
    blocks: list[str] = []
    for name, df in sources.items():
        block, day_name = format_dataset_block(name, df, target_date)
        if day_name and not header_day:
            header_day = day_name
        blocks.append(block)

    day_text = resolve_day_name(target_date, header_day)
    header = f"{target_date:%Y-%m-%d} ({day_text}) menüsü"
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


# -----------------------------
# TELEGRAM KOMUTLARI
# -----------------------------
async def send_menu(application: Application, chat_id: str, target_date: date) -> None:
    try:
        message = build_message(target_date)
    except Exception as exc:
        logging.exception("Mesaj oluşturulamadı: %s", exc)
        message = "Menü oluşturulurken bir hata oluştu."
    await application.bot.send_message(chat_id=chat_id, text=message)
    logging.info("Gönderildi -> %s (%s)", chat_id, target_date.isoformat())


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    today = datetime.now(IST).date()
    lines = [
        "Merhaba! /menu veya /bugun ile bugünün menüsünü, /yarin ile yarının menüsünü alabilirsin.",
        "Günlük otomatik gönderim açık.",
        f"Bugünün menüsünü istersen /bugun: {today:%Y-%m-%d}",
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
        await update.message.reply_text("Lütfen tarihi GG/AA/YYYY biçiminde gir: Örneğin /tarih 02/11/2025")
        return

    await send_menu(context.application, str(update.effective_chat.id), target_date)


async def cmd_ara(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return

    if not context.args:
        await update.message.reply_text("Lütfen aramak istediğiniz yemeği yazın. Örnek: /ara Trileçe")
        return

    raw_query = " ".join(context.args).strip()
    if not raw_query:
        await update.message.reply_text("Lütfen aramak istediğiniz yemeği yazın. Örnek: /ara Trileçe")
        return

    if is_generic_query(raw_query):
        await update.message.reply_text(
            "Lütfen daha spesifik bir yemek adı girin. Örneğin: Trileçe, Et Döner, Hamburger gibi."
        )
        return

    normalized_query = normalize_text_for_search(raw_query)
    now = datetime.now(IST)
    sources = load_sources()
    if not sources:
        await update.message.reply_text("Hiç veri yüklenemedi; CSV dosyalarını kontrol edin.")
        return

    matches = search_meals_by_query(normalized_query, sources, now)
    if not any(matches.values()):
        await update.message.reply_text(f'"{raw_query}" bu ayın menülerinde bulunamadı.')
        return

    lines = [f'"{raw_query}" için sonuçlar:']
    dataset_order = [cfg["name"] for cfg in DATASETS]
    for dataset_name in dataset_order:
        entries = matches.get(dataset_name)
        if not entries:
            continue
        lines.append(f"{dataset_name}:")
        for target_date, day_name, ogun_value in entries:
            lines.append(f"- {target_date:%Y-%m-%d} ({day_name}) - {ogun_value}")

    await update.message.reply_text("\n".join(lines))


# -----------------------------
# SCHEDULER KURULUMU
# -----------------------------
async def configure_scheduler(application: Application, chat_id: str) -> None:
    scheduler = AsyncIOScheduler(timezone=IST, event_loop=asyncio.get_running_loop())

    async def daily_job() -> None:
        target = datetime.now(IST).date()
        await send_menu(application, chat_id, target)

    def runner() -> None:
        application.create_task(daily_job())

    trigger = CronTrigger(hour=8, minute=0, timezone=IST)  # 08:00'da gönder
    try:
        scheduler.add_job(runner, trigger, id="daily_menu", replace_existing=True)
    except ConflictingIdError:
        scheduler.reschedule_job("daily_menu", trigger=trigger)

    scheduler.start()
    application.bot_data["scheduler"] = scheduler
    logging.info("Günlük gönderim planlandı: %s", trigger)


# -----------------------------
# MAIN
# -----------------------------
def main() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    logging.info("Çalışma dizini: %s", Path.cwd().resolve())

    # 1) Flask keep-alive server'ı başlat
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
