from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .scripture_engine import ScriptureSemanticEngine, SemanticEngineUnavailable

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("tib-backend")
_engine: ScriptureSemanticEngine | None = None
_pool = None
_firebase = None


class DeviceRegistration(BaseModel):
    installationId: UUID
    fcmToken: str = Field(min_length=16, max_length=4096)
    appLanguageCode: str = Field(min_length=2, max_length=12, pattern=r"^[a-zA-Z0-9-]+$")
    timezone: str = Field(min_length=1, max_length=64)
    bibleTranslationId: str = Field(default="en_kjv", min_length=1, max_length=80, pattern=r"^[a-zA-Z0-9_-]+$")
    dailyVerseEnabled: bool = False
    dailyVerseTime: str = Field(default="08:00", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    readingReminderEnabled: bool = False
    readingReminderTime: str = Field(default="19:00", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    notificationCopy: dict[str, str] = Field(default_factory=dict)


class DeviceRemoval(BaseModel):
    installationId: UUID
    fcmToken: str = Field(min_length=16, max_length=4096)


def _localized_copy(locale: str, key: str, fallback: str) -> str:
    translations = {
        "en": {"dailyTitle": "Your daily Bible verse", "dailyBody": "Open today's Scripture: {reference}", "reminderTitle": "A moment in Scripture", "reminderBody": "Take a quiet moment to continue reading."},
        "sw": {"dailyTitle": "Mstari wako wa Biblia wa leo", "dailyBody": "Fungua Maandiko ya leo: {reference}", "reminderTitle": "Muda wa Neno la Mungu", "reminderBody": "Tenga muda mfupi kuendelea kusoma."},
        "zh": {"dailyTitle": "\u4eca\u65e5\u5723\u7ecf\u7ecf\u6587", "dailyBody": "\u6253\u5f00\u4eca\u65e5\u7ecf\u6587\uff1a{reference}", "reminderTitle": "\u9759\u5fc3\u9605\u8bfb\u5723\u7ecf", "reminderBody": "\u82b1\u4e00\u70b9\u65f6\u95f4\u7ee7\u7eed\u9605\u8bfb\u5723\u7ecf\u3002"},
        "ko": {"dailyTitle": "\uc624\ub298\uc758 \uc131\uacbd \ub9d0\uc500", "dailyBody": "\uc624\ub298\uc758 \ub9d0\uc500\uc744 \ud655\uc778\ud558\uc138\uc694: {reference}", "reminderTitle": "\uc131\uacbd \uc77d\uae30 \uc2dc\uac04", "reminderBody": "\uc7a0\uc2dc \uc2dc\uac04\uc744 \ub0b4\uc5b4 \uc131\uacbd \uc77d\uae30\ub97c \uc774\uc5b4\uac00\uc138\uc694."},
        "fr": {"dailyTitle": "Votre verset biblique du jour", "dailyBody": "Lire le passage du jour : {reference}", "reminderTitle": "Un moment dans la Bible", "reminderBody": "Prenez un moment pour poursuivre votre lecture."},
        "es": {"dailyTitle": "Tu versículo bíblico de hoy", "dailyBody": "Abre el pasaje de hoy: {reference}", "reminderTitle": "Un momento en la Biblia", "reminderBody": "Dedica un momento a continuar leyendo."},
        "pt": {"dailyTitle": "Seu versículo bíblico de hoje", "dailyBody": "Abra a passagem de hoje: {reference}", "reminderTitle": "Um momento na Bíblia", "reminderBody": "Reserve um momento para continuar a leitura."},
    }
    values = translations.get(locale.split("-")[0], {})
    return values.get(key, fallback)


def _next_at(local_time: str, timezone_name: str, now: datetime | None = None) -> datetime:
    zone = ZoneInfo(timezone_name)
    current = (now or datetime.now(timezone.utc)).astimezone(zone)
    hour, minute = map(int, local_time.split(":"))
    candidate = datetime.combine(current.date(), time(hour, minute), tzinfo=zone)
    if candidate <= current:
        candidate += timedelta(days=1)
    return candidate.astimezone(timezone.utc)


def _initialize_firebase() -> None:
    global _firebase
    raw = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON", "").strip()
    if not raw:
        logger.info("FCM is disabled: no service-account secret configured")
        return
    try:
        import firebase_admin
        from firebase_admin import credentials
        info = json.loads(raw)
        _firebase = firebase_admin.initialize_app(credentials.Certificate(info))
    except Exception:
        logger.exception("Unable to initialize Firebase Admin; push delivery is disabled")


def _init_db() -> None:
    global _pool
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        return
    try:
        from psycopg_pool import ConnectionPool
        url = re.sub(r"^postgres://", "postgresql://", url)
        _pool = ConnectionPool(url, min_size=1, max_size=4, open=True)
        with _pool.connection() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS push_devices (
                    installation_id UUID PRIMARY KEY,
                    fcm_token TEXT NOT NULL UNIQUE,
                    locale VARCHAR(12) NOT NULL,
                    timezone_name VARCHAR(64) NOT NULL,
                    translation_id VARCHAR(80) NOT NULL,
                    daily_enabled BOOLEAN NOT NULL DEFAULT FALSE,
                    daily_time TIME NOT NULL DEFAULT '08:00',
                    reminder_enabled BOOLEAN NOT NULL DEFAULT FALSE,
                    reminder_time TIME NOT NULL DEFAULT '19:00',
                    notification_copy JSONB NOT NULL DEFAULT '{}'::jsonb,
                    next_daily_at TIMESTAMPTZ,
                    next_reminder_at TIMESTAMPTZ,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
    except Exception:
        logger.exception("Push database initialization failed")
        _pool = None


def _config(locale: str) -> dict[str, Any]:
    raw = os.getenv("REMOTE_CONFIG_JSON", "{}").strip()
    try:
        config = json.loads(raw)
        if not isinstance(config, dict):
            config = {}
    except json.JSONDecodeError:
        config = {}
    announcements = config.get("announcements", [])
    localized = []
    if isinstance(announcements, list):
        for item in announcements:
            if not isinstance(item, dict):
                continue
            translations = item.get("translations", {})
            if not isinstance(translations, dict):
                continue
            text = translations.get(locale) or translations.get(locale.split("-")[0]) or translations.get("en")
            if isinstance(text, dict):
                localized.append({"id": str(item.get("id", "")), "title": str(text.get("title", "")), "body": str(text.get("body", ""))})
    flags = config.get("flags", {})
    safe_flags = {str(k): bool(v) for k, v in flags.items() if isinstance(k, str) and isinstance(v, bool)} if isinstance(flags, dict) else {}
    return {"flags": safe_flags, "announcements": localized, "minimumAppVersion": config.get("minimumAppVersion"), "maintenanceMessage": config.get("maintenanceMessage")}


async def _push_loop() -> None:
    while True:
        try:
            await asyncio.to_thread(_send_due_notifications)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Push schedule iteration failed")
        await asyncio.sleep(30)


def _send_due_notifications() -> None:
    if _pool is None or _firebase is None:
        return
    import firebase_admin.messaging as messaging
    now = datetime.now(timezone.utc)
    with _pool.connection() as connection:
        with connection.transaction():
            rows = connection.execute("""
                SELECT installation_id, fcm_token, locale, timezone_name,
                       translation_id, daily_time, notification_copy, daily_enabled,
                       next_daily_at, reminder_time, reminder_enabled, next_reminder_at
                FROM push_devices
                WHERE (daily_enabled AND next_daily_at <= %s)
                   OR (reminder_enabled AND next_reminder_at <= %s)
                ORDER BY LEAST(COALESCE(next_daily_at, 'infinity'), COALESCE(next_reminder_at, 'infinity'))
                LIMIT 100 FOR UPDATE SKIP LOCKED
            """, (now, now)).fetchall()
            for row in rows:
                (installation_id, token, locale, zone, translation_id, daily_time,
                 copy, daily_enabled, next_daily, reminder_time,
                 reminder_enabled, next_reminder) = row
                copy = copy if isinstance(copy, dict) else {}
                if daily_enabled and next_daily and next_daily <= now:
                    try:
                        reference = _engine.daily_reference(now.astimezone(ZoneInfo(zone)).date().isoformat()) if _engine else None
                        if reference:
                            title = copy.get("dailyTitle") or _localized_copy(locale, "dailyTitle", "Your daily Bible verse")
                            body = copy.get("dailyBody") or _localized_copy(locale, "dailyBody", "Open the Bible to read today's verse.")
                            messaging.send(messaging.Message(token=token, notification=messaging.Notification(title=title, body=body), data={"type": "daily_verse", "bookId": reference["bookId"], "chapter": str(reference["chapter"]), "verse": str(reference["verse"]), "translationId": translation_id}, android=messaging.AndroidConfig(priority="normal"), apns=messaging.APNSConfig(headers={"apns-priority": "5"})), app=_firebase)
                    except Exception as error:
                        if error.__class__.__name__ == "UnregisteredError" or "registration-token-not-registered" in str(error).lower() or "invalid-registration-token" in str(error).lower() or "unregistered" in str(error).lower():
                            connection.execute("DELETE FROM push_devices WHERE installation_id = %s", (installation_id,))
                            continue
                        logger.info("Daily push delivery failed")
                    connection.execute("UPDATE push_devices SET next_daily_at = %s WHERE installation_id = %s", (_next_at(daily_time.isoformat(timespec="minutes"), zone, now), installation_id))
                if reminder_enabled and next_reminder and next_reminder <= now:
                    try:
                        title = copy.get("reminderTitle") or _localized_copy(locale, "reminderTitle", "A moment in Scripture")
                        body = copy.get("reminderBody") or _localized_copy(locale, "reminderBody", "Take a quiet moment to continue reading.")
                        messaging.send(messaging.Message(token=token, notification=messaging.Notification(title=title, body=body), data={"type": "reading_reminder"}, android=messaging.AndroidConfig(priority="normal"), apns=messaging.APNSConfig(headers={"apns-priority": "5"})), app=_firebase)
                    except Exception:
                        logger.info("Reminder push delivery failed")
                    connection.execute("UPDATE push_devices SET next_reminder_at = %s WHERE installation_id = %s", (_next_at(reminder_time.isoformat(timespec="minutes"), zone, now), installation_id))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    try:
        _engine = ScriptureSemanticEngine()
    except SemanticEngineUnavailable as error:
        logger.warning("Scripture Connections disabled: %s", error)
    _init_db()
    _initialize_firebase()
    task = asyncio.create_task(_push_loop())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        if _pool is not None:
            _pool.close()


app = FastAPI(title="True International Bible API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "scriptureEngine": _engine is not None, "pushConfigured": _firebase is not None, "databaseConfigured": _pool is not None}


@app.get("/v1/config")
def remote_config(locale: str = Query(default="en", max_length=12, pattern=r"^[a-zA-Z0-9-]+$")) -> dict[str, Any]:
    return _config(locale)


@app.put("/v1/devices/register")
def register_device(payload: DeviceRegistration) -> dict[str, bool]:
    try:
        ZoneInfo(payload.timezone)
    except ZoneInfoNotFoundError as error:
        raise HTTPException(status_code=422, detail="Unsupported timezone") from error
    if _pool is None:
        raise HTTPException(status_code=503, detail="Push registration is temporarily unavailable")
    locale = payload.appLanguageCode.split("-")[0].lower()
    allowed_copy = {key: value[:240] for key, value in payload.notificationCopy.items() if key in {"dailyTitle", "dailyBody", "reminderTitle", "reminderBody"} and isinstance(value, str)}
    next_daily = _next_at(payload.dailyVerseTime, payload.timezone) if payload.dailyVerseEnabled else None
    next_reminder = _next_at(payload.readingReminderTime, payload.timezone) if payload.readingReminderEnabled else None
    with _pool.connection() as connection:
        connection.execute("""
            INSERT INTO push_devices (installation_id, fcm_token, locale, timezone_name, translation_id,
                daily_enabled, daily_time, reminder_enabled, reminder_time, notification_copy,
                next_daily_at, next_reminder_at, updated_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW())
            ON CONFLICT (installation_id) DO UPDATE SET
                fcm_token=EXCLUDED.fcm_token, locale=EXCLUDED.locale, timezone_name=EXCLUDED.timezone_name,
                translation_id=EXCLUDED.translation_id, daily_enabled=EXCLUDED.daily_enabled,
                daily_time=EXCLUDED.daily_time, reminder_enabled=EXCLUDED.reminder_enabled,
                reminder_time=EXCLUDED.reminder_time, notification_copy=EXCLUDED.notification_copy,
                next_daily_at=EXCLUDED.next_daily_at, next_reminder_at=EXCLUDED.next_reminder_at, updated_at=NOW()
        """, (payload.installationId, payload.fcmToken, locale, payload.timezone, payload.bibleTranslationId,
              payload.dailyVerseEnabled, payload.dailyVerseTime, payload.readingReminderEnabled,
              payload.readingReminderTime, json.dumps(allowed_copy), next_daily, next_reminder))
    return {"registered": True}


@app.delete("/v1/devices/register")
def unregister_device(payload: DeviceRemoval) -> dict[str, bool]:
    if _pool is None:
        return {"removed": False}
    with _pool.connection() as connection:
        cursor = connection.execute("DELETE FROM push_devices WHERE installation_id=%s AND fcm_token=%s", (payload.installationId, payload.fcmToken))
    return {"removed": cursor.rowcount > 0}


@app.get("/v1/scripture/related")
def related_verses(
    translationId: str = Query(default="en_kjv", min_length=1, max_length=80, pattern=r"^[a-zA-Z0-9_-]+$"),
    bookId: str = Query(min_length=1, max_length=16, pattern=r"^[a-zA-Z0-9_-]+$"),
    chapter: int = Query(ge=1, le=200),
    verse: int = Query(ge=1, le=300),
    limit: int = Query(default=10, ge=1, le=20),
) -> dict[str, Any]:
    del translationId  # The Flutter client resolves results in its selected local translation.
    if _engine is None:
        raise HTTPException(status_code=503, detail="Scripture Connections index is not available")
    try:
        items = _engine.related((bookId, chapter, verse), limit)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Verse not present in the indexed Bible dataset") from error
    return {"source": {"bookId": bookId, "chapter": chapter, "verse": verse}, "items": items}


@app.get("/v1/premium/entitlements")
def premium_entitlements() -> None:
    # No account-based or client-asserted entitlement is accepted in this phase.
    raise HTTPException(status_code=503, detail="Store billing verification is not configured")


@app.post("/v1/premium/verify")
def verify_premium_proof(payload: dict[str, Any]) -> None:
    # Store billing verification is intentionally a future integration. Never
    # accept client-asserted entitlement names as proof.
    if not isinstance(payload.get("store"), str) or not isinstance(payload.get("purchaseToken"), str):
        raise HTTPException(status_code=422, detail="Store purchase proof is required")
    raise HTTPException(status_code=503, detail="Store billing verification is not configured")
