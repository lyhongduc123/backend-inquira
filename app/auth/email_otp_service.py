"""
Email OTP service for passwordless login/signup.
"""
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Literal
import httpx

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import BadRequestException, UnauthorizedException
from app.extensions.logger import create_logger
from app.models.email_otps import DBEmailOtp

logger = create_logger(__name__)

OtpMode = Literal["login", "signup"]


async def invalidate_active_email_otps(db: AsyncSession, email: str, mode: OtpMode) -> int:
    """Invalidate all active OTP records for a given email and mode."""
    normalized_email = _normalize_email(email)
    now = datetime.now(timezone.utc)

    stmt = (
        select(DBEmailOtp)
        .where(
            DBEmailOtp.email == normalized_email,
            DBEmailOtp.mode == mode,
            DBEmailOtp.consumed_at.is_(None),
        )
        .order_by(DBEmailOtp.created_at.desc())
    )
    records = (await db.execute(stmt)).scalars().all()

    invalidated = 0
    for record in records:
        record.consumed_at = now
        invalidated += 1

    return invalidated


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def _hash_otp(email: str, otp: str) -> str:
    value = f"{_normalize_email(email)}:{otp}:{settings.JWT_SECRET_KEY}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def generate_otp_code() -> str:
    """Generate a 6-digit OTP."""
    return str(secrets.randbelow(900000) + 100000)


async def send_otp_email(email: str, otp_code: str, mode: OtpMode, expires_minutes: int) -> None:
    """Send OTP email using Resend API; fallback to logging when not configured."""
    subject = f"Your {settings.APP_NAME} verification code"
    action = "sign in" if mode == "login" else "create your account"
    text_body = (
        f"Your verification code is: {otp_code}\n\n"
        f"Use this code to {action} on {settings.APP_NAME}.\n"
        f"The code expires in {expires_minutes} minutes."
    )
    html_body = (
        "<div style='font-family:Arial,sans-serif;line-height:1.5'>"
        f"<h2>Your {settings.APP_NAME} verification code</h2>"
        f"<p>Use this code to {action}:</p>"
        f"<p style='font-size:24px;font-weight:700;letter-spacing:2px'>{otp_code}</p>"
        f"<p>This code expires in {expires_minutes} minutes.</p>"
        "</div>"
    )

    if not settings.RESEND_API_KEY or not settings.RESEND_FROM_EMAIL:
        logger.warning(
            "Resend is not configured. OTP for %s (%s): %s",
            email,
            mode,
            otp_code,
        )
        return

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                settings.RESEND_API_URL,
                headers={
                    "Authorization": f"Bearer {settings.RESEND_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "from": settings.RESEND_FROM_EMAIL,
                    "to": [email],
                    "subject": subject,
                    "text": text_body,
                    "html": html_body,
                },
            )

        if response.status_code >= 400:
            logger.error(
                "Resend API failed (%s): %s",
                response.status_code,
                response.text,
            )
            raise BadRequestException("Could not send verification code. Please try again.")
    except Exception as exc:
        logger.error("Failed to send OTP email with Resend: %s", exc, exc_info=True)
        raise BadRequestException("Could not send verification code. Please try again.")


async def create_email_otp(db: AsyncSession, email: str, mode: OtpMode) -> tuple[DBEmailOtp, str]:
    """Create a new OTP record with cooldown checks."""
    normalized_email = _normalize_email(email)
    now = datetime.now(timezone.utc)

    cooldown_cutoff = now - timedelta(seconds=settings.EMAIL_OTP_RESEND_COOLDOWN_SECONDS)
    recent_stmt = (
        select(DBEmailOtp)
        .where(
            DBEmailOtp.email == normalized_email,
            DBEmailOtp.mode == mode,
            DBEmailOtp.consumed_at.is_(None),
            DBEmailOtp.created_at >= cooldown_cutoff,
        )
        .order_by(DBEmailOtp.created_at.desc())
    )
    recent = (await db.execute(recent_stmt.limit(1))).scalars().first()
    if recent:
        raise BadRequestException(
            f"Please wait {settings.EMAIL_OTP_RESEND_COOLDOWN_SECONDS} seconds before requesting another code."
        )

    await invalidate_active_email_otps(db, normalized_email, mode)

    otp_code = generate_otp_code()
    otp_hash = _hash_otp(normalized_email, otp_code)

    record = DBEmailOtp(
        email=normalized_email,
        otp_hash=otp_hash,
        mode=mode,
        attempts_used=0,
        max_attempts=settings.EMAIL_OTP_MAX_ATTEMPTS,
        expires_at=now + timedelta(minutes=settings.EMAIL_OTP_EXPIRE_MINUTES),
        consumed_at=None,
    )

    db.add(record)
    await db.commit()
    await db.refresh(record)

    return record, otp_code


async def verify_email_otp(
    db: AsyncSession,
    email: str,
    otp: str,
    mode: OtpMode,
    consume_on_success: bool = True,
) -> bool:
    """Validate OTP and consume it on success."""
    normalized_email = _normalize_email(email)
    now = datetime.now(timezone.utc)

    stmt = (
        select(DBEmailOtp)
        .where(
            DBEmailOtp.email == normalized_email,
            DBEmailOtp.mode == mode,
            DBEmailOtp.consumed_at.is_(None),
        )
        .order_by(DBEmailOtp.created_at.desc())
    )
    record = (await db.execute(stmt.limit(1))).scalars().first()

    if not record:
        raise UnauthorizedException("No active verification code. Please request a new code.")

    expires_at = record.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)

    if expires_at < now:
        record.consumed_at = now
        await db.commit()
        raise UnauthorizedException("Verification code has expired. Please request a new code.")

    if record.attempts_used >= record.max_attempts:
        raise UnauthorizedException("Too many invalid attempts. Please request a new code.")

    expected_hash = _hash_otp(normalized_email, otp)
    if not secrets.compare_digest(expected_hash, record.otp_hash):
        record.attempts_used += 1
        if record.attempts_used >= record.max_attempts:
            record.consumed_at = now
        await db.commit()
        raise UnauthorizedException("Invalid verification code")

    if consume_on_success:
        record.consumed_at = now
    await db.commit()
    return True
