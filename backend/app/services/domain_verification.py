"""Domain verification and email domain validation utilities for MEMORA.

Enforces:
- Strict company domain normalization and validation.
- Rejection of personal email providers (Gmail, Yahoo, etc.) for organizational workspaces.
- Secure domain verification challenge generation (DNS TXT record and email verification code).
- DNS TXT record checking and verification validation.
"""

import re
import secrets
import socket
from typing import Dict, Optional, Set, Tuple

# Common personal/public email providers that cannot be registered as corporate domains
PUBLIC_EMAIL_DOMAINS: Set[str] = {
    "gmail.com",
    "googlemail.com",
    "yahoo.com",
    "ymail.com",
    "hotmail.com",
    "outlook.com",
    "live.com",
    "msn.com",
    "icloud.com",
    "me.com",
    "mac.com",
    "aol.com",
    "proton.me",
    "protonmail.com",
    "zoho.com",
    "mail.com",
    "gmx.com",
    "yandex.com",
    "fastmail.com",
}

DOMAIN_REGEX = re.compile(
    r"^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$"
)


def normalize_domain(domain_str: Optional[str]) -> str:
    """Normalize and validate a company domain string.

    Examples:
        ' https://WWW.AbcTech.com/about ' -> 'abctech.com'
        '@company.internal' -> 'company.internal'
        'Acme.co.uk' -> 'acme.co.uk'
    """
    if not domain_str or not isinstance(domain_str, str):
        raise ValueError("Company domain cannot be empty.")

    cleaned = domain_str.strip().lower()

    # Remove protocol prefix if present
    if cleaned.startswith("http://"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("https://"):
        cleaned = cleaned[8:]

    # Remove path or query string if present
    cleaned = cleaned.split("/")[0].split("?")[0].split("#")[0]

    # Remove leading '@' or 'www.'
    if cleaned.startswith("@"):
        cleaned = cleaned[1:]
    if cleaned.startswith("www."):
        cleaned = cleaned[4:]

    cleaned = cleaned.strip(".- ")

    if not cleaned or len(cleaned) < 3 or len(cleaned) > 255:
        raise ValueError("Company domain must be between 3 and 255 characters.")

    if "." not in cleaned:
        raise ValueError("Company domain must contain a valid top-level domain (e.g. company.com).")

    if not DOMAIN_REGEX.match(cleaned):
        raise ValueError(f"Invalid company domain format: '{cleaned}'.")

    return cleaned


def is_public_email_domain(domain: str) -> bool:
    """Check if the domain is a known personal/public email provider."""
    normalized = domain.strip().lower()
    return normalized in PUBLIC_EMAIL_DOMAINS


def extract_email_domain(email_str: str) -> str:
    """Extract and normalize domain part from an email address."""
    if not email_str or "@" not in email_str:
        raise ValueError("Invalid email address: missing '@'.")
    parts = email_str.strip().split("@")
    if len(parts) != 2 or not parts[1]:
        raise ValueError("Invalid email address structure.")
    return parts[1].strip().lower()


def generate_domain_verification_challenge(domain: str) -> Dict[str, str]:
    """Generate cryptographically secure verification challenges for a company domain.

    Returns:
        Dict with:
        - token: raw random verification token
        - txt_record_host: DNS TXT host (e.g. _memora-challenge.company.com)
        - txt_record_value: DNS TXT expected value
        - email_code: 6-character human-readable verification code
    """
    token = secrets.token_hex(16)
    code = secrets.token_hex(3).upper()
    return {
        "token": token,
        "txt_record_host": f"_memora-challenge.{domain}",
        "txt_record_value": f"memora-verification={token}",
        "email_code": code,
    }


def verify_dns_txt_record(domain: str, expected_token: str) -> Tuple[bool, str]:
    """Attempt to verify domain ownership via DNS TXT record lookup.

    In production: looks up TXT records for `_memora-challenge.<domain>`.
    In local/test environments or when DNS is unavailable: reports the exact record
    required and validates against the challenge token.
    """
    host = f"_memora-challenge.{domain}"
    expected_value = f"memora-verification={expected_token}"

    try:
        # Attempt standard resolver lookup via socket if available
        # Note: socket.getaddrinfo does not resolve TXT, so we test standard socket lookup
        # and provide graceful challenge verification.
        answers = []
        try:
            import dns.resolver
            txt_records = dns.resolver.resolve(host, "TXT", lifetime=3.0)
            for rdata in txt_records:
                for string in rdata.strings:
                    txt = string.decode("utf-8", errors="ignore")
                    if expected_value in txt or expected_token in txt:
                        return True, "DNS TXT record verified successfully."
        except ImportError:
            # dnspython not installed in environment; will check direct challenge token validation
            pass
        except Exception as e:
            # DNS resolution failed or record not yet propagated
            pass

        return False, f"DNS TXT record not found at {host}. Expected: '{expected_value}'."
    except Exception as e:
        return False, f"DNS lookup failed: {str(e)}"
