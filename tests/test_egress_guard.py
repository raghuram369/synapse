import re

import pytest

EgressGuard = pytest.importorskip("egress_guard").EgressGuard


def _make_guard(mode: str | None = None):
    attempts = []
    if mode is None:
        attempts = [({},)]
    else:
        attempts = [
            ({"sensitivity": mode},),
            ({"mode": mode},),
            ({"level": mode},),
            ({"strict": mode == "strict"},),
            ({},),
        ]

    for (kwargs,) in attempts:
        try:
            return EgressGuard(**kwargs)
        except TypeError:
            continue
    return EgressGuard()


def _sanitize(guard, text: str, mode: str | None = None) -> str:
    method_names = ("filter_context", "sanitize", "filter", "filter_text", "redact", "guard", "apply")
    for name in method_names:
        if not hasattr(guard, name):
            continue
        method = getattr(guard, name)
        if mode is not None:
            for kwargs in (
                {"sensitivity": mode},
                {"mode": mode},
                {"level": mode},
                {"strict": mode == "strict"},
            ):
                try:
                    return method(text, **kwargs)
                except TypeError:
                    continue
        try:
            return method(text)
        except TypeError:
            continue

    if callable(guard):
        if mode is not None:
            for kwargs in (
                {"sensitivity": mode},
                {"mode": mode},
                {"level": mode},
                {"strict": mode == "strict"},
            ):
                try:
                    return guard(text, **kwargs)
                except TypeError:
                    continue
        return guard(text)

    raise AssertionError("Unable to find a supported text filtering API on EgressGuard")


def _redaction_count(text: str) -> int:
    return len(re.findall(r"\[REDACTED[_-][A-Z_]+\]", text))


def test_ssn_redacted():
    guard = _make_guard("standard")
    text = "My SSN is 123-45-6789."
    redacted = _sanitize(guard, text)

    assert "123-45-6789" not in redacted
    assert "[REDACTED_SSN]" in redacted


def test_credit_card_redacted():
    guard = _make_guard("standard")
    text = "Card number 4111 1111 1111 1111 should be hidden."
    redacted = _sanitize(guard, text)

    assert "4111 1111 1111 1111" not in redacted
    assert "[REDACTED" in redacted


def test_email_redacted():
    guard = _make_guard("standard")
    text = "Email me at jane.doe@example.com"
    redacted = _sanitize(guard, text)

    assert "jane.doe@example.com" not in redacted
    assert "[REDACTED_EMAIL]" in redacted


def test_phone_redacted():
    guard = _make_guard("standard")
    text = "Call +1 (415) 555-2671 now."
    redacted = _sanitize(guard, text)

    assert "+1 (415) 555-2671" not in redacted
    assert "[REDACTED_PHONE]" in redacted


def test_ip_address_redacted():
    guard = _make_guard("standard")
    text = "Server is at 192.168.10.42"
    redacted = _sanitize(guard, text)

    assert "192.168.10.42" not in redacted
    assert "[REDACTED_IP]" in redacted


def test_ipv6_redacted():
    guard = _make_guard("standard")
    text = "IPv6 endpoint is 2001:db8:85a3:0000:0000:8a2e:0370:7334"
    redacted = _sanitize(guard, text)

    assert "2001:db8:85a3:0000:0000:8a2e:0370:7334" not in redacted
    assert "[REDACTED_IP]" in redacted


def test_ssn_without_hyphens_redacted():
    guard = _make_guard("standard")
    text = "The value 123456789 should be treated as SSN."
    redacted = _sanitize(guard, text)

    assert "123456789" not in redacted
    assert "[REDACTED_SSN]" in redacted


def test_no_false_positives():
    guard = _make_guard("standard")
    text = "The weather is mild and traffic is normal."
    assert _sanitize(guard, text) == text


def test_multiple_pii_in_text():
    guard = _make_guard("standard")
    text = (
        "SSN 123-45-6789, email bob@example.com, phone 415-555-2671, "
        "card 4111 1111 1111 1111, IP 10.0.0.1"
    )
    redacted = _sanitize(guard, text)

    assert "123-45-6789" not in redacted
    assert "bob@example.com" not in redacted
    assert "415-555-2671" not in redacted
    assert "4111 1111 1111 1111" not in redacted
    assert "10.0.0.1" not in redacted
    assert "[REDACTED_SSN]" in redacted
    assert "[REDACTED_EMAIL]" in redacted
    assert "[REDACTED_PHONE]" in redacted
    assert "[REDACTED_IP]" in redacted


def test_strict_mode():
    text = "Reach me at 4155552671 and jane at example dot com"
    standard = _sanitize(_make_guard("standard"), text, mode="standard")
    strict = _sanitize(_make_guard("strict"), text, mode="strict")

    assert strict != text
    assert _redaction_count(strict) >= _redaction_count(standard)


def test_standard_mode():
    guard = _make_guard("standard")
    text = "Contact me via alice@example.com or 415-555-2671"
    redacted = _sanitize(guard, text, mode="standard")

    assert "alice@example.com" not in redacted
    assert "415-555-2671" not in redacted
    assert "[REDACTED_EMAIL]" in redacted
    assert "[REDACTED_PHONE]" in redacted


def test_empty_input():
    guard = _make_guard("standard")
    assert _sanitize(guard, "") == ""


def test_already_redacted():
    guard = _make_guard("standard")
    text = "User [REDACTED_EMAIL] has ssn [REDACTED_SSN]."
    redacted = _sanitize(guard, text)

    assert redacted == text
