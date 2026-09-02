from app import get_email


def test_found():
    assert get_email([{"name": "a", "email": "a@x.io"}], "a") == "a@x.io"


def test_missing_returns_none():
    assert get_email([], "a") is None
