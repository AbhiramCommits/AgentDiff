from app import safe_divide


def test_ok():
    assert safe_divide(6, 2) == 3.0


def test_zero():
    try:
        safe_divide(1, 0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
