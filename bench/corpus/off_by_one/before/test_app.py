from app import get_item


def test_in_range():
    assert get_item([1, 2, 3], 1) == 2


def test_out_of_range_raises():
    try:
        get_item([1, 2, 3], 3)
    except IndexError:
        return
    raise AssertionError("expected IndexError")
