from app import is_sorted_pairs


def test_sorted():
    assert is_sorted_pairs([(1, "a"), (2, "b")]) is True


def test_unsorted():
    assert is_sorted_pairs([(2, "a"), (1, "b")]) is False
