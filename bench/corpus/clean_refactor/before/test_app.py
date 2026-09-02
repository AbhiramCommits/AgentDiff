from app import classify


def test_classify():
    assert classify(-1) == "negative"
    assert classify(0) == "non-negative"
    assert classify(5) == "non-negative"
