from app import add_item


def test_add():
    assert add_item(2, [1]) == [1, 2]
