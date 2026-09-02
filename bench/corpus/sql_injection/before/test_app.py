from app import build_query


def test_build_query():
    query, params = build_query(42)
    assert "%s" in query
    assert params == (42,)
