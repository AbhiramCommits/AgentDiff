from app import read_first_line


def test_read(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("hello\nworld\n")
    assert read_first_line(p) == "hello"
