import threading

import app
from app import increment


def test_increment_threads():
    app.counter = 0
    threads = [threading.Thread(target=increment) for _ in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert app.counter == 100
