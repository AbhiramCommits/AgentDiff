import threading


counter = 0
lock = threading.Lock()


def increment():
    global counter
    with lock:
        counter += 1
