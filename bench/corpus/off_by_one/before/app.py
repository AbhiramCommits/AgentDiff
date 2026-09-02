def get_item(items, index):
    if index >= len(items):
        raise IndexError("index out of range")
    return items[index]
