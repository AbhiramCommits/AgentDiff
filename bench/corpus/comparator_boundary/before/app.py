def is_sorted_pairs(pairs):
    for i in range(len(pairs) - 1):
        if pairs[i][0] > pairs[i + 1][0]:
            return False
    return True
