from run import MAX_SIZE


def extend_y(y):
    return [0] * y[0] + [1] * (y[1] - y[0] + 1) + [0] * (MAX_SIZE - y[1] - 1)