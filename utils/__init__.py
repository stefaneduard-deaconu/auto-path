def fibn_generator(n: int):
    """
    generates first n fibonacci numbers.
    """
    # edge cases
    if n == 0:
        return 1
    elif n == 1:
        return 1, 1
    # general case
    f0, f1 = 1, 1
    yield 1  # for i=0
    yield 1  # for i=1
    for i in range(2, n):
        fn = f0 + f1
        yield fn
        f0 = f1
        f1 = fn