import numpy as np


# %%
def AND(x1, x2):
    w1, w2, t = 0.5, 0.5, 0.7
    tmp = w1 * x1 + w2 * x2
    if tmp <= t:
        return 0
    elif tmp > t:
        return 1
    pass


AND(0, 0)
AND(0, 1)
AND(1, 0)
AND(1, 1)

# %%
# 가중치와 편향 도입
x = np.array([0, 1])
w = np.array([0.5, 0.5])
b = -0.7
w * x
np.sum(x * w)
np.sum(x * w) + b


# %%
# 가중치와 편향 구하기
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7

    tmp = np.sum(x*w) + b

    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1
    pass


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7

    tmp = np.sum(x * w) + b

    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1
    pass


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2

    tmp = np.sum(x * w) + b

    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1
    pass


AND(0, 1)
