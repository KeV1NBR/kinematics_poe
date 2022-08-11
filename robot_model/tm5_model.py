import numpy as np

def getModel():
    L1 = .425
    L2 = .3922
    W1 = .1333
    W2 = .0996
    H1 = .1625
    H2 = .0997

    M = np.array([
        [-1., 0., 0., L1+L2],
        [ 0., 0., 1., W1+W2],
        [ 0., 1., 0., H1-H2],
        [ 0., 0., 0.,    1.]])
    Slist = np.array([
        [0.,  0.,  0.,    0.,    0.,    0.],
        [0.,  1.,  1.,    1.,    0.,    1.],
        [1.,  0.,  0.,    0.,   -1.,    0.],
        [0., -H1, -H1,   -H1,   -W1, H2-H1],
        [0.,  0.,  0.,    0., L1+L2,    0.],
        [0.,  0.,  L1, L1+L2,    0., L1+L2]
    ])
    return M, Slist
