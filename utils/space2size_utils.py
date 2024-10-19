from gymnasium.spaces.discrete import Discrete


def space2size(space):
    if type(space) is Discrete:
        size = space.n
    else:
        size = 1
        for s in space.shape:
            size *= s
    return size
