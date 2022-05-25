import numpy as np
from scipy import signal
from paderbox.array.segment import segment_axis


def correlate(x, filt, axis=-1, mode='same'):
    """

    Args:
        x:
        filt:
        axis:
        mode:

    Returns:

    """
    assert mode in ["valid", "same", "full"], mode
    if axis < 0:
        axis = x.ndim + axis
    if axis != x.ndim - 1:
        x = np.swapaxes(x, axis, -1)
    shape = x.shape
    y = np.apply_along_axis(
        lambda m: np.correlate(m, filt, mode=mode),
        axis=-1, arr=x.reshape((-1, shape[-1]))
    )
    if mode == "full":
        shape = (*shape[:-1], shape[-1]+len(filt)-1)
    elif mode == "valid":
        shape = (*shape[:-1], shape[-1]-(len(filt)-1))
    y = y.reshape(shape)
    if axis != x.ndim - 1:
        y = np.swapaxes(y, axis, -1)
    return y


def meanfilt(x, n, axis=-1, mode='same'):
    """

    Args:
        x:
        n:
        axis:

    Returns:

    >>> x = np.ones((2, 5, 3)).cumsum(1)
    >>> x[0] **= 2
    >>> meanfilt(x, 3, axis=1).shape
    """
    filt = np.ones(n) / n
    return correlate(x, filt, axis=axis, mode=mode)


def medfilt(x, n, axis=-1):
    """

    Args:
        x:
        n:
        axis:

    Returns:

    >>> x = np.ones((2, 5, 3)).cumsum(1)
    >>> x[0] **= 2
    >>> medfilt(x, 3, axis=1).shape
    """
    if n == 1:
        return x
    if axis < 0:
        axis = x.ndim + axis
    if axis != x.ndim - 1:
        x = np.swapaxes(x, axis, -1)
    shape = x.shape
    y = np.apply_along_axis(
        lambda m: signal.medfilt(m, n),
        axis=-1, arr=x.reshape((-1, shape[-1]))
    ).reshape(shape)
    if axis != x.ndim - 1:
        y = np.swapaxes(y, axis, -1)
    return y


def maxfilt(x, n, axis=-1):
    """

    Args:
        x:
        n:
        axis:

    Returns:

    >>> x = np.ones((2, 5, 3)).cumsum(1)
    >>> x[0] **= 2
    >>> maxfilt(x, 3, axis=1).shape
    """
    assert n % 2 == 1, n
    if axis < 0:
        axis = x.ndim + axis
    pad_width = [[0, 0] for _ in range(x.ndim)]
    pad_width[axis] = [(n-1)//2, (n-1)//2]
    x = np.pad(x, pad_width, mode="constant")
    x = segment_axis(
        x, n, shift=1, axis=axis, pad_mode="cut"
    ).max(axis+1)
    return x


def stepfilt(x, n, axis=-1):
    """

    Args:
        x:
        n:
        axis:

    Returns:

    >>> x = np.ones((2, 5, 3)).cumsum(1)
    >>> x[0] **= 2
    >>> stepfilt(x, 4, axis=1).shape
    """
    assert n % 2 == 0
    if axis < 0:
        axis = x.ndim + axis
    step_filter = np.concatenate(
        (-np.ones(n//2), np.ones(n//2))
    ) / (n//2)
    pad_width = [[0, 0] for _ in range(x.ndim)]
    pad_width[axis] = [n//2, n//2-1]
    x = np.pad(x, pad_width, mode="constant")
    return correlate(x, step_filter, axis=axis, mode="valid")
