import numpy as np
from pathlib import Path
from scipy import signal


def correlate(x, filt, axis=-1, mode='same'):
    """

    Args:
        x:
        filt:
        axis:
        mode:

    Returns:

    >>> x = np.ones((2, 5, 3)).cumsum(1)
    >>> x[0] **= 2
    >>> meanfilt(x, 3, axis=1).shape
    """
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
    y = y.reshape(shape)
    if axis != x.ndim - 1:
        y = np.swapaxes(y, axis, -1)
    return y


def meanfilt(x, n, axis=-1):
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
    return correlate(x, filt, axis=axis)


def medfilt(x, n, axis=-1):
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
    if axis < 0:
        axis = x.ndim - 1 + axis
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


def join_tsv_files(input_files, output_file):
    with Path(output_file).open('w') as fout:
        fout.write('filename\tonset\toffset\tevent_label\n')
        for file in input_files:
            with Path(file).open() as fin:
                fout.writelines(fin.readlines()[1:])
