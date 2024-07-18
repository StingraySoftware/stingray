"""
The code in this file was copied from
https://github.com/fasiha/array_range
https://github.com/fasiha/nextprod-py
https://github.com/fasiha/overlap_save-py

Thanks to Ahmed Fasih for releasing this to the larger public with the
Unlicense.

# Fast-convolution via overlap-save: a partial drop-in replacement for scipy.signal.fftconvolve

Features:

- 1D and 2D (both tested) and higher (untested) arrays
  - (Currently unsupported is convolving different-dimensional signals)
- Real and complex arrays
- Memory-mapped input/outputs fully supported (and tested)
- Supports alternative FFT engines such as PyFFTW
- Supports reflect-mode (signal assumed to reflect infinitely, instead of 0
  outside its support; useful for avoiding edge effects)
- Relatively straightforward to parallelize each step of the algorithm
- Extensively unit-tested

When it can be used as a drop-in replacement for `fftconvolve`:

- when you call `fftconvolve` with `mode='same'` and `axes=None`
- [See docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.fftconvolve.html)

Example (culled from unit tests):
```py
import numpy as np
from scipy.signal import fftconvolve

# Generate a 100x100 signal array and a 10x10 filter array
nx = 100
nh = 10
x = np.random.randint(-10, 10, size=(nx, nx)) + 1.0
h = np.random.randint(-10, 10, size=(nh, nh)) + 1.0

# Compute the linear convolution using the FFT, keeping the center 100x100 samples
expected = fftconvolve(x, h, mode='same')

# Use overlap-save, computing the output in 6x5-sized chunks. Instead of one huge FFT, we do a
# several tiny ones
from ols import ols
actual = ols(x, h, [6, 5])

# The two will match
assert np.allclose(expected, actual)
```
"""

import numpy as np
from numpy import flip

from itertools import product
from typing import List, Iterable, Tuple

from typing import List


def nextpow(a: float, x: float) -> float:
    """The smallest `a^n` not less than `x`, where `n` is a non-negative integer.

    `a` must be greater than 1, and `x` must be greater than 0.

    Examples
    --------
    >>> assert nextpow(2, 7) == 8.0
    >>> assert nextpow(2, 9) == 16.0
    >>> assert nextpow(5, 20) == 25.0
    >>> assert nextpow(4, 16) == 16.0
    """
    assert x > 0 and a > 1
    if x <= 1:
        return 1.0
    import math

    n = np.ceil(math.log(x, a))
    p = a ** (n - 1)
    return p if p >= x else a**n


def nextprod(a: List[int], x: int) -> int:
    """Find a multiple of a few factors that approximates x

    Next integer greater than or equal to `x` that can be written
    as ``\\prod k_i^{a_i}`` for integers ``a_1``, ``a_2``, etc.

    Examples
    --------
    >>> nextprod([2, 3], 105)
    108
    """
    k = len(a)
    v = [1] * k  # current value of each counter
    mx = [nextpow(ai, x) for ai in a]  # maximum value of each counter
    v[0] = mx[0]  # start at first case that is >= x
    p = mx[0]  # initial value of product in this case
    best = p
    icarry = 1

    while v[-1] < mx[-1]:
        if p >= x:
            best = p if p < best else best  # keep the best found yet
            carrytest = True
            while carrytest:
                p = p // v[icarry - 1]
                v[icarry - 1] = 1
                icarry += 1
                p *= a[icarry - 1]
                v[icarry - 1] *= a[icarry - 1]
                carrytest = v[icarry - 1] > mx[icarry - 1] and icarry < k
            if p < x:
                icarry = 1
        else:
            while p < x:
                p *= a[0]
                v[0] *= a[0]
    return int(mx[-1] if mx[-1] < best else best)


def array_range(start: List[int], stop: List[int], step: List[int]) -> Iterable[Tuple]:
    """Make an iterable of non-overlapping slices.

    Returns an iterable of tuples of slices, each of which can be used to
    index into a multidimensional array such as Numpy's ndarray.

    >> [arr[tup] for tup in array_range([0, 0], arr.shape, [5, 7])]

    where `arr` can be indexed with a tuple of slices (e.g., Numpy), will
    evaluate to a list of sub-arrays.

    Same arguments as `range` except all three arguments are required and
    expected to be list-like of same length. `start` indicates the indexes
    to start each dimension. `stop` indicates the stop index for each
    dimension. `step` is the size of the chunk in each dimension.

    Parameters
    ----------
    start : list of ints
        Starting indices of slices
    stop : list of ints
        Stopping indices of slices
    step : list of ints
        Steps for each slice

    Returns
    -------
    slices : tuple
        Tuple containing all the generated slices.

    Examples
    --------
    >>> list(array_range([0, 0], [10, 10], [5, 7]))
    [(slice(0, 5, None), slice(0, 7, None)),
     (slice(0, 5, None), slice(7, 10, None)),
     (slice(5, 10, None), slice(0, 7, None)),
     (slice(5, 10, None), slice(7, 10, None))]

    """
    assert len(start) == len(stop)
    assert len(stop) == len(step)
    assert all(map(lambda x: x > 0, step))
    startRangesGen = map(lambda v: range(*v), zip(start, stop, step))
    startToSliceMapper = lambda multiStart: tuple(
        slice(i, min(i + step, stop)) for i, stop, step in zip(multiStart, stop, step)
    )
    return map(startToSliceMapper, product(*startRangesGen))


def prepareh(h, nfft: List[int], rfftn=None):
    """Pre-process a filter array.

    Given a real filter array `h` and the length of the FFT `nfft`,
    returns the frequency-domain array. Needs to be computed
    only once before all steps of the overlap-save algorithm run.

    ``rfftn`` defaults to `numpy.fft.rfftn` and may be overridden.

    Parameters
    ----------
    h : array of floats
        Filter array
    nfft : int
        Length of the FFT

    Optional parameters
    -------------------
    rfftn : function
        Substitute of `numpy.fft.rfftn`, provided by the user

    Returns
    -------
    hfftconj : array of complex
        The FFT-transformed, conjugate filter array
    """
    rfftn = rfftn or np.fft.rfftn
    return np.conj(rfftn(flip(np.conj(h)), nfft, axes=np.arange(len(nfft))))


def slice2range(s: slice):
    "Convert slice to range."
    return range(s.start, s.stop, s.step or 1)


def edgesReflect(x, slices):
    """Find the edges of `x` that np.pad in *REFLECT* mode will need.

    Parameters
    ----------
    x : array
        Input array
    slices : list of slice objects
        Input slices

    Returns
    -------
    edges : tuple of slice objects

    """
    starts = [
        0 if s.start < 0 else np.min([s.start, xdim - (s.stop - xdim)])
        for (s, xdim) in zip(slices, x.shape)
    ]
    stops = [
        xdim if s.stop > xdim else np.max([s.stop, -s.start]) for (s, xdim) in zip(slices, x.shape)
    ]
    edges = tuple(slice(lo, hi) for (lo, hi) in zip(starts, stops))
    return edges


def edgesConstant(x, slices):
    """Find the edges of `x` that np.pad in CONSTANT mode will need.

    Parameters
    ----------
    x : array
        Input array
    slices : list of slice objects
        Input slices

    Returns
    -------
    edges : tuple of slice objects
    """
    return tuple(
        slice(np.maximum(0, s.start), np.minimum(xdim, s.stop))
        for (s, xdim) in zip(slices, x.shape)
    )


def padEdges(x, slices, mode="constant", **kwargs):
    """Wrapper around `np.pad`

    This wrapper seeks to call `np.pad` with the smallest amount of data as
    needed, as dictated by `slices`.
    """
    if all(map(lambda s, xdim: s.start >= 0 and s.stop <= xdim, slices, x.shape)):
        return x[slices]
    beforeAfters = [
        (-s.start if s.start < 0 else 0, s.stop - xdim if s.stop > xdim else 0)
        for (s, xdim) in zip(slices, x.shape)
    ]
    if mode == "constant":
        edges = edgesConstant(x, slices)
    elif mode == "reflect":
        edges = edgesReflect(x, slices)
    else:
        assert False

    xpadded = np.pad(x[edges], beforeAfters, mode=mode, **kwargs)
    # we now have an array that's padded just right to the top/left but
    # maybe too big bottom/right
    firsts = tuple(slice(0, len(slice2range(s))) for s in slices)
    return xpadded[firsts]


def olsStep(
    x,
    hfftconj,
    starts: List[int],
    lengths: List[int],
    nfft: List[int],
    nh: List[int],
    rfftn=None,
    irfftn=None,
    mode="constant",
    **kwargs
):
    """Implements a single step of the overlap-save algorithm

    Given an entire signal array `x` and the pre-transformed filter array
    `hfftconj` (i.e., the output of `prepareh`), compute a chunk of the total
    convolution. Specifically, the subarray of the total output starting at
    `starts`, with each dimension's length in `lengths`, is returned. The FFT
    length `nfft` (which was used in `prepareh`) is also required, as is `nh` the
    shape of the filter array (`h.shape`).

    For convenience, `lengths` is treated as a *maximum* length in each dimension,
    so `starts + lengths` is allowed to exceed the total size of `x`: the function
    won't read past the end of any arrays.

    The lists `starts`, `lengths`, `nft`, and `nh` are all required to be the same
    length, matching the number of dimensions of `x` and `hfftconj`.

    If `rfftn` and `irfftn` are not provided, `numpy.fft`'s functions are used.
    This can be overridden to use, e.g., PyFFTW's multi-threaded alternatives.

    `mode` and `**kwargs` are passed to `numpy.pad`, see
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html
    The default, `'constant'` will treat values outside the bounds of `x` as
    constant, and specifically zero. This matches the standard definition of
    convolution. However, other useful alternatives are supported:
    - `'reflect'` where the input `x` is reflected infinitely in all dimensions or

    N.B. These are the only modes supported by this module. Others are
    *UNSUPPORTED*.

    Parameters
    ----------
    x : float/complex array
        This is the array that we need to convolve
    hfftconj : complex array
        filter array, pre-transformed by ``prepareh``
    starts: list of ints
        Starting indices for each dimension
    lengths: list of ints
        Length of interval in each dimension
    nfft: int
        Length of the FFT
    nh: tuple of int
        Shape of filter array

    Optional parameters
    -------------------
    rfftn : function, default None
        Substitute of `numpy.fft.rfftn`, provided by the user
    irfftn : function, default None
        Substitute of `numpy.fft.irfftn`, provided by the user
    mode : str
        The mode of the convolution. The only accepted modes are
        ``constant`` and ``reflect``
    **kwargs : dict
        Keyword arguments to be passed to ``np.pad``

    Returns
    -------
    outarray : array of complex
        The convolved array. The dimension depends on the ``mode``.
        See the docs of `scipy.convolve`
    """
    assert len(x.shape) == len(hfftconj.shape)
    assert len(x.shape) == len(starts) and len(x.shape) == len(lengths)
    assert len(x.shape) == len(nfft) and len(x.shape) == len(nh)
    lengths = np.minimum(np.array(lengths), x.shape - np.array(starts))
    assert np.all(np.array(nfft) >= lengths + np.array(nh) - 1)

    rfftn = rfftn or np.fft.rfftn
    irfftn = irfftn or np.fft.irfftn

    border = np.array(nh) // 2
    slices = tuple(
        slice(start - border, start + length + nh - 1 - border)
        for (start, length, nh, border) in zip(starts, lengths, nh, border)
    )
    xpart = padEdges(x, slices, mode=mode, **kwargs)
    output = irfftn(
        rfftn(xpart, nfft, axes=np.arange(len(nfft))) * hfftconj, nfft, axes=np.arange(len(nfft))
    )
    return output[tuple(slice(0, s) for s in lengths)]


def ols(x, h, size=None, nfft=None, out=None, rfftn=None, irfftn=None, mode="constant", **kwargs):
    """Perform multidimensional overlap-save fast-convolution.

    As mentioned in the module docstring, the output of this function will be
    within machine precision of `scipy.signal.fftconvolve(x, h, mode='same')`.

    However, rather than computing three potentially-large FFTs (one for `x`, one
    for `h`, and an inverse FFT for their product), the overlap-save algorithm
    performs a sequence of smaller FFTs. This makes it appropriate for situations
    where you may not be able to store the signals' FFTs in RAM, or even cases
    where you cannot even store the signals themselves in RAM, i.e., when you have
    to memory-map them from disk.

    `x` and `h` can be multidimensional (1D and 2D are extensively tested), but
    must have the same rank, i.e., `len(x.shape) == len(h.shape)`.

    If provided, the results will be stored in `out`. This is useful for
    memory-mapped outputs, e.g.

    If not provided, `rfftn` and `irfftn` default to those in `numpy.fft`. Other
    implementations matching Numpy's, such as PyFFTW, will also work.

    Parameters
    ----------
    x : float/complex array
        This is the array that we need to convolve
    h : complex array
        Filter array. Must have the same rank as x, i.e.
        `len(x.shape) == len(h.shape)`

    Other Parameters
    ----------------
    size: list of ints
        List of integers that specifies the sizes of the output that will
        be computed in each iteration of the overlap-save algorithm. It must
        be the same length as `x.shape` and `h.shape`. If not provided,
        defaults to `[4 * x for x in h.shape]`, i.e., will break up the output
        into chunks whose size is governed by the size of `h`.
    lengths: list of ints
        Length of interval in each dimension
    nfft: int
        List of integers that specifies the size of the FFT to be used.
        Its length must be equal to the length of `size`. Each element of this
        list must be large enough to store the *linear* convolution, i.e.,
        `all([nfft[i] >= size[i] + h.shape[i] - 1 for i in range(len(nfft))])`
        must be `True`. Set this to a multiple of small prime factors, which
        is the default.
    nh: tuple of int
        Shape of filter array

    Optional parameters
    -------------------
    rfftn : function, default None
        Substitute of `numpy.fft.rfftn`, provided by the user
    irfftn : function, default None
        Substitute of `numpy.fft.irfftn`, provided by the user
    out : array, default None
        If provided, the results are stored here. Useful, e.g., for
        memory-mapped arrays.
    mode : str
        The mode of the convolution. The only accepted modes are
        ``constant`` and ``reflect``. By default, `mode='constant'` assumes
        elements of `x` outside its boundaries are 0, which matches the
        textbook definition of convolution. `mode='reflect'` is also
        supported. It should be straightforward to add support for other modes
        supported by `np.pad`.
    **kwargs : dict
        Keyword arguments to be passed to ``np.pad``

    Returns
    -------
    outarray : array of complex
        The convolved array. The dimension depends on the ``mode``.
        See the docs of `scipy.convolve`
    """
    assert len(x.shape) == len(h.shape)
    size = size or [4 * x for x in h.shape]
    nfft = nfft or [nextprod([2, 3, 5, 7], size + nh - 1) for size, nh in zip(size, h.shape)]
    rfftn = rfftn or np.fft.rfftn
    irfftn = irfftn or np.fft.irfftn
    assert len(x.shape) == len(size)
    assert len(x.shape) == len(nfft)

    hpre = prepareh(h, nfft, rfftn=rfftn)
    if out is None:
        out = np.zeros(x.shape, dtype=x.dtype)

    for tup in array_range([0 for _ in out.shape], out.shape, size):
        out[tup] = olsStep(
            x,
            hpre,
            [s.start for s in tup],
            size,
            nfft,
            h.shape,
            rfftn=rfftn,
            irfftn=irfftn,
            mode=mode,
            **kwargs
        )
    return out
