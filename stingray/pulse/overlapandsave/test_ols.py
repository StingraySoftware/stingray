import pytest
import numpy as np
from scipy.signal import fftconvolve
from .ols import prepareh, olsStep, ols, nextprod


@pytest.mark.skipif("not HAS_PYFFTW")
def testPyFFTW_complex():
    nx = 21
    nh = 7
    x = np.random.randint(-30, 30, size=(nx, nx)) + 1j * np.random.randint(-30, 30, size=(nx, nx))
    h = np.random.randint(-20, 20, size=(nh, nh)) + 1j * np.random.randint(-20, 20, size=(nh, nh))
    gold = fftconvolve(x, h, mode="same")

    y = ols(x, h, rfftn=np.fft.fftn, irfftn=np.fft.ifftn)
    # Establish baseline
    assert np.allclose(gold, y)
    # Verify PyFFTW
    import pyfftw.interfaces.numpy_fft as fftw

    # standard, 1 thread
    y2 = ols(x, h, rfftn=fftw.fftn, irfftn=fftw.ifftn)
    assert np.allclose(gold, y2)

    # 2 threads
    def fft(*args, **kwargs):
        return fftw.fftn(*args, **kwargs, threads=2)

    def ifft(*args, **kwargs):
        return fftw.ifftn(*args, **kwargs, threads=2)

    y3 = ols(x, h, rfftn=fft, irfftn=ifft)
    assert np.allclose(gold, y3)


@pytest.mark.skipif("not HAS_PYFFTW")
def testPyFFTW():
    nx = 21
    nh = 7
    x = np.random.randint(-30, 30, size=(nx, nx)) + 1.0
    h = np.random.randint(-20, 20, size=(nh, nh)) + 1.0
    y = ols(x, h)
    gold = fftconvolve(x, h, mode="same")
    # Establish baseline
    assert np.allclose(gold, y)
    # Verify PyFFTW
    import pyfftw.interfaces.numpy_fft as fftw

    # standard, 1 thread
    y2 = ols(x, h, rfftn=fftw.rfftn, irfftn=fftw.irfftn)
    assert np.allclose(gold, y2)

    # 2 threads
    def fft(*args, **kwargs):
        return fftw.rfftn(*args, **kwargs, threads=2)

    def ifft(*args, **kwargs):
        return fftw.irfftn(*args, **kwargs, threads=2)

    y3 = ols(x, h, rfftn=fft, irfftn=ifft)
    assert np.allclose(gold, y3)


def testReflect():
    nx = 21
    nh = 7
    x = np.random.randint(-30, 30, size=(nx, nx)) + 1.0
    h = np.random.randint(-20, 20, size=(nh, nh)) + 1.0
    y = ols(x, h)
    gold = fftconvolve(x, h, mode="same")
    assert np.allclose(gold, y)

    px, py = 24, 28
    x0pad = np.pad(x, [(py, py), (px, px)], mode="constant")
    y0pad = ols(x0pad, h)
    assert np.allclose(y, y0pad[py : py + nx, px : px + nx])

    xpadRef = np.pad(x, [(py, py), (px, px)], mode="reflect")
    ypadRef = ols(xpadRef, h)
    for sizex in [2, 3, 4, 7, 8, 9, 10]:
        for sizey in [2, 3, 4, 7, 8, 9, 10]:
            yRef = ols(x, h, [sizey, sizex], mode="reflect")
            assert np.allclose(yRef, ypadRef[py : py + nx, px : px + nx])


@pytest.mark.slow
def testOls():
    def testouter(nx, nh):
        x = np.random.randint(-30, 30, size=(nx, nx)) + 1.0
        np.save("x", x)
        h = np.random.randint(-20, 20, size=(nh, nh)) + 1.0

        gold = fftconvolve(x, h, "same")

        def testinner(maxlen):
            nfft = [nextprod([2, 3], x) for x in np.array(h.shape) + maxlen - 1]
            hpre = prepareh(h, nfft)

            for xlen0 in range(maxlen):
                for ylen0 in range(maxlen):
                    ylen, xlen = 1 + ylen0, 1 + xlen0
                    dirt = np.vstack(
                        [
                            np.hstack(
                                [
                                    olsStep(x, hpre, [ystart, xstart], [ylen, xlen], nfft, h.shape)
                                    for xstart in range(0, x.shape[0], xlen)
                                ]
                            )
                            for ystart in range(0, x.shape[1], ylen)
                        ]
                    )
                    assert np.allclose(dirt, gold)

                    dirt2 = ols(x, h, [ylen, xlen], nfft)
                    assert np.allclose(dirt2, gold)
                    dirt3 = ols(x, h, [ylen, xlen])
                    assert np.allclose(dirt3, gold)

                    memx = np.lib.format.open_memmap("x.npy")
                    memout = np.lib.format.open_memmap(
                        "out.npy", mode="w+", dtype=x.dtype, shape=x.shape
                    )
                    dirtmem = ols(memx, h, [ylen, xlen], out=memout)
                    assert np.allclose(dirtmem, gold)
                    del memout
                    del memx

                    dirtmem2 = np.load("out.npy")
                    assert np.allclose(dirtmem2, gold)
                    del dirtmem2
                    del dirtmem

        testinner(8)

    for nx in [10, 11]:
        for nh in [3, 4]:
            testouter(nx, nh)


def test1d():
    def testInner(nx, nh):
        x = np.random.randint(-30, 30, size=nx) + 1.0
        h = np.random.randint(-20, 20, size=nh) + 1.0
        gold = fftconvolve(x, h, mode="same")
        for size in [2, 3]:
            dirt = ols(x, h, [size])
            assert np.allclose(gold, dirt)

    for nx in [20, 21]:
        for nh in [9, 10, 17, 18]:
            testInner(nx, nh)


if __name__ == "__main__":  # pragma: no cover
    testPyFFTW_complex()
    testPyFFTW()
    testReflect()
    test1d()
    testOls()
