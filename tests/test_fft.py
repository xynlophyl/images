from processing import fast_fourier_transform, inverse_fast_fourier_transform
import numpy as np
from scipy.fft import fft, ifft

N = 2**10

def is_close(x, y):

    return (np.abs(x - y) < 1e-2).all()

def test_fft_sequence():

    arr = np.random.random(N)

    res = inverse_fast_fourier_transform(fast_fourier_transform(arr, in_place = False), in_place = False)

    assert is_close(arr, res[:N])

def test_fft():

    arr = np.random.random(N)
    res = fast_fourier_transform(arr)
    truth = fft(arr)

    print(res.shape, truth.shape, arr.shape)

    assert is_close(res[:N], truth)

def test_ifft():

    arr = np.random.random(N)
    res = inverse_fast_fourier_transform(arr)
    truth = ifft(arr)

    assert is_close(res[:N], truth)
