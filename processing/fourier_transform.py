import numpy as np
import functools
    
def get_next_power_of_two(x):
    exponent = np.log2(x)

    return np.ceil(exponent).astype(int)

@functools.cache
def generate_twiddle(n):
    return np.exp(2*np.pi*1j*np.arange(n//2)/n)

def fast_fourier_transform(mat: np.ndarray):
    def _fft_recur(polynomial: np.ndarray):
        n = len(polynomial)
        if n == 1:
            return polynomial

        evens, odds = polynomial[::2], polynomial[1::2]
        even_fft, odd_fft = _fft_recur(evens), _fft_recur(odds)

        # compute twiddle factors
        twiddle = generate_twiddle(n)*odd_fft

        # generate output of current recursion
        output = np.zeros((n,), dtype = twiddle.dtype)

        output[:n//2] = even_fft+twiddle
        output[n//2:] = even_fft-twiddle

        return output
    
    polynomial = mat.reshape(-1,)
    N = len(polynomial)

    # add padding such that the padded polynomial has length M= 2^x
    M = 2**get_next_power_of_two(N)
    padded_polynomial = np.pad(polynomial, (0, M-N))

    return _fft_recur(padded_polynomial)

def inverse_fast_fourier_transform(mat: np.ndarray):
    def _inverse_fft_recur(polynomial: np.ndarray):
        n = len(polynomial)
        if n == 1:
            return polynomial
        
        evens, odds = polynomial[::2], polynomial[1::2]
        even_fft, odd_fft = _inverse_fft_recur(evens), _inverse_fft_recur(odds)

        # compute twiddle factors
        twiddle_inv = (1/generate_twiddle(n))*odd_fft
        
        # generate output of current recursion
        output = np.zeros((n,), dtype = twiddle_inv.dtype)

        output[:n//2] = even_fft+twiddle_inv
        output[n//2:] = even_fft-twiddle_inv

        return output
    
    polynomial = mat.reshape(-1,)
    n = len(polynomial)
    output = _inverse_fft_recur(polynomial) / n

    return np.round(output)

"""
TODO: 
1. FIGURE OUT HOW TO REMOVE PADDING FOR FINAL OUTPUT IMAGE
2. OPTIMIZE PERFORMANCE (creating the output matrix takes too long currently, figure out a way of allocating the space before running through the recursion?)
"""