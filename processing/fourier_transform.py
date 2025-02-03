import numpy as np
    
def get_next_power_of_two(x):
    exponent = np.log2(x)

    return np.ceil(exponent).astype(int)

def fast_fourier_transform(mat: np.ndarray):
    def _fft_recur(polynomial: np.ndarray):
        n = len(polynomial)
        if n == 1:
            return polynomial

        evens, odds = polynomial[::2], polynomial[1::2]
        even_fft, odd_fft = _fft_recur(evens), _fft_recur(odds)

        # compute twiddle factors
        twiddle = np.exp(2*np.pi*1j*np.arange(n//2)/n)*odd_fft

        return np.concat([even_fft+twiddle, even_fft-twiddle])
    
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
        twiddle_inv = np.exp(-2*np.pi*1j*np.arange(n//2)/n)*odd_fft

        return np.concat([even_fft+twiddle_inv, even_fft-twiddle_inv])
    
    polynomial = mat.reshape(-1,)
    n = len(polynomial)
    output = _inverse_fft_recur(polynomial) / n

    return np.round(output)