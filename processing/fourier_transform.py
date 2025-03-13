import numpy as np
import functools
    
def get_next_power_of_two(x: int) -> int:
    exponent = np.log2(x)

    return np.ceil(exponent).astype(int)

@functools.cache
def generate_twiddle(n: int) -> np.ndarray:
    return np.exp(-2j*np.pi*np.arange(n//2)/n)
  
def fast_fourier_transform(mat: np.ndarray, in_place: bool = True) -> np.ndarray:
    def _fft_recur(polynomial: np.ndarray) -> np.ndarray:
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
    
    def _fft_recur2(polynomial: np.ndarray, start: int = 0):
        """
        create output array at the start and generate solution in-place
            instead of creating a new array for each recursion
        """
        n = len(polynomial)
        if n == 1:
            output[start] = polynomial[0]
        else:
            evens, odds = polynomial[::2], polynomial[1::2]
            _fft_recur2(evens, start= start), _fft_recur2(odds, start = start + n//2)

            even_fft, odd_fft = output[start: start+n//2], output[start+n//2: start+n]

            # compute twiddle factors
            twiddle = generate_twiddle(n)*odd_fft

            output[start:start+n//2], output[start+n//2:start+n] = even_fft+twiddle, even_fft-twiddle

    polynomial = mat.reshape(-1,)
    N = len(polynomial)

    # add padding such that the padded polynomial has length M= 2^x
    M = 2**get_next_power_of_two(N)
    padded_polynomial = np.pad(polynomial, (0, M-N))

    if in_place:
        output = np.zeros_like(padded_polynomial, dtype = np.complex128)
        _fft_recur2(padded_polynomial)
    else:
        output = _fft_recur(padded_polynomial)
    return output

def inverse_fast_fourier_transform(mat: np.ndarray, in_place: bool = True) -> np.ndarray:
    def _inverse_fft_recur(polynomial: np.ndarray) -> np.ndarray:
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

    def _inverse_fft_recur2(polynomial: np.ndarray, start: int = 0) -> np.ndarray:
        n = len(polynomial)
        if n == 1:
            output[start] = polynomial[0]
        else:
            evens, odds = polynomial[::2], polynomial[1::2]
            _inverse_fft_recur2(evens, start= start), _inverse_fft_recur2(odds, start = start + n//2)

            even_fft, odd_fft = output[start: start+n//2], output[start+n//2: start+n]

            # compute twiddle factors
            twiddle_inv = (1/generate_twiddle(n))*odd_fft

            output[start:start+n//2], output[start+n//2:start+n] = even_fft+twiddle_inv, even_fft-twiddle_inv
    
    polynomial = mat.reshape(-1,)
    N = len(polynomial)

    # add padding such that the padded polynomial has length M= 2^x
    M = 2**get_next_power_of_two(N)
    
    padded_polynomial = np.pad(polynomial, (0, M-N))
    
    if in_place:
        output = np.zeros_like(padded_polynomial, dtype = np.complex128)
        _inverse_fft_recur2(padded_polynomial)
    else:
        output = _inverse_fft_recur(padded_polynomial)

    return output / M
