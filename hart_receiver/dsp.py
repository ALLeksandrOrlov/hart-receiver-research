"""Numerical building blocks for the sampled HART modem.

Samples represent voltage. Filtering is offline: centered FIRs remove their
known group delay, while the analog RC response keeps its physical phase.
"""

import numpy as np

SAMPLE_RATE = 25_600


def _scalar(value, name):
    item = np.asarray(value)
    if item.shape or item.dtype.kind not in 'iuf' or not np.isfinite(item):
        raise ValueError(f'{name} must be a finite real number')
    return float(item)


def _vector(values, name, *, complex_ok=False):
    array = np.asarray(values)
    kinds = 'iufc' if complex_ok else 'iuf'
    if (array.ndim != 1 or array.size == 0 or array.dtype.kind not in kinds
            or not np.isfinite(array).all()):
        raise ValueError(f'{name} must be a nonempty finite vector')
    return array.astype(complex if np.iscomplexobj(array) else float, copy=False)


def rrc(offset=0.0):
    """A unit-energy, 97-tap RRC pulse: 8 samples/symbol and roll-off 0.5.

    The span is the model's choice. Jin's ROM dimensions do not specify its
    filter coefficients, so the pulse is evaluated analytically in floating point.
    """
    offset = _scalar(offset, 'offset')
    time = (np.arange(-48, 49, dtype=float) - offset) / 8
    beta = 0.5
    pulse = np.empty(time.shape)
    origin = np.isclose(time, 0, rtol=0, atol=1e-12)
    singular = np.isclose(np.abs(time), 1/(4*beta), rtol=0, atol=1e-12)
    regular = ~(origin | singular)
    t = time[regular]
    pulse[regular] = (
        np.sin(np.pi*t*(1-beta)) + 4*beta*t*np.cos(np.pi*t*(1+beta))
    ) / (np.pi*t*(1-(4*beta*t)**2))
    pulse[origin] = 1 + beta*(4/np.pi-1)
    pulse[singular] = beta/np.sqrt(2) * (
        (1+2/np.pi)*np.sin(np.pi/(4*beta))
        + (1-2/np.pi)*np.cos(np.pi/(4*beta))
    )
    length = np.linalg.norm(pulse)
    if not np.isfinite(length) or length == 0:
        raise ValueError('RRC offset is outside the numerical range')
    return pulse / length


def fir(samples, taps):
    """Centered odd-length FIR; zero extension at the record boundaries."""
    values = _vector(samples, 'samples', complex_ok=True)
    coefficients = _vector(taps, 'taps', complex_ok=True)
    if coefficients.size % 2 != 1:
        raise ValueError('A centered FIR needs an odd number of taps')
    delay = coefficients.size // 2
    filtered = np.convolve(values, coefficients)
    return filtered[delay:delay + values.size]


def sample(values, positions):
    """Linearly interpolate a real or complex sequence at fractional indices."""
    values = _vector(values, 'values', complex_ok=True)
    positions = np.asarray(positions)
    if positions.dtype.kind not in 'iuf' or not np.isfinite(positions).all():
        raise ValueError('Sample positions must be finite real numbers')
    return np.interp(positions, np.arange(values.size), values)


def quantize(values, bits, full_scale=1.0):
    """Signed fractional quantization with nearest-even rounding and saturation.

    Paper-specified widths are 8 bits for the ADC and 10 for sine tables.
    Q format, rounding and saturation are explicit modelling choices because
    the paper does not define them. The returned values retain voltage units.
    """
    if (not isinstance(bits, (int, np.integer)) or isinstance(bits, (bool, np.bool_))
            or not 2 <= bits <= 32):
        raise ValueError('Word length must be an integer from 2 to 32')
    full_scale = _scalar(full_scale, 'full_scale')
    if full_scale <= 0:
        raise ValueError('Full scale must be positive')
    values = np.asarray(values)
    if values.dtype.kind not in 'iuf' or not np.isfinite(values).all():
        raise ValueError('Quantizer input must be finite and real')
    magnitude = 2**(int(bits)-1)
    normalized = np.clip(values, -full_scale, full_scale) / full_scale
    codes = np.rint(normalized*magnitude)
    return np.clip(codes, -magnitude, magnitude-1) / magnitude * full_scale


def _lookup(cycles, address_bits):
    size = 1 << address_bits
    phases = 2*np.pi*np.arange(size)/size
    table = quantize(np.cos(phases), 10) + 1j*quantize(np.sin(phases), 10)
    addresses = np.floor(np.remainder(cycles, 1)*size).astype(np.int64)
    return table[addresses % size]


def tone(count, frequency, sign=1, quantized=False):
    """Complex oscillator; optionally use Jin's 512-address, 10-bit ROM.

    Only the lookup address is truncated. The phase increment retains the
    requested frequency; accumulator precision is not specified in the paper.
    """
    if (not isinstance(count, (int, np.integer)) or isinstance(count, (bool, np.bool_))
            or count < 0):
        raise ValueError('Sample count must be a nonnegative integer')
    frequency = _scalar(frequency, 'frequency')
    sign = _scalar(sign, 'sign')
    if abs(frequency) >= SAMPLE_RATE/2 or sign not in (-1, 1):
        raise ValueError('Frequency must be below Nyquist and sign must be -1 or +1')
    cycles = sign*(frequency/SAMPLE_RATE)*np.arange(count)
    return _lookup(cycles, 9) if quantized else np.exp(2j*np.pi*cycles)


def spectrum(samples, quantized=False):
    """Eight 16-point windows; columns are energies at 1600 and 4800 Hz.

    Each window is transformed and squared separately, as in Jin equation (1).
    The channel-estimation ROM has 256 addresses and 10-bit signed values;
    product/accumulator formats are unspecified and remain floating point.
    """
    values = _vector(samples, 'samples')
    if values.size < 128:
        raise ValueError('Spectral estimation requires at least 128 samples')
    windows = values[:128].reshape(8, 16)
    cycles = -np.outer(np.arange(16), np.array([1600, 4800])/SAMPLE_RATE)
    weights = _lookup(cycles, 8) if quantized else np.exp(2j*np.pi*cycles)
    return np.abs(windows @ weights)**2


def rc_filter(samples, pole_hz, inverse=False):
    """Apply H(f)=1/(1+j*f/fp) to a finite bandlimited real recording.

    Zero guards isolate the FFT's periodic copies. The self-conjugate Nyquist
    bin is real. The inverse is useful for preset channel compensation.
    """
    values = _vector(samples, 'samples')
    pole_hz = _scalar(pole_hz, 'pole_hz')
    if not 0 < pole_hz < SAMPLE_RATE/2:
        raise ValueError('RC pole must be between zero and Nyquist')
    with np.errstate(over='ignore', divide='ignore'):
        guard = 20*SAMPLE_RATE/(2*np.pi*pole_hz)
    if not np.isfinite(guard) or guard > 2**20:
        raise ValueError('RC pole is too low for this finite-record model')
    padding = max(512, int(np.ceil(guard)))
    required = values.size + 2*padding
    fft_size = 1 << (int(required)-1).bit_length()
    padded = np.zeros(fft_size)
    padded[padding:padding+values.size] = values
    frequencies = np.fft.rfftfreq(fft_size, d=1/SAMPLE_RATE)
    response = 1 + 1j*frequencies/pole_hz
    if not inverse:
        response = 1/response
    response[-1] = response[-1].real
    transformed = np.fft.rfft(padded)*response
    result = np.fft.irfft(transformed, n=fft_size)
    return result[padding:padding+values.size]
