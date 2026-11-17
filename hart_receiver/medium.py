"""The experiment's physical channel: one RC pole followed by Gaussian noise."""

import numpy as np

from .dsp import rc_filter


def propagate(samples, *, pole_hz=4000.0, snr_db=12.0, seed=1):
    """Return the received voltage; None disables the corresponding impairment.

    SNR is per sample, measured over the transmitted interval after RC filtering.
    AWGN is added to the whole recording, including idle intervals. Its variance
    is prescribed in expectation; individual realizations are not rescaled.
    The paper supplies 4 kHz and 12 dB, but not the noise injection point/band;
    adding iid noise after the filter is this model's explicit convention.
    """
    values = np.asarray(samples)
    if (values.ndim != 1 or values.size == 0 or values.dtype.kind not in 'iuf'
            or not np.isfinite(values).all()):
        raise ValueError('Channel input must be a nonempty finite real vector')
    if (not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_))
            or seed < 0):
        raise ValueError('Seed must be a nonnegative integer')
    if snr_db is not None:
        snr = np.asarray(snr_db)
        if snr.shape or snr.dtype.kind not in 'iuf' or not np.isfinite(snr):
            raise ValueError('SNR must be a finite real number')
        snr_db = float(snr)
    received = values.astype(float, copy=True)
    # Locate transmission before filtering: arbitrarily small RC/FFT tails
    # must not make padding count as signal when establishing the noise level.
    peak = np.max(np.abs(received))
    occupied = np.flatnonzero(np.abs(received) > peak*1e-6)
    if pole_hz is not None:
        received = rc_filter(received, pole_hz)
    if snr_db is not None:
        if occupied.size:
            interval = received[occupied[0]:occupied[-1]+1]
            # Normalize before squaring to avoid unnecessary large-value overflow.
            scale = float(np.max(np.abs(interval)))
            mean_square = float(np.mean((interval/scale)**2)) if scale else 0.
        else:
            scale, mean_square = 0., 0.
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            exponent = (np.log10(scale) + 0.5*np.log10(mean_square) - snr_db/20
                        if scale else 0.)
            deviation = np.power(10., exponent) if scale else 0.
        if not np.isfinite(deviation):
            raise ValueError('Noise level overflows; use a higher SNR or lower signal level')
        received += np.random.default_rng(seed).normal(0., deviation, received.size)
    if not np.isfinite(received).all():
        raise ValueError('Channel output overflows')
    return received
