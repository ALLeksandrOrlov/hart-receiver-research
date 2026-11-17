"""Construct one voltage recording from a validated HART PDU."""

import numpy as np

from .dsp import SAMPLE_RATE, rrc, tone
from .protocol import (C8_POINTS, C8_START, C8_TRAINING, c8_encode,
                       uart_encode, validate_frame)


def _fsk(frame, phase):
    bits = uart_encode(b'\xff' * 5 + frame)
    seconds_per_bit = 1 / 1200
    frequency = 2200 - 1000 * bits.astype(float)
    times = np.arange(int(np.ceil(len(bits) * SAMPLE_RATE / 1200))) / SAMPLE_RATE
    symbol = np.minimum((times * 1200).astype(int), len(bits) - 1)
    phase_at_start = np.concatenate(([0.], np.cumsum(frequency * seconds_per_bit)))
    cycles = phase_at_start[symbol] + frequency[symbol] * (times - symbol * seconds_per_bit)
    return np.cos(2 * np.pi * cycles + phase)


def _c8psk(frame, phase):
    symbols = np.concatenate((C8_TRAINING, C8_START, c8_encode(frame)))
    impulses = np.zeros(8 * (len(symbols) - 1) + 1, dtype=complex)
    impulses[::8] = C8_POINTS[symbols]
    envelope = np.convolve(impulses, rrc()) * np.sqrt(8)
    carrier = tone(len(envelope), 3200) * np.exp(1j * phase)
    return np.real(envelope * carrier)


def transmit(pdu, mode, *, phase=0., fsk_mvpp=200., c8_dbm=-6.):
    """Return 25600 samples/s in volts, with acquisition and trailing silence.

    FSK level is peak-to-peak millivolts. C8PSK level is nominal dBm into
    500 ohms; RRC shaping determines the actual instantaneous envelope.
    """
    frame = validate_frame(pdu, mode)
    if not np.isfinite(phase):
        raise ValueError('Phase must be finite')
    if mode == 'fsk':
        if not np.isfinite(fsk_mvpp) or fsk_mvpp <= 0:
            raise ValueError('FSK level must be finite and positive')
        amplitude = fsk_mvpp / 2000
        samples = _fsk(frame, phase)
    else:
        if not np.isfinite(c8_dbm):
            raise ValueError('C8PSK level must be finite')
        with np.errstate(over='ignore', under='ignore'):
            amplitude = np.sqrt(2 * 500 * .001 * np.power(10., c8_dbm / 10))
        if not np.isfinite(amplitude) or amplitude <= 0:
            raise ValueError('C8PSK level is outside the numerical range')
        samples = _c8psk(frame, phase)
    recording = np.concatenate((np.zeros(137), amplitude * samples, np.zeros(256)))
    if not np.isfinite(recording).all():
        raise ValueError('Signal level overflow')
    return recording
