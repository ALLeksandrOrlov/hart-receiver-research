"""C8PSK acquisition and individual components of the tracking feedback loops.

Array-processing blocks consume a complete recording. CarrierLoop and
TimingLoop hold state for one burst and advance once per recovered symbol.
Sample positions refer to the original 25600 Hz grid; phases are in radians.
"""

from dataclasses import dataclass
import numpy as np

from .dsp import sample, tone
from .protocol import C8_POINTS, C8_START, C8_TRAINING


@dataclass(frozen=True)
class Acquisition:
    """First pilot position [samples], gain, phase [rad], and phase step [rad/symbol]."""

    position: float
    amplitude: float
    phase: float
    phase_step: float


class PreambleAcquisition:
    """Complex baseband and burst onset -> initial clock, gain, and carrier state.

    The search is bounded around the onset. Diagnostics describe the current
    attempt, including its training score if a later acquisition check fails.
    """

    def __init__(self, pulse_size=97):
        self.pulse_size = pulse_size
        self.diagnostics = {}

    def process(self, baseband, onset):
        self.diagnostics = {}
        known = C8_POINTS[np.concatenate((C8_TRAINING, C8_START))]
        reference = np.zeros(8*(known.size - 1) + 1, dtype=complex)
        reference[::8] = known
        # Keep the search within the preamble; payload cannot reacquire it.
        radius = self.pulse_size - 1 + 8
        first = max(0, onset - radius)
        last = min(baseband.size - reference.size, onset + radius)
        if last < first:
            raise ValueError('c8psk_training_too_short')
        region = baseband[first:last + reference.size]
        projection = np.correlate(region, reference, 'valid')
        energy = np.correlate(abs(region)**2, (abs(reference) > 0).astype(float), 'valid')
        scores = abs(projection)**2 / np.maximum(known.size*energy, 1e-20)
        integer_position = first + int(np.argmax(scores))
        candidates = np.arange(max(first, integer_position-1), min(last, integer_position+1)+.001, .125)
        symbol_offsets = 8*np.arange(known.size)
        positions = candidates[:, None] + symbol_offsets
        values = sample(baseband, positions.ravel()).reshape(positions.shape)
        normalized = abs(values @ known.conj())**2 / np.maximum(known.size*np.sum(abs(values)**2, axis=1), 1e-20)
        best = int(np.argmax(normalized))
        self.diagnostics['training_score'] = float(normalized[best])
        if normalized[best] < .85:
            raise ValueError('c8psk_training_not_found')
        training = values[best]
        residual_phase = np.unwrap(np.angle(training * known.conj()))
        indices = np.arange(6, 38)
        slope, phase = np.polyfit(indices, residual_phase[indices], 1)
        amplitude = float(np.median(abs(training[indices])))
        marker = training[40:44] * np.exp(-1j*(phase + slope*np.arange(40, 44)))
        labels = np.argmin(abs(marker[:, None]/amplitude - C8_POINTS), axis=1)
        if np.any(abs(marker) < .4*amplitude) or not np.array_equal(labels, C8_START):
            raise ValueError('c8psk_start_marker_not_found')
        position = float(candidates[best])
        self.diagnostics.update(first_symbol_sample=position,
                                carrier_offset_hz=float(slope*3200/(2*np.pi)))
        return Acquisition(position, amplitude, float(phase), float(slope))


class SymbolInterpolator:
    """A sampled recording plus fractional positions -> interpolated values."""

    def __init__(self, values):
        self.values = values

    def process(self, positions):
        return sample(self.values, positions)


class EnvelopePower:
    """Complex baseband voltage -> real squared envelope I² + Q²."""

    @staticmethod
    def process(baseband):
        return abs(baseband)**2


class ClockBandpass:
    """Squared envelope -> complex amplitude of its 3200 Hz clock component.

    Frequency translation followed by averaging over 32 symbol periods is the
    complex equivalent of a narrow clock BPF. The window spans complete clock
    periods, rejecting the squared envelope's DC component.
    """

    @staticmethod
    def process(power):
        clock_carrier = tone(power.size, 3200, sign=-1)
        return np.convolve(power * clock_carrier, np.full(256, 1/256), 'same')


class CarrierLoop:
    """Carrier phase/frequency state for one burst, initialized at payload start.

    correct() rotates and scales a complex symbol sample. advance() accepts
    the angle detector's error [rad] and updates the NCO for the next symbol.
    The two loop gains are model choices, not specified values from the paper.
    """

    def __init__(self, acquired):
        self.amplitude = acquired.amplitude
        self.phase_step = acquired.phase_step
        self.phase = acquired.phase + 44*self.phase_step

    def correct(self, value):
        return value*np.exp(-1j*self.phase)/self.amplitude

    def advance(self, error):
        self.phase_step += .0005*error
        self.phase += self.phase_step + .08*error


class SymbolSlicer:
    """A normalized complex symbol -> nearest C8PSK constellation label 0..7."""

    @staticmethod
    def process(rotated):
        return int(np.argmin(abs(rotated - C8_POINTS)))


class PhaseErrorDetector:
    """Normalized symbol and its decided label -> signed carrier error [rad]."""

    @staticmethod
    def process(rotated, label):
        return float(np.angle(rotated * C8_POINTS[label].conjugate()))


class TimingErrorDetector:
    """Clock component, sampling position and gain -> timing error [samples]."""

    @staticmethod
    def process(reference, position, amplitude):
        error = -np.angle(reference*np.exp(2j*np.pi*position/8))*8/(2*np.pi)
        error = float(np.clip(error, -2., 2.))
        if abs(reference) < 1e-4*amplitude**2:
            error = 0.
        return error


class TimingLoop:
    """Sampling-clock state for one burst, measured on the input sample grid.

    advance(error) applies a timing error [samples] and returns the position
    just used, that error, and the updated period [samples/symbol]. position
    then points to the next symbol. Loop gains are explicit model choices.
    """

    def __init__(self, acquired):
        self.position = acquired.position + 44*8
        self.period = 8.

    def advance(self, error):
        self.period = float(np.clip(self.period + .0004*error, 7.96, 8.04))
        observation = (self.position, error, self.period)
        self.position += self.period + .12*error
        return observation


class BurstEndDetector:
    """Three consecutive symbol samples and acquired gain -> low-envelope flag.

    This only stops sampling. The frame decoder separately requires the
    transmitted C8PSK stop flag before accepting any bytes.
    """

    @staticmethod
    def process(upcoming, amplitude):
        return np.all(abs(upcoming) < .4*amplitude)
