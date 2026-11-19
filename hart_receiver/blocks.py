"""Whole-record signal-processing blocks, using the common 25600 Hz grid.

Each ``process`` call consumes and returns numerical arrays or a small
measurement record. Frame decoding, tracing and file I/O belong to the
receiver that connects these blocks. FIRs retain the existing offline,
zero-extended boundary convention; these are not streaming chunk processors.
"""

from dataclasses import dataclass
import numpy as np

from .dsp import SAMPLE_RATE, fir, quantize, rc_filter, rrc, spectrum, tone


@dataclass(frozen=True)
class CarrierDetection:
    """Burst onset in input samples, running RMS in volts, and its peak."""

    onset: int
    rms: np.ndarray
    peak: float


@dataclass(frozen=True)
class SpectralEstimate:
    """Start sample, eight pairs of DFT energies, and E1600/E4800."""

    start: int
    energies: np.ndarray
    ratio: float


class InputADC:
    """Finite real voltage record -> voltage record, optionally quantized.

    An automatic burst-wide range reserves 10% headroom for the 8-bit ADC.
    ``used_full_scale`` reports that range in volts for the latest record;
    quantization returns physical voltages, not integer ADC codes.
    """

    def __init__(self, quantized=False, full_scale=None):
        if full_scale is not None:
            if not np.isfinite(full_scale) or full_scale <= 0:
                raise ValueError('ADC full scale must be positive and finite')
            if not quantized:
                raise ValueError('An ADC range requires quantized reception')
        self.quantized = quantized
        self.full_scale = full_scale
        self.used_full_scale = None

    def process(self, waveform):
        self.used_full_scale = None
        samples = np.asarray(waveform)
        if (samples.ndim != 1 or samples.dtype.kind not in 'fiu'
                or not np.all(np.isfinite(samples))):
            raise ValueError('Expected finite real voltage samples')
        if samples.size < 256:
            raise ValueError('record_too_short')
        samples = samples.astype(float, copy=True)
        if self.quantized:
            full_scale = self.full_scale
            if full_scale is None:
                full_scale = max(float(np.max(np.abs(samples))) / .9, 1e-12)
            samples = quantize(samples, 8, full_scale=full_scale)
            self.used_full_scale = full_scale
        return samples


class CarrierDetector:
    """Real voltage samples -> sustained-carrier onset and RMS envelope."""

    def process(self, samples):
        with np.errstate(over='ignore'):
            power = samples * samples
        if not np.all(np.isfinite(power)):
            raise ValueError('signal_power_overflow')
        rms = np.sqrt(np.maximum(fir(power, np.full(17, 1/17)), 0))
        peak = float(rms.max())
        if peak < 1e-5:
            raise ValueError('no_carrier')
        present = rms > max(1e-5, .35 * peak)
        runs = np.convolve(present.astype(int), np.ones(24, dtype=int), 'valid')
        positions = np.flatnonzero(runs == 24)
        if positions.size == 0:
            raise ValueError('no_sustained_carrier')
        return CarrierDetection(int(positions[0]), rms, peak)


class PreambleSpectrum:
    """Voltage samples and onset -> DFT energies at 1600 and 4800 Hz."""

    def __init__(self, quantized=False):
        self.quantized = quantized

    def process(self, samples, onset):
        beginning = onset + 64
        preamble = samples[beginning:beginning + 128]
        if preamble.size != 128:
            raise ValueError('preamble_too_short')
        energies = spectrum(preamble, quantized=self.quantized)
        total = energies.sum(axis=0)
        ratio = float(total[0] / max(total[1], np.finfo(float).tiny))
        return SpectralEstimate(beginning, energies, ratio)


class ModeClassifier:
    """Received preamble and spectral estimate -> ``fsk`` or ``c8psk``."""

    def process(self, samples, estimate):
        ratio = estimate.ratio
        if ratio > 6:
            return 'fsk'
        if .5 < ratio < 4:
            return 'c8psk'
        if 4 <= ratio <= 6:
            # Mark dominance resolves the gap in the two-line criterion.
            # This guard is a model choice supplementing the paper.
            preamble = samples[estimate.start:estimate.start + 128]
            windows = preamble.reshape(2, 64)
            bins = np.exp(-2j*np.pi*np.outer(np.arange(64), [1200, 1600, 4800])/SAMPLE_RATE)
            powers = np.sum(abs(windows @ bins)**2, axis=0)
            dominance = powers[0] / max(powers[1] + powers[2], np.finfo(float).tiny)
            if dominance > 4:
                return 'fsk'
        raise ValueError('ambiguous_modulation')


class Mixer:
    """Voltage samples -> complex samples mixed with a unit-gain NCO.

    Frequency is in hertz. The oscillator phase starts at zero for each
    recording; ``sign`` chooses the direction of frequency translation.
    """

    def __init__(self, frequency, sign=1, quantized=False):
        self.frequency = frequency
        self.sign = sign
        self.quantized = quantized

    def process(self, samples):
        return samples * tone(samples.size, self.frequency, sign=self.sign,
                              quantized=self.quantized)


class RRCFilter:
    """Complex samples -> centered root-raised-cosine filtered samples.

    ``dc`` gives unity DC gain for the FSK branch. ``symbols`` applies the
    C8PSK matched-filter scaling. Both use the same unit-energy RRC pulse.
    """

    def __init__(self, normalization='dc'):
        if normalization not in ('dc', 'symbols'):
            raise ValueError('RRC normalization must be dc or symbols')
        self.normalization = normalization
        self.pulse = rrc()

    def process(self, samples):
        if self.normalization == 'dc':
            return fir(samples, self.pulse / self.pulse.sum())
        return fir(samples, self.pulse) / np.sqrt(8)


class FirstDifference:
    """Complex samples -> first difference; the initial difference is zero."""

    def process(self, samples):
        return np.diff(samples, prepend=samples[0])


class L1Magnitude:
    """Complex samples -> the real nonnegative quantity abs(I) + abs(Q)."""

    def process(self, samples):
        return np.abs(samples.real) + np.abs(samples.imag)


class FskThreshold:
    """Difference magnitude and onset -> hard binary FSK decisions.

    The known FF preamble establishes ``level``, the threshold in the same
    units as the input magnitude. No payload bytes enter this estimate.
    """

    def __init__(self):
        self.level = None

    def process(self, values, onset):
        self.level = None
        bit_period = SAMPLE_RATE / 1200
        left = onset + int(bit_period)
        right = min(values.size, int(onset + 33*bit_period))
        reference = float(np.median(values[left:right])) if right > left else 0.
        if reference < 1e-8:
            raise ValueError('no_fsk_modulation')
        self.level = reference / 2
        return (values > self.level).astype(np.uint8)


class MovingAverage:
    """Sample array -> centered arithmetic mean with an odd-length window."""

    def __init__(self, length=5):
        if (not isinstance(length, (int, np.integer)) or isinstance(length, (bool, np.bool_))
                or length < 1 or length % 2 != 1):
            raise ValueError('Moving-average length must be a positive odd integer')
        self.length = length

    def process(self, samples):
        return fir(samples, np.full(self.length, 1/self.length))


class UARTSampler:
    """Smoothed binary levels and carrier gate -> 11-bit words and starts.

    Samples each 8O1 word at 1200 bit/s and retains invalid parity/stop bits
    for the frame decoder. Start positions use the input sample grid.
    """

    def process(self, logic, carrier):
        bit_period = SAMPLE_RATE / 1200
        rms, peak = carrier.rms, carrier.peak
        edges = np.flatnonzero((logic[:-1] > .5) & (logic[1:] <= .5)) + 1
        words, starts = [], []
        ready_at = 0.
        grid = np.arange(logic.size)
        for edge in edges:
            if edge < ready_at or rms[edge] < .35*peak:
                continue
            crossing = edge - 1 + (.5 - logic[edge-1])/(logic[edge] - logic[edge-1])
            centres = crossing + (np.arange(11) + .5)*bit_period
            if centres[-1] >= logic.size or np.interp(centres[-1], grid, rms) < .25*peak:
                continue
            word = (np.interp(centres, grid, logic) > .5).astype(np.uint8)
            if word[0] != 0:
                continue
            words.append(word)
            starts.append(crossing)
            ready_at = centres[-1]
        return (np.asarray(words, dtype=np.uint8).reshape(-1, 11),
                np.asarray(starts))


class PresetEqualizer:
    """Voltage samples and preamble ratio -> compensated voltage samples.

    The selected analog-pole preset is reported by ``pole_hz`` in hertz.
    It is None for a flat channel or disabled equalization. The preset bank
    and analog inverse response are explicit model choices.
    """

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.pole_hz = None

    def process(self, samples, ratio):
        self.pole_hz = None
        if self.enabled:
            poles = np.array([2500., 3000., 3700., 4000., 5000., 6000., 8000., 10000.])
            responses = np.r_[(poles**2 + 4800**2)/(poles**2 + 1600**2), 1.]
            selected = int(np.argmin(abs(responses - ratio)))
            if selected < poles.size:
                pole = float(poles[selected])
                samples = rc_filter(samples, pole, inverse=True)
                self.pole_hz = pole
        return samples
