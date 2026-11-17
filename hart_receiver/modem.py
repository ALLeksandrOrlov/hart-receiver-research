"""One-record implementation of the dual-mode digital receiver from Jin et al.

The FSK path uses the difference magnitude, not a frequency discriminator.
C8PSK uses preamble acquisition followed by carrier and timing tracking.
Loop gains, the finite RRC length and acquisition thresholds are explicit
model choices; the article does not supply their implementation parameters.
"""

from dataclasses import dataclass, field
import numpy as np

from .dsp import SAMPLE_RATE, fir, quantize, rc_filter, rrc, sample, spectrum, tone
from .protocol import C8_POINTS, C8_START, C8_TRAINING, c8_decode, uart_decode


@dataclass
class Reception:
    mode: str = 'unknown'
    data: bytes = b''
    error: str | None = None
    symbols: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Acquisition:
    position: float
    amplitude: float
    phase: float
    phase_step: float


class Receiver:
    """Decode one burst using received samples only; trace is optional."""

    def __init__(self, *, equalize=True, quantized=False, adc_full_scale=None,
                 capture_trace=False):
        if adc_full_scale is not None:
            if not np.isfinite(adc_full_scale) or adc_full_scale <= 0:
                raise ValueError('ADC full scale must be positive and finite')
            if not quantized:
                raise ValueError('An ADC range requires quantized reception')
        self.equalize = equalize
        self.quantized = quantized
        self.adc_full_scale = adc_full_scale
        self.capture_trace = capture_trace
        self.trace = {}
        self._pulse = rrc()

    def _record(self, **values):
        if self.capture_trace:
            self.trace.update({name: np.asarray(value) for name, value in values.items()})

    def _input(self, waveform):
        samples = np.asarray(waveform)
        if (samples.ndim != 1 or samples.dtype.kind not in 'fiu'
                or not np.all(np.isfinite(samples))):
            raise ValueError('Expected finite real voltage samples')
        if samples.size < 256:
            raise ValueError('record_too_short')
        samples = samples.astype(float, copy=True)
        if self.quantized:
            # A single burst-wide AGC setting reserves 10% ADC headroom.
            # Return quantized voltages in the original physical scale.
            full_scale = self.adc_full_scale
            if full_scale is None:
                full_scale = max(float(np.max(np.abs(samples))) / .9, 1e-12)
            samples = quantize(samples, 8, full_scale=full_scale)
            self._record(adc_full_scale=full_scale)
        self._record(adc_output=samples)
        return samples

    @staticmethod
    def _carrier(samples):
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
        return int(positions[0]), rms, peak

    def _classify(self, samples, onset, diagnostics):
        beginning = onset + 64
        preamble = samples[beginning:beginning + 128]
        if preamble.size != 128:
            raise ValueError('preamble_too_short')
        energies = spectrum(preamble, quantized=self.quantized)
        total = energies.sum(axis=0)
        ratio = float(total[0] / max(total[1], np.finfo(float).tiny))
        self._record(spectral_start=beginning, spectral_energies=energies, spectral_ratio=ratio)
        diagnostics.update(spectral_ratio=ratio, onset_sample=onset)
        if ratio > 6:
            return 'fsk', ratio
        if .5 < ratio < 4:
            return 'c8psk', ratio
        if 4 <= ratio <= 6:
            # Resolve the otherwise undecided gap using mark dominance.
            # This preamble guard supplements the paper's two-line criterion.
            windows = preamble.reshape(2, 64)
            bins = np.exp(-2j*np.pi*np.outer(np.arange(64), [1200, 1600, 4800])/SAMPLE_RATE)
            powers = np.sum(abs(windows @ bins)**2, axis=0)
            dominance = powers[0] / max(powers[1] + powers[2], np.finfo(float).tiny)
            if dominance > 4:
                return 'fsk', ratio
        raise ValueError('ambiguous_modulation')

    def receive_frame(self, waveform):
        self.trace = {}
        result = Reception()
        try:
            samples = self._input(waveform)
            onset, rms, peak = self._carrier(samples)
            result.mode, ratio = self._classify(samples, onset, result.diagnostics)
            if result.mode == 'fsk':
                result.data = self._fsk(samples, onset, rms, peak)
            else:
                baseband = self._c8_baseband(samples, ratio, result.diagnostics)
                acquired = self._acquire(baseband, onset, result.diagnostics)
                result.symbols = self._track(baseband, acquired, result.diagnostics)
                result.data = c8_decode(result.symbols)
                result.symbols = result.symbols[:3*len(result.data) + 3]
        except ValueError as error:
            result.data = b''
            result.error = str(error)
        return result

    def _fsk(self, samples, onset, rms, peak):
        # Equations (3), (4), (7), (8): the complex mixer has unit gain.
        s2 = samples * tone(samples.size, 2200, quantized=self.quantized)
        s3 = fir(s2, self._pulse / self._pulse.sum())
        s4 = np.diff(s3, prepend=s3[0])
        s5 = np.abs(s4.real) + np.abs(s4.imag)
        # Estimate the mark level on the known FF preamble, not payload data.
        bit_period = SAMPLE_RATE / 1200
        left = onset + int(bit_period)
        right = min(samples.size, int(onset + 33*bit_period))
        reference = float(np.median(s5[left:right])) if right > left else 0.
        if reference < 1e-8:
            raise ValueError('no_fsk_modulation')
        threshold = reference / 2
        s6 = (s5 > threshold).astype(np.uint8)
        logic = fir(s6, np.full(5, .2))  # Fig. 5 averages AFTER thresholding.
        self._record(s2=s2, s3=s3, s4=s4, s5=s5, s6=s6,
                     fsk_output=logic, fsk_threshold=threshold)

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
        words = np.asarray(words, dtype=np.uint8).reshape(-1, 11)
        starts = np.asarray(starts)
        self._record(uart_words=words, uart_start_samples=starts)
        return uart_decode(words, starts)

    def _c8_baseband(self, samples, ratio, diagnostics):
        if self.equalize:
            # Preset analog-pole bank, selected only from received preamble R.
            poles = np.array([2500., 3000., 3700., 4000., 5000., 6000., 8000., 10000.])
            responses = np.r_[(poles**2 + 4800**2)/(poles**2 + 1600**2), 1.]
            selected = int(np.argmin(abs(responses - ratio)))
            if selected < poles.size:
                pole = float(poles[selected])
                samples = rc_filter(samples, pole, inverse=True)
                diagnostics['equalizer_pole_hz'] = pole
        mixed = samples * tone(samples.size, 3200, sign=-1, quantized=self.quantized)
        baseband = fir(mixed, self._pulse) / np.sqrt(8)
        self._record(c8_filtered=baseband)
        return baseband

    def _acquire(self, baseband, onset, diagnostics):
        known = C8_POINTS[np.concatenate((C8_TRAINING, C8_START))]
        reference = np.zeros(8*(known.size - 1) + 1, dtype=complex)
        reference[::8] = known
        # Bound search by the burst onset so payload cannot reacquire the modem.
        radius = self._pulse.size - 1 + 8
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
        diagnostics['training_score'] = float(normalized[best])
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
        diagnostics.update(first_symbol_sample=position, carrier_offset_hz=float(slope*3200/(2*np.pi)))
        return Acquisition(position, amplitude, float(phase), float(slope))

    def _track(self, baseband, acquired, diagnostics):
        # Fig. 1: I^2 + Q^2 -> clock BPF -> averaging -> interpolator.
        # Translate the 3200 Hz clock line to DC and average 32 symbol periods.
        # This is the complex equivalent of a narrow BPF at the symbol rate;
        # the integer-period window rejects the envelope's DC component.
        # Window length and loop gains are explicit model choices.
        clock_carrier = tone(baseband.size, 3200, sign=-1)
        clock = np.convolve(abs(baseband)**2 * clock_carrier,
                            np.full(256, 1/256), 'same')
        position = acquired.position + 44*8
        period = 8.
        phase_step = acquired.phase_step
        phase = acquired.phase + 44*phase_step
        labels, constellation, timing = [], [], []
        while position + 2*period < baseband.size - 1:
            upcoming = sample(baseband, position + period*np.arange(3))
            if np.all(abs(upcoming) < .4*acquired.amplitude):
                break
            current = upcoming[0]
            rotated = current*np.exp(-1j*phase)/acquired.amplitude
            decision = int(np.argmin(abs(rotated - C8_POINTS)))
            labels.append(decision)
            constellation.append(rotated)
            phase_error = float(np.angle(rotated * C8_POINTS[decision].conjugate()))
            phase_step += .0005*phase_error
            phase += phase_step + .08*phase_error
            # The line phase gives the distance to the next symbol centre.
            # Wrap it to one period; retain the acquired clock if the line fades.
            reference = sample(clock, position)
            error = -np.angle(reference*np.exp(2j*np.pi*position/8))*8/(2*np.pi)
            error = float(np.clip(error, -2., 2.))
            if abs(reference) < 1e-4*acquired.amplitude**2:
                error = 0.
            period = float(np.clip(period + .0004*error, 7.96, 8.04))
            timing.append((position, error, period))
            position += period + .12*error
        symbols = np.asarray(labels, dtype=int)
        self._record(c8_constellation=np.asarray(constellation), c8_symbols=symbols,
                     c8_clock=clock, c8_timing=np.asarray(timing))
        diagnostics['recovered_samples_per_symbol'] = period
        return symbols
