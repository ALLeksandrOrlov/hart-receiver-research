"""Connect the signal-processing blocks for one complete HART recording.

Shared: ADC -> carrier detection -> preamble spectrum -> mode selection.
FSK: mixer -> RRC -> difference -> L1 magnitude -> threshold -> average
     -> UART sampling -> frame decoder.
C8PSK: preset equalizer -> mixer -> RRC -> preamble acquisition
       -> interpolator + carrier/timing feedback -> symbol decisions -> decoder.

The blocks operate on complete recordings. Each feedback loop is initialized
for one burst; file I/O and command-line options belong to the three scripts.
"""

from dataclasses import dataclass, field
import numpy as np

from .blocks import (
    InputADC, CarrierDetector, PreambleSpectrum, ModeClassifier,
    Mixer, RRCFilter, FirstDifference, L1Magnitude, FskThreshold,
    MovingAverage, UARTSampler, PresetEqualizer,
)
from .synchronization import (
    PreambleAcquisition, SymbolInterpolator, EnvelopePower, ClockBandpass,
    CarrierLoop, SymbolSlicer, PhaseErrorDetector, TimingErrorDetector,
    TimingLoop, BurstEndDetector,
)
from .protocol import c8_decode, uart_decode


@dataclass
class Reception:
    mode: str = 'unknown'
    data: bytes = b''
    error: str | None = None
    symbols: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    diagnostics: dict = field(default_factory=dict)


class Receiver:
    """Route voltages through connected blocks and return one validated frame."""

    def __init__(self, *, equalize=True, quantized=False, adc_full_scale=None,
                 capture_trace=False):
        self.capture_trace = capture_trace
        self.trace = {}
        self.adc = InputADC(quantized=quantized, full_scale=adc_full_scale)
        self.carrier_detector = CarrierDetector()
        self.preamble_spectrum = PreambleSpectrum(quantized=quantized)
        self.mode_classifier = ModeClassifier()

        self.fsk_mixer = Mixer(2200, quantized=quantized)
        self.fsk_rrc = RRCFilter(normalization='dc')
        self.difference = FirstDifference()
        self.magnitude = L1Magnitude()
        self.threshold = FskThreshold()
        self.average = MovingAverage(length=5)
        self.uart = UARTSampler()

        self.equalizer = PresetEqualizer(enabled=equalize)
        self.c8_mixer = Mixer(3200, sign=-1, quantized=quantized)
        self.c8_rrc = RRCFilter(normalization='symbols')
        self.acquisition = PreambleAcquisition(pulse_size=self.c8_rrc.pulse.size)
        self.envelope_power = EnvelopePower()
        self.clock_filter = ClockBandpass()
        self.symbol_slicer = SymbolSlicer()
        self.phase_detector = PhaseErrorDetector()
        self.timing_detector = TimingErrorDetector()
        self.burst_end = BurstEndDetector()

    def _record(self, **values):
        if self.capture_trace:
            self.trace.update({name: np.asarray(value) for name, value in values.items()})

    def receive_frame(self, waveform):
        self.trace = {}
        result = Reception()
        try:
            samples = self.adc.process(waveform)
            if self.adc.used_full_scale is not None:
                self._record(adc_full_scale=self.adc.used_full_scale)
            self._record(adc_output=samples)
            carrier = self.carrier_detector.process(samples)
            spectral = self.preamble_spectrum.process(samples, carrier.onset)
            self._record(spectral_start=spectral.start, spectral_energies=spectral.energies,
                         spectral_ratio=spectral.ratio)
            result.diagnostics.update(spectral_ratio=spectral.ratio, onset_sample=carrier.onset)
            result.mode = self.mode_classifier.process(samples, spectral)
            if result.mode == 'fsk':
                result.data = self._receive_fsk(samples, carrier)
            else:
                result.symbols = self._receive_c8(samples, carrier.onset, spectral.ratio,
                                                  result.diagnostics)
                result.data = c8_decode(result.symbols)
                result.symbols = result.symbols[:3*len(result.data) + 3]
        except ValueError as error:
            result.data = b''
            result.error = str(error)
        return result

    def _receive_fsk(self, samples, carrier):
        # Jin 2010, Fig. 5: 1200 Hz (bit 1) becomes a 1000 Hz component;
        # 2200 Hz (bit 0) becomes DC and vanishes after differencing.
        # The bit labels preceding Eqs. (8) and (9) are swapped in the paper.
        s2 = self.fsk_mixer.process(samples)
        s3 = self.fsk_rrc.process(s2)
        s4 = self.difference.process(s3)
        s5 = self.magnitude.process(s4)
        s6 = self.threshold.process(s5, carrier.onset)
        logic = self.average.process(s6)  # Fig. 5 averages after thresholding.
        self._record(s2=s2, s3=s3, s4=s4, s5=s5, s6=s6,
                     fsk_output=logic, fsk_threshold=self.threshold.level)
        words, starts = self.uart.process(logic, carrier)
        self._record(uart_words=words, uart_start_samples=starts)
        return uart_decode(words, starts)

    def _receive_c8(self, samples, onset, ratio, diagnostics):
        equalized = self.equalizer.process(samples, ratio)
        if self.equalizer.pole_hz is not None:
            diagnostics['equalizer_pole_hz'] = self.equalizer.pole_hz
        mixed = self.c8_mixer.process(equalized)
        baseband = self.c8_rrc.process(mixed)
        self._record(c8_filtered=baseband)
        try:
            acquired = self.acquisition.process(baseband, onset)
        finally:
            diagnostics.update(self.acquisition.diagnostics)
        return self._track_symbols(baseband, acquired, diagnostics)

    def _track_symbols(self, baseband, acquired, diagnostics):
        # The envelope branch supplies a clock reference to the timing loop.
        clock = self.clock_filter.process(self.envelope_power.process(baseband))
        signal_samples = SymbolInterpolator(baseband)
        clock_samples = SymbolInterpolator(clock)
        carrier_loop = CarrierLoop(acquired)
        timing_loop = TimingLoop(acquired)
        labels, constellation, timing = [], [], []
        while timing_loop.position + 2*timing_loop.period < baseband.size - 1:
            positions = timing_loop.position + timing_loop.period*np.arange(3)
            upcoming = signal_samples.process(positions)
            if self.burst_end.process(upcoming, acquired.amplitude):
                break
            rotated = carrier_loop.correct(upcoming[0])
            decision = self.symbol_slicer.process(rotated)
            labels.append(decision)
            constellation.append(rotated)

            # Decisions feed the carrier loop; the clock branch feeds timing.
            phase_error = self.phase_detector.process(rotated, decision)
            carrier_loop.advance(phase_error)
            reference = clock_samples.process(timing_loop.position)
            error = self.timing_detector.process(reference, timing_loop.position,
                                                acquired.amplitude)
            timing.append(timing_loop.advance(error))

        symbols = np.asarray(labels, dtype=int)
        self._record(c8_constellation=np.asarray(constellation), c8_symbols=symbols,
                     c8_clock=clock, c8_timing=np.asarray(timing))
        diagnostics['recovered_samples_per_symbol'] = timing_loop.period
        return symbols
