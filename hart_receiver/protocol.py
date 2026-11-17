"""HART frame layout and the FSK/C8PSK byte-to-symbol encodings."""

import numpy as np


C8_POINTS = np.exp(1j * np.pi / 180 * np.array(
    [-112.5, -157.5, -67.5, -22.5, 112.5, 157.5, 67.5, 22.5]))
C8_TRAINING = np.tile(np.array([6, 2], dtype=np.uint8), 20)
C8_START = np.array([4, 1, 2, 1], dtype=np.uint8)


def frame_length(prefix):
    """Return the complete PDU length once its byte-count field is available."""
    if len(prefix) == 0:
        return None
    delimiter = int(prefix[0])
    if (delimiter & 7) not in (1, 2, 6):
        raise ValueError('invalid_delimiter')
    address_size = 5 if delimiter & 0x80 else 1
    expansion_size = (delimiter >> 5) & 3
    count_at = address_size + expansion_size + 2
    if len(prefix) <= count_at:
        return None
    return count_at + 2 + int(prefix[count_at])


def validate_frame(pdu, mode=None):
    """Check delimiter, declared length, XOR and, optionally, physical mode."""
    frame = bytes(pdu)
    expected = frame_length(frame)
    if expected is None or len(frame) < expected:
        raise ValueError('truncated_pdu')
    if len(frame) != expected:
        raise ValueError('extra_pdu_bytes')
    checksum = 0
    for byte in frame:
        checksum ^= byte
    if checksum != 0:
        raise ValueError('checksum_error')
    if mode is not None:
        layers = {'fsk': 0, 'c8psk': 1}
        if mode not in layers:
            raise ValueError('Unknown modulation mode')
        if ((frame[0] >> 3) & 3) != layers[mode]:
            raise ValueError('physical_layer_mismatch')
    return frame


def _characters(data):
    """Rows contain eight least-significant-bit-first bits and odd parity."""
    octets = np.frombuffer(bytes(data), dtype=np.uint8)
    bits = (octets[:, None] >> np.arange(8)) & 1
    parity = 1 - bits.sum(axis=1) % 2
    return np.column_stack((bits, parity)).astype(np.uint8)


def uart_encode(data):
    characters = _characters(data)
    words = np.ones((len(characters), 11), dtype=np.uint8)
    words[:, 0] = 0
    words[:, 1:10] = characters
    return words.ravel()


def uart_decode(words, starts):
    """Find a frame after consecutive FF words; never discard corrupt words.

    Start times use the 25600 Hz input grid. The gap guard includes a 1%
    transmitter-clock allowance and half a bit of edge-estimation uncertainty.
    """
    words = np.asarray(words)
    starts = np.asarray(starts, dtype=float)
    if words.ndim != 2 or words.shape[1] != 11 or not np.isin(words, [0, 1]).all():
        raise ValueError('invalid_uart_words')
    if (starts.shape != (len(words),) or not np.isfinite(starts).all()
            or np.any(np.diff(starts) <= 0)):
        raise ValueError('invalid_uart_starts')
    words = words.astype(np.uint8)
    valid = ((words[:, 0] == 0) & (words[:, 10] == 1)
             & ((words[:, 1:10].sum(axis=1) & 1) == 1))
    octets = words[:, 1:9].astype(np.uint16) @ (1 << np.arange(8))
    spacing_limit = (22 / .99 + .5) * (25600 / 1200)
    separated = (np.r_[False, np.diff(starts) > spacing_limit]
                 if len(starts) else np.empty(0, bool))
    preamble_words = 0
    frame = None
    for index, octet in enumerate(octets):
        if frame is None:
            if separated[index]:
                preamble_words = 0
            if valid[index] and octet == 0xff:
                preamble_words += 1
                continue
            if preamble_words < 2:
                preamble_words = 0
                continue
            frame = bytearray()
        elif separated[index]:
            raise ValueError('intercharacter_gap')
        if not valid[index]:
            raise ValueError('uart_error')
        frame.append(int(octet))
        size = frame_length(frame)
        if size is not None and len(frame) == size:
            return validate_frame(frame, 'fsk')
    raise ValueError('truncated_pdu' if frame is not None else 'preamble_too_short')


def _scramble(bits, *, decode=False):
    """V.27ter x^-6/x^-7 feedback and the repeated-pattern escape rule.

    Bit zero of the register holds the most recent transmitted bit. Both
    directions update that register with line bits, so the same state machine
    can encode and decode. A fresh frame starts with a zero register/counter.
    """
    source = np.asarray(bits)
    if source.ndim != 1 or not np.isin(source, [0, 1]).all():
        raise ValueError('Expected binary data')
    result = np.empty(len(source), dtype=np.uint8)
    register = repetitions = 0
    for index, bit in enumerate(source):
        escape = repetitions == 33
        decoded = int(bit) ^ ((register >> 5) & 1) ^ ((register >> 6) & 1) ^ escape
        line_bit = int(bit) if decode else decoded
        repeated = any(line_bit == ((register >> delay) & 1) for delay in (7, 8, 11))
        repetitions = repetitions + 1 if repeated and not escape else 0
        register = ((register << 1) | line_bit) & 0xfff
        result[index] = decoded
    return result


def c8_encode(pdu):
    frame = validate_frame(pdu, 'c8psk')
    bits = _scramble(_characters(frame).ravel())
    tribits = bits.reshape(-1, 3) @ np.array([4, 2, 1], dtype=np.uint8)
    return np.concatenate((tribits, np.zeros(3, dtype=np.uint8)))


def c8_decode(symbols):
    """Require the raw stop flag after the PDU, even if the carrier has ended."""
    symbols = np.asarray(symbols)
    if (symbols.ndim != 1 or symbols.dtype.kind not in 'fiu'
            or not np.isfinite(symbols).all() or not np.isin(symbols, np.arange(8)).all()):
        raise ValueError('invalid_c8_symbols')
    labels = symbols.astype(np.uint8)
    bits = ((labels[:, None] >> np.array([2, 1, 0])) & 1).ravel()
    plain = _scramble(bits, decode=True)
    complete_bits = len(plain) // 9 * 9
    characters = plain[:complete_bits].reshape(-1, 9)
    octets = (characters[:, :8] @ (1 << np.arange(8))).astype(np.uint8)
    size = frame_length(octets)
    if size is None or len(octets) < size:
        raise ValueError('truncated_pdu')
    if np.any(characters[:size].sum(axis=1) % 2 != 1):
        raise ValueError('parity_error')
    frame = validate_frame(octets[:size].tobytes(), 'c8psk')
    stop = labels[3*size:3*size+3]
    if len(stop) != 3 or np.any(stop):
        raise ValueError('missing_stop_flag')
    return frame
