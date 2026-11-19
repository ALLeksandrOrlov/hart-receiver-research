"""Text frames, voltage recordings, and protected output files."""

import argparse
import math
import os
from pathlib import Path
import re
import tempfile
from zipfile import BadZipFile
from zlib import error as CompressionError

import numpy as np

from .dsp import SAMPLE_RATE


def finite_number(text):
    """argparse converter shared by the three command-line programs."""
    try:
        number = float(text)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError('ожидается число') from error
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError('число должно быть конечным')
    return number


def positive_number(text):
    number = finite_number(text)
    if number <= 0:
        raise argparse.ArgumentTypeError('число должно быть больше нуля')
    return number


def read_hex(path):
    """Read one frame; each token is a byte and # starts a line comment."""
    tokens = []
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        tokens.extend(line.partition('#')[0].split())
    if not tokens or any(not re.fullmatch('[0-9A-Fa-f]{2}', token) for token in tokens):
        raise ValueError('ожидается кадр из HEX-байтов через пробел, например 02 80 00 00 82')
    return bytes.fromhex(' '.join(tokens))


def _same_file(left, right):
    if left == right:
        return True
    return left.exists() and right.exists() and os.path.samefile(left, right)


def prepare_outputs(inputs, outputs, *, force=False):
    """Validate every path before invalidating outputs from an earlier run.

    Resolved paths catch symbolic links; samefile additionally catches hard
    links. Inputs include comparison references as well as the signal source.
    """
    sources = tuple(Path(name).resolve() for name in inputs)
    destinations = tuple(Path(name).resolve() for name in outputs)
    for index, destination in enumerate(destinations):
        if any(_same_file(destination, source) for source in sources):
            raise ValueError('выходной файл совпадает с входом или эталоном')
        if any(_same_file(destination, other) for other in destinations[:index]):
            raise ValueError('выходным файлам нужны разные имена')
        if destination.exists():
            if not destination.is_file():
                raise ValueError(f'путь выхода занят не файлом: {destination}')
            if not force:
                raise ValueError(f'файл уже существует: {destination}; добавьте --force для замены')
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
    for destination in destinations:
        if destination.exists():
            destination.unlink()
    return destinations


def _atomic_write(path, write):
    destination = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', dir=destination.parent,
                                         prefix=f'.{destination.name}.', delete=False) as stream:
            temporary = Path(stream.name)
            write(stream)
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_hex(path, data):
    payload = bytes(data)
    if not payload:
        raise ValueError('нельзя записать пустой кадр')
    encoded = (payload.hex(' ').upper() + '\n').encode('utf-8')
    _atomic_write(path, lambda stream: stream.write(encoded))


def _voltage_samples(samples):
    values = np.asarray(samples)
    if (values.ndim != 1 or values.size < 256 or values.dtype.kind not in 'fiu'
            or not np.isfinite(values).all()):
        raise ValueError('нужны минимум 256 конечных вещественных отсчётов в одномерном массиве')
    values = values.astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError('отсчёты не помещаются в формат float64')
    return values


def load_signal(path):
    """NPZ deliberately carries only a real waveform and its sample rate."""
    try:
        archive = np.load(path, allow_pickle=False)
        if not isinstance(archive, np.lib.npyio.NpzFile):
            raise ValueError('ожидается NPZ, содержащий samples и sample_rate')
        with archive:
            if len(archive.files) != 2 or set(archive.files) != {'samples', 'sample_rate'}:
                raise ValueError('в NPZ должны быть только samples и sample_rate')
            rate = archive['sample_rate']
            if (not isinstance(rate, np.ndarray) or rate.shape != ()
                    or rate.dtype.kind not in 'fiu' or float(rate) != SAMPLE_RATE):
                raise ValueError(f'частота дискретизации должна быть {SAMPLE_RATE} Гц')
            return _voltage_samples(archive['samples'])
    except (EOFError, BadZipFile, CompressionError, NotImplementedError, RuntimeError) as error:
        raise ValueError('файл сигнала пуст или повреждён') from error


def save_signal(path, samples):
    values = _voltage_samples(samples)
    _atomic_write(path, lambda stream: np.savez_compressed(
        stream, samples=values, sample_rate=np.int64(SAMPLE_RATE)))


def save_trace(path, trace):
    arrays = {'sample_rate': np.asarray(SAMPLE_RATE, dtype=np.int64)}
    for name, value in trace.items():
        if name == 'sample_rate' or not isinstance(name, str):
            raise ValueError('некорректное имя промежуточного сигнала')
        array = np.asarray(value)
        if array.dtype.kind not in 'biufc' or not np.isfinite(array).all():
            raise ValueError(f'некорректные значения промежуточного сигнала {name}')
        arrays[name] = array
    _atomic_write(path, lambda stream: np.savez_compressed(stream, **arrays))
