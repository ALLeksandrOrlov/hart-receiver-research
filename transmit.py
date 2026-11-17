"""Create a HART waveform from one hexadecimal frame."""

import argparse

from hart_receiver.dsp import SAMPLE_RATE
from hart_receiver.storage import finite_number, positive_number, prepare_outputs, read_hex, save_signal
from hart_receiver.waveform import transmit


def main():
    cli = argparse.ArgumentParser(description='Передатчик: HEX-кадр → файл сигнала')
    cli.add_argument('input', help='текстовый файл с байтами кадра')
    cli.add_argument('output', help='выходной NPZ с отсчётами напряжения')
    cli.add_argument('--mode', required=True, choices=['fsk', 'c8psk'])
    cli.add_argument('--phase', type=finite_number, default=0., help='начальная фаза, рад; по умолчанию 0')
    cli.add_argument('--fsk-mvpp', type=positive_number, help='размах FSK, мВ; по умолчанию 200')
    cli.add_argument('--c8-dbm', type=finite_number, help='мощность C8PSK на 500 Ом, дБм; по умолчанию -6')
    cli.add_argument('--force', action='store_true', help='заменить существующий выходной файл')
    args = cli.parse_args()
    try:
        destination, = prepare_outputs([args.input], [args.output], force=args.force)
        if args.mode == 'fsk' and args.c8_dbm is not None:
            raise ValueError('--c8-dbm используется только с --mode c8psk')
        if args.mode == 'c8psk' and args.fsk_mvpp is not None:
            raise ValueError('--fsk-mvpp используется только с --mode fsk')
        frame = read_hex(args.input)
        waveform = transmit(frame, args.mode, phase=args.phase,
                            fsk_mvpp=200. if args.fsk_mvpp is None else args.fsk_mvpp,
                            c8_dbm=-6. if args.c8_dbm is None else args.c8_dbm)
        save_signal(destination, waveform)
        print(f'{args.mode}: {len(frame)} байт → {len(waveform)} отсчётов ({SAMPLE_RATE} Гц)')
        print(destination)
    except (OSError, ValueError, OverflowError) as error:
        cli.exit(1, f'Ошибка: {error}\n')


if __name__ == '__main__':
    main()
