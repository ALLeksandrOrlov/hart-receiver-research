"""Recover a frame from voltage samples; optional comparison follows reception."""

import argparse

from hart_receiver.modem import Receiver
from hart_receiver.storage import load_signal, positive_number, prepare_outputs, read_hex, save_trace, write_hex


def main():
    cli = argparse.ArgumentParser(description='Приёмник: файл сигнала → HEX-кадр')
    cli.add_argument('input', help='NPZ с отсчётами напряжения')
    cli.add_argument('output', help='текстовый файл с принятыми байтами')
    cli.add_argument('--compare', help='проверить результат по исходному HEX-файлу')
    cli.add_argument('--no-equalizer', action='store_true', help='отключить компенсацию канала C8PSK')
    cli.add_argument('--quantized', action='store_true', help='АЦП 8 бит и таблицы NCO/DFT 10 бит')
    cli.add_argument('--adc-full-scale', type=positive_number, help='положительная граница диапазона АЦП, В')
    cli.add_argument('--trace', help='отдельный NPZ с промежуточными сигналами')
    cli.add_argument('--force', action='store_true', help='заменить существующие выходные файлы')
    args = cli.parse_args()
    try:
        sources = [args.input] + ([args.compare] if args.compare else [])
        targets = [args.output] + ([args.trace] if args.trace else [])
        destinations = prepare_outputs(sources, targets, force=args.force)
        if args.adc_full_scale is not None and not args.quantized:
            raise ValueError('--adc-full-scale требует --quantized')
        modem = Receiver(equalize=not args.no_equalizer, quantized=args.quantized,
                         adc_full_scale=args.adc_full_scale, capture_trace=bool(args.trace))
        reception = modem.receive_frame(load_signal(args.input))
        if args.trace:
            save_trace(destinations[1], modem.trace)
        if reception.error:
            raise ValueError(f'кадр не принят: {reception.error}')
        write_hex(destinations[0], reception.data)
        print(f'{reception.mode}: принято {len(reception.data)} байт')
        print(destinations[0])
        if args.trace and 'spectral_ratio' in modem.trace:
            print(f"E1600/E4800 = {float(modem.trace['spectral_ratio']):.6f}")
        # Only the decoded result is compared. These bytes never enter the modem.
        if args.compare:
            if read_hex(args.compare) != reception.data:
                raise ValueError('принятые байты не совпадают с исходным файлом')
            print('Байты совпадают с исходными')
    except (OSError, ValueError, OverflowError) as error:
        cli.exit(1, f'Ошибка: {error}\n')


if __name__ == '__main__':
    main()
