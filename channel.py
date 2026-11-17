"""Pass a saved waveform through the RC and Gaussian-noise model."""

import argparse

from hart_receiver.medium import propagate
from hart_receiver.storage import finite_number, positive_number, prepare_outputs, load_signal, save_signal


def main():
    cli = argparse.ArgumentParser(description='Среда: однополюсный RC-канал и белый гауссов шум')
    cli.add_argument('input', help='NPZ с переданным сигналом')
    cli.add_argument('output', help='NPZ с сигналом на входе приёмника')
    cli.add_argument('--pole-hz', type=positive_number, help='частота полюса, Гц; по умолчанию 4000')
    cli.add_argument('--snr-db', type=finite_number, help='отношение сигнал/шум, дБ; по умолчанию 12')
    cli.add_argument('--seed', type=int, default=1, help='начальное состояние генератора шума; по умолчанию 1')
    cli.add_argument('--clean', action='store_true', help='передать отсчёты без фильтра и шума')
    cli.add_argument('--force', action='store_true', help='заменить существующий выходной файл')
    args = cli.parse_args()
    try:
        destination, = prepare_outputs([args.input], [args.output], force=args.force)
        if args.clean and (args.pole_hz is not None or args.snr_db is not None):
            raise ValueError('--clean несовместим с --pole-hz и --snr-db')
        if args.seed < 0:
            raise ValueError('--seed должен быть неотрицательным')
        pole = None if args.clean else (4000. if args.pole_hz is None else args.pole_hz)
        snr = None if args.clean else (12. if args.snr_db is None else args.snr_db)
        received = propagate(load_signal(args.input), pole_hz=pole, snr_db=snr, seed=args.seed)
        save_signal(destination, received)
        print('Без искажений' if args.clean else f'RC: {pole:g} Гц; SNR: {snr:g} дБ; seed: {args.seed}')
        print(destination)
    except (OSError, ValueError, OverflowError) as error:
        cli.exit(1, f'Ошибка: {error}\n')


if __name__ == '__main__':
    main()
