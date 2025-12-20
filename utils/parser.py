import argparse

def get_main_parser():
    parser = argparse.ArgumentParser(
        description="Raman Amplifier Control CLI"
    )

    parser.add_argument(
        "-s", "--save-plots",
        action="store_true",
        help="If set, saves the results to a file"
    )

    parser.add_argument(
        "-l", "--live-plot",
        action="store_true",
        help="If set, plots figures live"
    )

    parser.add_argument(
        "-c", "--customize",
        action="store_true",
        help="If set, you will be prompted to customize simulation elements"
    )

    parser.add_argument(
        "-i", "--iterations",
        type=int,
        action="store",
        help="Sets the number of iterations to simulate"
    )

    parser.add_argument(
        "--bounds",
        type=int,
        nargs=2,
        metavar=("LOWER", "UPPER"),
        help="Lower and upper bounds of the experiment (integers, [nm])"
    )

    return parser

def data_generator_parser():
    parser = argparse.ArgumentParser(
        description="Raman Amplifier Control CLI"
    )

    parser.add_argument(
        "-n", "--num_samples",
        type=int,
        action="store",
        help="Sets number of samples to create",
        default=1000,
    )

    parser.add_argument(
        "-p", "--pumps",
        type=int,
        action="store",
        help="Sets number of pumps",
        default=3,
    )

    parser.add_argument(
        "-r", "--ratio",
        type=float,
        action="store",
        help="Sets pumping ratio (must be in [0.0, 1.0])",
        default=0.0
    )

    parser.add_argument(
        "-l", "--fiber_length",
        type=float,
        action="store",
        help="Sets fiber length (in km)",
        default=100.0
    )

    return parser
