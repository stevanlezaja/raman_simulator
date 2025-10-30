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
