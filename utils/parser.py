import argparse

def get_parser():
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
        "--bounds",
        type=int,
        nargs=2,
        metavar=("LOWER", "UPPER"),
        help="Lower and upper bounds of the experiment (integers, [nm])"
    )

    return parser
