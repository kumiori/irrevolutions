import argparse

# Define a set of admissible parameters
admissible_parameters = {"param1", "param2", "param3"}


def main():
    parser = argparse.ArgumentParser(description="Script that accepts a parameter.")

    # Define the parameter argument
    parser.add_argument(
        "parameter", choices=admissible_parameters, help="The parameter to use."
    )

    args = parser.parse_args()
    parameter = args.parameter

    print(f"The parameter '{parameter}' is valid.")


if __name__ == "__main__":
    main()
