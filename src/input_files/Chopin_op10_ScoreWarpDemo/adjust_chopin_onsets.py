import argparse
import json
from pathlib import Path


OFFSET_SECONDS = 2.5
DEFAULT_INPUT = Path(__file__).with_name("Chopin_op10_no3_p11-mei.maps.json")
DEFAULT_OUTPUT = Path(__file__).with_name("Chopin_op10_no3_p11-mei.maps.shifted.json")


def shift_onsets(input_path: Path, output_path: Path) -> None:
    with input_path.open(encoding="utf-8") as input_file:
        records = json.load(input_file)

    if not isinstance(records, list):
        raise ValueError("The MAPS JSON must contain a list of records.")

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} must be an object.")
        onset = record.get("obs_mean_onset")
        if not isinstance(onset, (int, float)) or isinstance(onset, bool):
            raise ValueError(f"Record {index} has no numeric obs_mean_onset.")
        record["obs_mean_onset"] = onset - OFFSET_SECONDS

    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(records, output_file, indent=4)
        output_file.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Subtract 2.5 seconds from every MAPS obs_mean_onset value."
    )
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("output", nargs="?", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()

    shift_onsets(arguments.input, arguments.output)


if __name__ == "__main__":
    main()