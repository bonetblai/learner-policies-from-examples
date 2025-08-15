import logging
import argparse
import shutil, sys
from pathlib import Path
from typing import List, Tuple

if __name__ == "__main__":
    # Setup logger
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    #handler.terminator = ""
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # Arguments
    parser = argparse.ArgumentParser(description="Get sketch.")
    parser.add_argument("--name", type=str, required=True, help="Domain name.")
    parser.add_argument("--logfile", type=Path, required=True, help="Path to logfile.")
    parser.add_argument("--sketches", type=Path, required=True, help="Path to sketches folder.")
    parser.add_argument("--workspace", type=Path, required=True, help="Path to workspace folder.")
    args = parser.parse_args()

    if not args.logfile.exists():
        logging.error(f"Error: '{args.logfile}' doesn't exist!")
        exit(-1)

    with args.logfile.open("r") as fd:
        lines: List[str] = [line.strip("\n") for line in fd.readlines()]

    uuid: List[str] = [line for line in lines if "UUID:" in line]
    assert len(uuid) == 1
    uuid: str = uuid[0].split(" ")[-1]
    logging.info(f"UUID: {uuid}")

    output_folder: Path = args.workspace / f"output.{uuid}"
    if not output_folder.exists():
        logging.error(f"Error: '{output_folder}' doesn't exist!")
        exit(-1)

    sketch_filename: Path = output_folder / "sketch_minimized_0.txt"
    if not sketch_filename.exists():
        logging.error(f"Error: '{sketch_filename}' doesn't exist!")
        exit(-1)

    new_sketch_filename: Path = args.sketches / f"{args.name}_sketch_minimized_0.txt"
    shutil.copy(sketch_filename, new_sketch_filename)
    logging.info(f"{sketch_filename.name} COPIED-TO {new_sketch_filename}")

