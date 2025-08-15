import logging
import shutil
import sys
import argparse

from pathlib import Path
from math import ceil, log10
from typing import Dict, List, Tuple, Optional, Union

import subprocess


def _singularity_wrapper(domain_filepath: Path, instance_filepath: Path, sketch_filepath: Path, plan_filepath: Path) -> Tuple[int, List[str]]:
    args: List[str] = ["./experiments/run-singularity-siwr.sh", "planners/siwr.sif", str(domain_filepath), str(instance_filepath), str(sketch_filepath), "0", str(plan_filepath)]
    stdout: List[str] = []
    with subprocess.Popen(args=args, stdout=subprocess.PIPE, text=True) as proc:
        for line in proc.stdout:
            stdout.append(line.strip("\n"))
    rc = proc.returncode
    return rc, stdout

def is_solvable(planner: Optional[str], domain_filepath: Path, instance_filepath) -> bool:
    if planner is None:
        return True
    else:
        plan_filepath: Path = Path(f"{instance_filepath.name}_{domain_filepath.name}_{planner}.txt")
        logging.debug(f"plan_filepath: {plan_filepath}")
        args: List[str] = ["python", "call_planner.py", planner, str(domain_filepath), str(instance_filepath), str(plan_filepath)]
        logging.debug(f"args: {args}")
        p = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        p.wait()
        with plan_filepath.open("r") as fd:
            plan: List[str] = [line.strip("\n") for line in fd.readlines()]
        logging.debug(f"Plan: {plan}")
        is_unsolvable: bool = "NO-PLAN" in plan
        plan_filepath.unlink()
        logging.debug(f"is_unsolvable: {is_unsolvable}")
        logging.info(f"  {instance_filepath.name}: {not is_unsolvable}")
        return not is_unsolvable


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
    parser = argparse.ArgumentParser(description="Test sketch.")
    parser.add_argument("--domain", type=Path, required=True, help="Path to domain folder.")
    parser.add_argument("--sketches", type=Path, required=True, help="Path to sketches folder.")
    parser.add_argument("--output", type=Path, required=True, help="Path to output folder.")
    parser.add_argument("--planner", type=str, default="siw+bfws", choices=["bfws", "siw", "siw_plus", "siw+bfws"], help="Set planner. Default is 'siw+bfws'.")
    parser.add_argument("--analyze", action="store_true", default=False, help="Just analyze results in output folder. Default is False.")
    parser.add_argument("--no_filtering", action="store_true", default=False, help="Don't filter instances with planner. Default is False.")
    args = parser.parse_args()

    for filepath in [args.domain, args.sketches]:
        if not filepath.exists():
            logging.error(f"Error: '{filepath}' doesn't exist!")
            exit(-1)

    name: str = args.domain.name
    domain_filepath: Path = args.domain / "domain.pddl"
    benchmark_filepath: Path = args.domain
    sketch_filepath: Path = args.sketches / f"{name}_sketch_minimized_0.txt"
    instance_filepaths: List[Path] = [instance_filepath for instance_filepath in args.domain.glob("*.pddl") if instance_filepath.name != domain_filepath.name]
    num_instances = len(instance_filepaths)

    for filepath in [domain_filepath, sketch_filepath]:
        if not filepath.exists():
            logger.warning(f"Exiting because {filepath}' doesn't exist!")
            exit(0)

    logging.info(f"BENCHMARK folder '{benchmark_filepath}' found")
    logging.info(f"SKETCH '{sketch_filepath}' found")

    output_filepath: Path = args.output / name
    output_filepath.mkdir(parents=True, exist_ok=True)

    # Analyze existing results, if requested
    if args.analyze:
        dot_output_files: List[Path] = sorted([filepath for filepath in output_filepath.glob("*.output")], key=lambda p: p.name)
        dot_plan_files: List[Path] = sorted([filepath for filepath in output_filepath.glob("*.plan")], key=lambda p: p.name)
        solved_by_sketch: List[str] = []
        for dot_output_file, dot_plan_file in zip(dot_output_files, dot_plan_files):
            with dot_output_file.open("r") as fd:
                lines: List[str] = [line.strip("\n") for line in fd.readlines()]
                valid_plan: bool = any([line.startswith("Plan valid") for line in lines[-20:]])
                if valid_plan: solved_by_sketch.append(dot_output_file)

        # Statistics
        num_solved_by_planner = len(dot_output_files)
        num_solved_by_sketch = len(solved_by_sketch)
        logging.info(f"{num_solved_by_planner} instance(s) solvable by {args.planner}")
        logging.info(f"{num_solved_by_sketch} instance(s) solved by sketch: {100 * num_solved_by_sketch / num_solved_by_planner:0.2f} {100 * num_solved_by_sketch / num_instances:0.2f}")
        exit(0)


    # Discard instances that are not solvable by planner
    if not args.no_filtering:
        logging.info(f"Checking solvability modulo {args.planner}...")
        solved_by_planner: List[Path] = [instance_filepath for instance_filepath in instance_filepaths if is_solvable(args.planner, domain_filepath, instance_filepath)]
    else:
        logging.info(f"SKIPPING check of solvability modulo {args.planner}...")
        solved_by_planner: List[Path] = [instance_filepath for instance_filepath in instance_filepaths]
    num_solved_by_planner = len(solved_by_planner)
    logging.info(f"{num_solved_by_planner} instance(s) solvable by '{args.planner}'")

    # Run sketch in solved_by_planner instances
    solved_by_sketch: Dict[str, int] = dict()
    ndigits: int = int(ceil(log10(num_solved_by_planner + 1/2)))
    for i, instance_filepath in enumerate(solved_by_planner):
        logging.info(f"Testing sketch on instance {i+1:{ndigits}}/{num_solved_by_planner} [{instance_filepath}]...")

        plan_filepath: Path = output_filepath / f"{instance_filepath.name}.plan"
        stdout_filepath: Path = output_filepath / f"{instance_filepath.name}.output"
        plan_filepath.unlink(missing_ok=True)
        stdout_filepath.unlink(missing_ok=True)

        rc, stdout = _singularity_wrapper(domain_filepath, instance_filepath, sketch_filepath, plan_filepath)
        plan_found_line: str = None
        with stdout_filepath.open("w") as fd:
            for line in stdout:
                fd.write(f"{line}\n")
                if line.startswith("Plan found with cost:"):
                    plan_found_line = line

        if plan_found_line is not None:
            cost = int(plan_found_line.split(" ")[-1])
            solved_by_sketch[instance_filepath.name] = cost

    # Statistics
    num_solved_by_sketch = len(solved_by_sketch)
    logging.info(f"{num_solved_by_planner} instance(s) solvable by {args.planner}")
    logging.info(f"{num_solved_by_sketch} instance(s) solved by sketch: {100 * num_solved_by_sketch / num_solved_by_planner:0.2f} {100 * num_solved_by_sketch / num_instances:0.2f}")

