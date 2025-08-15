import sys, shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import multiprocessing as mp
import subprocess
import os


def _call_planner(domain_filepath: Path, instance_filepath: Path, plan_filepath: Path, planner: str, remove_files: bool) -> Union[str, List[str]]:
    # "Complex" planners
    if "+" in planner:
        # This is a compound call
        planners = planner.split("+")
        for subplanner in planners:
            print(f"planner: using {subplanner}")
            output: Union[str, List[str]] = _call_planner(domain_filepath, instance_filepath, plan_filepath, subplanner, remove_files)
            if type(output) != str:
                return output
        return "PLANNER-ERROR"

    # Atomic planners
    if planner == "bfws":
        from planners.libbfws import BFWS as Planner
        from fd import grounding
        try:
            task = Planner()
            grounding.default(str(domain_filepath), str(instance_filepath), task)
            task.plan_filename = str(plan_filepath)
            task.log_filename = "bfws.log"
            task.search = "DUAL-BFWS"
            task.setup()
            task.solve()
        except SystemExit as se:
            print(f"Catched system-exit={se} during {planner} call...")
            return "PLANNER-ERROR"

    elif planner == "siw":
        from planners.libsiw import SIW_Planner as Planner
        from fd import grounding
        try:
            task = Planner()
            grounding.default(str(domain_filepath), str(instance_filepath), task)
            task.plan_filename = str(plan_filepath)
            task.log_filename = "siw.log"
            task.iw_bound = 2
            task.setup()
            task.solve()
        except SystemExit as se:
            print(f"Catched system-exit={se} during {planner} call...")
            return "PLANNER-ERROR"

    elif planner == "siw_plus":
        from planners.libsiw_plus import SIW_Plus_Planner as Planner
        from fd import grounding
        try:
            task = Planner()
            grounding.default(str(domain_filepath), str(instance_filepath), task)
            task.plan_filename = str(plan_filepath)
            task.log_filename = "siw_plus.log"
            task.iw_bound = 2
            task.setup()
            task.solve()
        except SystemExit as se:
            print(f"Catched {se} during {planner} call...")
            return "PLANNER-ERROR"

    else:
        print(f"ERROR: Unexpected planner '{planner}'")
        raise RuntimeError(f"ERROR: Unexpected planner '{planner}'")

    plan_ipc_filepath = Path("plan.ipc")
    if plan_ipc_filepath.exists():
        plan_ipc_filepath.rename(plan_filepath)
        assert plan_filepath.exists() and not plan_ipc_filepath.exists()

    # Extract plan
    assert plan_filepath.exists()
    with plan_filepath.open("r") as fd:
        plan: List[str] = [line.strip() for line in fd.readlines()]

    # Clean up
    execution_details_filepath = Path("execution.details")
    if execution_details_filepath.exists() and remove_files:
        execution_details_filepath.unlink()

    if plan_filepath.exists() and remove_files:
        plan_filepath.unlink()

    return plan

def _planner_wrapper(domain_filepath: Path, instance_filepath: Path, plan_filepath: Path, planner: str, remove_files: bool, q):
    output: Union[str, List[str]] = _call_planner(domain_filepath, instance_filepath, plan_filepath, planner, remove_files)
    if type(output) == str:
        q.put((False, []))
    else:
        q.put((True, output))

def get_plan(domain_filepath: Path, instance_filepath: Path, plan_filepath: Path, planner: str, remove_files: bool = True) -> Tuple[bool, List[str]]:
    q: mp.Queue = mp.Queue()
    p: mp.Process = mp.Process(target=_planner_wrapper, args=(domain_filepath, instance_filepath, plan_filepath, planner, remove_files, q))
    p.start()
    output: Tuple[bool, List[str]] = q.get()
    print(f"Planner:  status: {output[0]}")
    print(f"Planner:    plan: {output[1]}")
    p.join()
    return output


if __name__ == "__main__":
    # Check sufficient number of arguments
    if len(sys.argv) < 5:
        print(f"Usage: {sys.argv[0]} <planner> <domain> <instance> <plan>")
        exit(-1)

    # Read and verify arguments
    planner: str = sys.argv[1]
    domain_filepath: Path = Path(sys.argv[2])
    instance_filepath: Path = Path(sys.argv[3])
    plan_filepath: Path = Path(sys.argv[4])

    output: Tuple[bool, List[str]] = get_plan(domain_filepath, instance_filepath, plan_filepath, planner, remove_files=False)
    if not output[0]:
        # Planner failed, create dummy plan file
        with plan_filepath.open("w") as fd:
            fd.write("NO-PLAN\n")
    exit(0)
