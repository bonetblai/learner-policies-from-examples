import logging
import sys

from termcolor import colored
from typing import Set, Tuple, List, Union, Dict, Any, Optional, Union
from pathlib import Path

# Run planner in forked process to isolated it
import multiprocessing as mp
import subprocess
import os


def _call_planner(domain_filepath: Path,
                  instance_filepath: Path,
                  plan_filepath: Path,
                  planner: str, 
                  remove_files: bool) -> Union[str, List[str]]:
    if "+" in planner:
        # This is a compound call
        planners = planner.split("+")
        for subplanner in planners:
            logging.info(f"planner: using {subplanner}")
            output: Union[str, List[str]] = _call_planner(domain_filepath, instance_filepath, plan_filepath, subplanner, remove_files)
            if type(output) != str:
                return output
        return "PLANNER-ERROR"

    else:
        # Atomic planners
        if planner == "bfws":
            from ..planners.libbfws import BFWS as Planner
            from ..fd import grounding
            try:
                task = Planner()
                grounding.default(str(domain_filepath), str(instance_filepath), task)
                task.plan_filename = str(plan_filepath)
                task.log_filename = "bfws.log"
                task.search = "DUAL-BFWS"
                task.setup()
                task.solve()
            except SystemExit as se:
                logging.info(f"Catched system-exit={se} during {planner} call...")
                return "PLANNER-ERROR"

        elif planner == "siw":
            from ..planners.libsiw import SIW_Planner as Planner
            from ..fd import grounding
            try:
                task = Planner()
                grounding.default(str(domain_filepath), str(instance_filepath), task)
                task.plan_filename = str(plan_filepath)
                task.log_filename = "siw.log"
                task.iw_bound = 2
                task.setup()
                task.solve()
            except SystemExit as se:
                logging.info(f"Catched system-exit={se} during {planner} call...")
                return "PLANNER-ERROR"

        elif planner == "siw_plus":
            from ..planners.libsiw_plus import SIW_Plus_Planner as Planner
            from ..fd import grounding
            try:
                task = Planner()
                grounding.default(str(domain_filepath), str(instance_filepath), task)
                task.plan_filename = str(plan_filepath)
                task.log_filename = "siw_plus.log"
                task.iw_bound = 2
                task.setup()
                task.solve()
            except SystemExit as se:
                logging.info(f"Catched {se} during {planner} call...")
                return "PLANNER-ERROR"

        else:
            logging.error(f"ERROR: Unexpected planner '{planner}'")
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

def get_plan(domain_filepath: Path,
             instance_filepath: Path,
             plan_filepath: Path,
             planner: str,
             remove_files: bool = True) -> Tuple[bool, List[str]]:
    logging.info(f"Planner:  planner: {planner}")
    logging.info(f"Planner:   domain: {domain_filepath}")
    logging.info(f"Planner: instance: {instance_filepath}")
    logging.info(f"Planner:     plan: {plan_filepath}")
    logging.info(f"Planner:      cwd: {os.getcwd()}")

    q: mp.Queue = mp.Queue()
    p: mp.Process = mp.Process(target=_planner_wrapper, args=(domain_filepath, instance_filepath, plan_filepath, planner, remove_files, q))
    p.start()
    output: Tuple[bool, List[str]] = q.get()
    logging.info(f"Planner:  status: {output[0]}")
    logging.info(f"Planner:    plan: {output[1]}")
    p.join()

    return output

def get_plan_v2(domain_filepath: Path, instance_filepath: Path, plan_filepath: Path, planner: str, remove_files: bool = True) -> Tuple[bool, List[str]]:
    logging.debug(f"Planner(v2):  planner: {planner}")
    logging.debug(f"Planner(v2):   domain: {domain_filepath}")
    logging.debug(f"Planner(v2): instance: {instance_filepath}")
    logging.debug(f"Planner(v2):     plan: {plan_filepath}")
    logging.debug(f"Planner(v2):      cwd: {os.getcwd()}")

    call_planner_filepath: Path = Path(os.path.dirname(os.path.abspath(__file__))) / "../planners/call_planner.py"

    args: List[str] = ["python", str(call_planner_filepath), planner, str(domain_filepath), str(instance_filepath), str(plan_filepath)]
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    p.wait()
    #output = p.communicate()[0]
    #print(output)

    # Read output and determine status
    if plan_filepath.exists():
        with plan_filepath.open("r") as fd:
            plan: List[str] = [line.strip("\n") for line in fd.readlines()]
            is_unsolvable: bool = "NO-PLAN" in plan
            status: bool = not is_unsolvable
            if is_unsolvable: plan = []
    else:
        logging.debug(f"No plan filepath '{plan_filepath}'")
        status: bool = False
        plan: List[str] = []

    logging.debug(f"Planner(v2):  status: {status}")
    logging.debug(f"Planner(v2):    plan: {plan}")
 
    # Clean up and return
    if remove_files: plan_filepath.unlink()
    return status, plan
