import sys
import argparse
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Set, Any
from collections import deque, defaultdict
from subprocess import Popen, PIPE


def submit_job(jdesc: Dict[str, Any], suffix: str, arguments: Any) -> Tuple[str]:
    job_name: str = jdesc.get("name")
    job_domain: str = jdesc.get("domain")
    job_args: List[str] = jdesc.get("args", []) + jdesc.get("extra", [])
    job_time: str = arguments.time or jdesc.get("time") or "8:00:00"
    assert job_name is not None and job_domain is not None

    args: List[str] = ["sbatch", "--partition", arguments.partition, "--time", job_time, "scripts/submit/job.slurm", job_domain, suffix] + job_args
    logging.debug(f"Submit: name=|{job_name}|, partition=|{arguments.partition}|, domain=|{job_domain}|, suffix=|{suffix}|, args=|{args}|")

    with Popen(args, stdout=PIPE) as proc:
        output = proc.stdout.read().decode("utf-8").strip("\n")

    logging.debug(f"Submit: output=|{output}|")
    job_id: str = output.split(" ")[-1]
    logging.debug(f"Submit: {job_name} {job_id}")
    return job_name, job_id

def remove_comment(line):
    pos = line.find("#")
    return line if pos == -1 else line[:pos]

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
    default_database: Path = Path("scripts/default_database.txt")
    parser = argparse.ArgumentParser(description="Submit SLURM job:")
    parser.add_argument("--clear_default_log_folder", action='store_true', default=False, help="Whether to remove existing files in default log folder. Default is False.")
    parser.add_argument("--db", type=Path, default=default_database, required=False, help=f"Path to jobs database (DB). Default is '{default_database}'.")
    parser.add_argument("--db_inline", type=str, default=None, required=False, help="Inline database extension; must evaluate to DB. Default is None.")
    parser.add_argument("--debug", action='store_true', default=False, help="Turn on debugging output. Default is False.")
    parser.add_argument("--dryrun", action='store_true', default=False, help="Whether to perfrom a dry run. Default is False.")
    parser.add_argument("--partition", type=str, default="rleap_cpu", help="Slurm partition. Default is 'rleap_cpu'.")
    parser.add_argument("--time", type=str, default=None, help="Override any time limit; format is Slurm's format. Default is None.")
    parser.add_argument("--suffix", type=str, default=None, help="Alternate suffix for log folder. Default is <jspec>.")
    parser.add_argument("jspec", type=str, nargs=1, help="Job spec to submit: either entry in database, PREFIX:<prefix>, or ALL.")
    parser.add_argument("extra_arguments", type=str, default=None, nargs="*", help="Extra (overriding) arguments for job. Default is None.")
    arguments = parser.parse_args()

    if arguments.debug:
        logger.setLevel(logging.DEBUG)

    logging.debug(f"Parsed:              db=|{arguments.db}|")
    logging.debug(f"Parsed:           jspec=|{arguments.jspec}|")
    logging.debug(f"Parsed: extra_arguments=|{arguments.extra_arguments}|")

    if arguments.db is None or not arguments.db.exists():
        logging.error(f"Invalid or inexistent database '{arguments.db}'")
        exit(-1)

    # Clear log folder
    if arguments.clear_default_log_folder:
        default_log_folder: Path = Path("slurm/logs/default")
        log_files: List[Path] = [file for file in default_log_folder.glob("*") if file.is_file()]
        for file in log_files:
            file.unlink()
        logging.info(f"{len(log_files)} file(s) removed in {default_log_folder}")

    # Read job database
    with arguments.db.open("r") as fd:
        lines: List[str] = [remove_comment(line.strip("\n ")) for line in fd.readlines()]
        database: Dict[str, Any] = eval(" ".join(lines))

    # Extend database with additional entries
    if arguments.db_inline is not None:
        database_inline: Dict[str, Any] = eval(arguments.db_inline)
        database.update(database_inline)
    logging.debug(f"Database: {database}")

    # Get list of jobs by parsing job spec
    jobs: List[Dict[str, Any]] = []
    jspec: str = arguments.jspec[0]

    # Define suffix
    suffix: str = f".{jspec}" if arguments.suffix is None else arguments.suffix

    # Read database
    if jspec in database:
        q = deque([{"name": jspec}])
        while len(q) > 0:
            jdesc: Dict[str, Any] = q.popleft()
            jname: str = jdesc.get("name")
            xrefs: List[str] = jdesc.get("xrefs", [])
            args: List[str] = jdesc.get("args", [])
            time: str = jdesc.get("time")
            if jname not in database:
                logging.warning(f"Skipping job '{jname}' because it is not database.")
                continue
            elif "group" in database.get(jname):
                xrefs: List[str] = list(database.get(jname).get("xrefs", [])) + xrefs
                args: List[str] = list(database.get(jname).get("args", [])) + args
                time: str = database.get(jname).get("time") or time
                for jspec in database.get(jname).get("group"):
                    q.append({"name": jspec, "xrefs": xrefs, "args": args, "time": time})
            else:
                xrefs: List[str] = list(database.get(jname).get("xrefs", [])) + xrefs
                args: List[str] = list(database.get(jname).get("args", [])) + args
                time: str = database.get(jname).get("time") or time
                jdesc: Dict[str, Any] = {"name": jname}
                jdesc.update(database.get(jname))
                jdesc.update({"xrefs": xrefs, "args": args, "time": time})
                jobs.append(jdesc)

    elif jspec == "ALL":
        jobs = [{"name": job} for job, desc in database.items() if "group" not in desc]

    elif jspec.startswith("PREFIX:"):
        prefix: str = jspec[7:]
        jobs = [{"name": job} for job, desc in database.items() if "group" not in desc and job.startswith(prefix)]

    else:
        logging.warning(f"Job spec '{jspec}' not found in database")

    # Submit jobs with additional arguments (if any)
    logging.debug(f"Jobs: {jobs}")
    for jdesc in jobs:
        # Resolve xrefs
        args: List[str] = list(jdesc.get("args", []))
        queue = deque(list(jdesc.get("xrefs", [])))
        while len(queue) > 0:
            xref: str = queue.popleft()
            if xref in database:
                logging.debug(f"Resolving xref '{xref}' for job '{jdesc.get('name')}'")
                for xref_sub in database.get(xref).get("xrefs", []):
                    queue.appendleft(xref_sub)
                args: List[str] = database.get(xref).get("args", []) + args
        logging.debug(f"args: {args}")

        # Add extra arguments (if any)
        jdesc.update({"args": args, "extra": arguments.extra_arguments})
        logging.debug(f"jdesc: {jdesc}")

        if not arguments.dryrun:
            job_name, job_id = submit_job(jdesc, suffix, arguments)
            logging.info(f"Job id {job_id} -> {job_name}")
        else:
            logging.info(f"Dry run: {jdesc}")

    if not arguments.dryrun:
        logging.info(f"{len(jobs)} job(s) submitted")

