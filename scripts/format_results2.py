import sys
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple, List, Any, Set


def read_data(filepath: Path) -> Dict[str, Any]:
    assert ".output." in filepath.name

    success: str = None
    lines: List[str] = []
    with filepath.open("r") as fd:
        raw_line: str = fd.readline().strip(" \n")
        while raw_line:
            lines.append(raw_line)
            if "SUCCESS" in raw_line: success = raw_line
            raw_line: str = fd.readline().strip("\n")

    parsing: str = None
    data: Dict[str, Any] = {
        "domain": filepath.name[:filepath.name.find(".output.")],
        "success": success[2 + success.find("- "):] if success is not None else None,
        "filepath": filepath,
        "stats": defaultdict(dict),
        "failure": defaultdict(dict),
    }

    # Required initializations
    data["stats"]["iterations"]["inner"] = 0
    data["stats"]["iterations"]["inner/last"] = 0

    # Process lines
    for raw_line in lines:
        parsed_line: str = raw_line[2 + raw_line.find(" - "):].strip(" ")
        fields: List[str] = parsed_line.split(" ")

        # "Call" line
        if "Call:" in parsed_line:
            call_line: str = parsed_line[parsed_line.find("python"):]
            data["call"] = call_line

            planner: str = call_line[9 + call_line.rfind("--planner"):]
            planner: str = planner[: planner.find("--")].strip(" ")
            data["planner"] = planner

        # Initial date/time
        elif "Read and curate" in parsed_line:
            word1, word2 = raw_line.split(" ")[:2]
            data["initial/date"] = word1
            data["initial/time"] = word2[:word2.find(",")]

        # UUID
        elif "UUID:" in parsed_line:
            parsed_line: str = parsed_line[parsed_line.find("UUID:"):]
            uuid: str = parsed_line[2 + parsed_line.find(":"):]
            data["uuid"] = uuid

        # Instances
        elif "instance(s) after removal and re-ordering;" in parsed_line:
            num_instances: int = int(fields[0])
            data["instances"] = num_instances

        # Relevant vertices in collection
        elif "initial relevant vertice(s)" in parsed_line:
            relevant_vertices: int = int(fields[0])
            data["vertices"] = relevant_vertices

        # Feature repository
        elif "Found matching feature repository" in raw_line:
            parsed_line: str = parsed_line[parsed_line.find("Found matching feature repository"):]
            feature_repo: str = Path(eval(parsed_line[parsed_line.find("'"):])).name
            data["feature_repo"] = feature_repo

        elif "Feature statistics:" in parsed_line:
            parsed_line: str = parsed_line[parsed_line.find("Feature statistics:"):]
            features: Dict[str, Any] = eval(parsed_line[1 + parsed_line.find(":"):])
            data["features"] = features

        # iterations: outer and inner
        elif "Iteration:" in parsed_line:
            parsed_line: str = parsed_line[parsed_line.find("Iteration:"):]
            iterations: int = 1 + int(parsed_line[2 + parsed_line.find(":"):])
            data["stats"]["iterations"]["outer"] = iterations
            data["stats"]["iterations"]["inner/last"] = 0

        elif "Inner iterations:" in parsed_line:
            parsed_line: str = parsed_line[parsed_line.find("Inner iterations:"):]
            inner_iterations: int = int(parsed_line[2 + parsed_line.find(":"):])
            assert inner_iterations == data["stats"]["iterations"]["inner/last"]
            #data["stats"]["iterations"]["inner"] += inner_iterations
            #data["stats"]["iterations"]["inner/last"] = inner_iterations

        elif "Total inner iterations:" in parsed_line:
            parsed_line: str = parsed_line[parsed_line.find("Total inner iterations:"):]
            total_inner_iterations: int = int(parsed_line[2 + parsed_line.find(":"):])
            data["stats"]["iterations"]["inner/total"] = total_inner_iterations
            assert total_inner_iterations == data["stats"]["iterations"].get("inner")

        # Bundle for last iteration
        elif parsed_line.startswith("Bundle:"):
            assert "Bundle: index=0/0, bundle=0." in parsed_line, parsed_line
            parsed_line: str = parsed_line[len("Bundle: index=0/0, bundle=0.") + parsed_line.find("Bundle: index=0/0, bundle=0."): parsed_line.find(", #relevant_vertices")]
            bundle: Dict[int, Tuple[int]] = dict(eval(parsed_line))
            data["stats"]["last"]["bundle"] = bundle
            data["stats"]["iterations"]["inner"] += 1
            data["stats"]["iterations"]["inner/last"] += 1

        # Last iteration
        elif "requirement(s) split as" in parsed_line:
            data["stats"]["last"]["requirements"] = eval(parsed_line[parsed_line.find("{"):])

        # Ext_edges (i.e. good edges) for last iteration
        elif "ext_edges:" in parsed_line:
            parsed_line: str = parsed_line[parsed_line.find("ext_edges:"):]
            ext_edges: List[Tuple[int, int]] = eval(parsed_line[parsed_line.find("["):])
            data["stats"]["last"]["ext_edges"] = ext_edges

        # Deadends
        elif "deadend_ext_states" in parsed_line:
            parsed_line: str = parsed_line[parsed_line.find("deadend_ext_states:"):]
            deadend_ext_states: List[Tuple[int, int]] = eval(parsed_line[parsed_line.find("["):])
            data["stats"]["last"]["deadend_ext_states"] = deadend_ext_states

        elif "Deadend paths:" in parsed_line:
            parsed_line: str = parsed_line[parsed_line.find("Deadend paths:"):]
            deadend_paths: List[Tuple[Tuple[int, int]]] = eval(parsed_line[parsed_line.find("["):])
            data["stats"]["last"]["deadend_paths"] = deadend_paths

        # Pricing algorithm
        elif "second(s) for preprocessing termination" in parsed_line:
            parsed_line: str = parsed_line[2 + parsed_line.find("- "):].strip(" ")
            fields: List[str] = parsed_line.split(" ")
            pricing_preprocessing = float(fields[0])
            data["stats"]["last"]["pricing/preprocessing"] = pricing_preprocessing

        elif "Greedy solver finished in" in parsed_line:
            parsed_line: str = parsed_line[2 + parsed_line.find("- "):].strip(" ")
            fields: List[str] = parsed_line.split(" ")
            pricing_algorithm = float(fields[4])
            data["stats"]["last"]["pricing/algorithm"] = pricing_algorithm

        # Failures
        elif "No eligible features:" in parsed_line:
            parsing = "no_eligible_features"
            data["failure"]["reason"] = "Eligible"
            data["failure"]["requirements"] = defaultdict(list)

        elif parsing == "no_eligible_features" and "key=" in parsed_line:
            parsed_line: str = parsed_line[parsed_line.find("key="):]
            key: str = parsed_line[4: parsed_line.find(",")]
            data["failure"]["requirements"][key].append(raw_line[2 + raw_line.find(" - "):].strip(" "))

        elif "No feature to separate transition to deadend state" in parsed_line:
            data["failure"]["reason"] = "Dead-end"

        elif "** NO MORE TRAINING INSTANCES" in parsed_line:
            data["failure"]["reason"] = "Exhausted"

        elif "ERROR: Unexpected deadend state" in parsed_line:
            data["failure"]["reason"] = "Tutor-DEAD"

        elif "Zero instances remain" in parsed_line:
            data["failure"]["reason"] = "Tutor"

        # "SOLVES ALL"
        elif "SOLVES ALL" in parsed_line:
            data["final/date"] = fields[0]
            data["final/time"] = fields[1][:fields[1].find(",")]

            parsed_line: str = parsed_line[parsed_line.find("Sketch"):]
            data["solves_all"] = parsed_line

            bundle: str = parsed_line[1 + parsed_line.find("["):]
            bundle: Dict[int, Tuple[int]] = dict(eval(bundle[bundle.find("["): -1]))
            data["solves_all/bundle"] = bundle
            assert bundle == data["stats"]["last"].get("bundle")

        # Learning statistics
        elif "num_training_instances:" in parsed_line:
            data["stats"]["learning"]["instances"] = int(fields[1])
            assert data["stats"]["learning"]["instances"] == data["instances"]
        elif "num_selected_training_instances (|P|):" in parsed_line:
            data["stats"]["learning"]["selected"] = int(fields[2])
        elif "num_states_in_selected_training_instances (|S|):" in parsed_line:
            data["stats"]["learning"]["states"] = int(fields[2])
        elif "num_features_in_pool (|F|):" in parsed_line:
            data["stats"]["learning"]["features"] = int(fields[2])

        # Memory statistics
        elif "Total memory:" in parsed_line:
            data["stats"]["memory"]["total"] = float(fields[2])

        # Timers
        elif "Total time:" in parsed_line:
            data["stats"]["time"]["total"] = float(fields[2])
        elif "Preprocessing time:" in parsed_line:
            data["stats"]["time"]["preprocessing"] = float(fields[2])
        elif "Feature pool generation time:" in parsed_line:
            data["stats"]["time"]["pool-generation"] = float(fields[4])
        elif "Preprocessing of feature termination time:" in parsed_line:
            data["stats"]["time"]["feature-termination"] = float(fields[5])
        elif "Learner time:" in parsed_line:
            data["stats"]["time"]["learner"] = float(fields[2])
        elif "Verification time:" in parsed_line:
            data["stats"]["time"]["verification"] = float(fields[2])
        elif "ASP time:" in parsed_line:
            data["stats"]["time"]["asp"] = float(fields[2])
        elif "Planner time:" in parsed_line:
            data["stats"]["time"]["planner"] = float(fields[2])
        elif "MPairs time:" in parsed_line:
            data["stats"]["time"]["mpairs"] = float(fields[2])
        elif "Indexing time:" in parsed_line:
            data["stats"]["time"]["indexing"] = float(fields[2])
        elif "Pricing/preprocessing time:" in parsed_line:
            data["stats"]["time"]["pricing/preprocessing"] = float(fields[2])
        elif "Pricing/algorithm time:" in parsed_line:
            data["stats"]["time"]["pricing/algorithm"] = float(fields[2])

        # Final sketch
        elif "Numer of sketch rules:" in parsed_line:
            data["stats"]["policy"]["rules"] = int(fields[4])
        elif "Number of selected features:" in parsed_line:
            data["stats"]["policy"]["features"] = int(fields[4])
        elif "Maximum complexity of selected feature:" in parsed_line:
            data["stats"]["policy"]["max_complexity"] = int(fields[5])
        elif "Sum complexity of selected features:" in parsed_line:
            data["stats"]["policy"]["sum_complexity"] = int(fields[5])

    if data.get("success") is None and data["failure"].get("reason") is None:
        # This is an unplained failure, try to find cause in ".error" file
        error_filepath: Path = filepath.parent / str(filepath.name).replace(".output.", ".error.")
        if error_filepath.exists():
            with error_filepath.open("r") as fd:
                lines: List[str] = [line.strip("\n") for line in fd.readlines()]
            if len(lines) > 0 and "TIME LIMIT" in lines[0]:
                data["failure"]["reason"] = "Timeout"
            elif len(lines) > 0 and "out-of-memory handler" in lines[0]:
                data["failure"]["reason"] = "Memout"
        if data["failure"].get("reason") is None:
            data["failure"]["reason"] = "UNEXPLAINED"

    assert data.get("success") is not None or data["failure"].get("reason") is not None
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Formatted pring of experimental results.")
    parser.add_argument("--only_successes", action='store_true', default=False, help="Whether only print successful records. Default is False.")
    parser.add_argument("--only_failures", action='store_true', default=False, help="Whether only print failed records. Default is False.")
    parser.add_argument("--no_blacklist", action='store_true', default=False, help="Report results for all domains. Default is False.")
    parser.add_argument("--no_translation", action='store_true', default=False, help="Don't rename domains. Default is False.")
    parser.add_argument("folders", nargs="*", help="Folders containing results.")
    args = parser.parse_args()

    # Read data files to be processed
    folder_paths: List[Path] = [Path(folder) for folder in args.folders]
    output_files: List[Path] = [filepath for folder_path in folder_paths for filepath in folder_path.glob("*") if ".output." in filepath.name]

    # Process each data file
    datas: List[Dict[str, Any]] = []
    for filepath in output_files:
        if not filepath.name.endswith(".swp"):
            data: Dict[str, Any] = read_data(filepath)
            datas.append(data)

    num_records = len(datas)
    num_successes = len([data for data in datas if data.get("success") is not None])
    print(f"{num_records} record(s): {num_successes} success(es) + {num_records - num_successes} failure(s)")

    # Domain translations
    translation: Dict[str, str] = {
        "barman-1cocktail": "Barman-1cocktail",
        "barman-1cocktail-1shot": "Barman-1cocktail-1shot",
        "blocks_3": "Blocks3ops",
        "blocks_3-v2": "Blocks3ops",
        "blocks_4-v2": "Blocks4ops",
        "blocks_4_clear_no_constants": "Blocks4ops-clear",
        "blocks_4_on_no_constants": "Blocks4ops-on",
        "childsnack": "Childsnack",
        "delivery": "Delivery",
        "delivery1": "Delivery-1pkg",
        "depot": "Depot",
        "driverlog": "Driverlog",
        "gripper": "Gripper",
        "ferry": "Ferry",
        "freecell": "Freecell",
        "grid-indexicals": "Grid-indexicals",
        "miconic": "Miconic",
        "logistics": "Logistics",
        "logistics-1pkg": "Logistics-1pkg",
        "logistics-1truck": "Logistics-1truck",
        "logistics-1truck-indexicals": "Logistics-1truck-idx",
        "logistics2-indexicals": "Logistics2-indexicals",
        "reward": "Reward",
        "rovers": "Rovers",
        "satellite": "Satellite",
        "sokoban-1stone-v3-7x7": "Sokoban-1stone-v3-7x7",
        "spanner": "Spanner",
        "spanner-1nut": "Spanner-1nut",
        "tetris-opt14-strips-no-costs": "Tetris-opt14-strips-no-costs",
        "tidybot-opt11-strips": "Tidybot-opt11-strips",
        "hiking": "Hiking",
        "tpp": "Tpp",
        "visitall": "Visitall",
        "zenotravel-1plane": "Zenotravel-1plane",
        "zenotravel-1person": "Zenotravel-1person",
        "zenotravel-1plane-1person": "Zenotravel-1plane-1person",
    }

    # Report successes
    if not args.only_failures:
        blacklist: Set[str] = {
            "3puzzle",
            "barman2-no-costs",
            "blocks_4",
            "childsnack_untyped",
            "delivery_untyped",
            "logistics-1truck-indexicals-v2",
            "logistics2-1truck-indexicals-v2",
            "logistics2-1truck",
            "grid-indexicals",
            "grid-indexicals-v2",
            "visitall2",
        }
        print(f"Blacklist: {sorted(blacklist)}")
        print("")

        def key(item):
            strategy: str = item["filepath"].parent.name.split(".")[-1]
            outer = int(item["stats"]["iterations"]["outer"])
            size_Qp = int(item["stats"]["learning"]["selected"])
            inner = int(item["stats"]["iterations"]["inner/last"])
            domain = item["domain"]
            return (strategy, outer, size_Qp, inner, domain)

        num_successes: int = 0
        selected_datas: List[Dict[str, Any]] = [data for data in datas if data["success"] is not None]
        sorted_selected_datas: List[Dict[str, Any]] = sorted([data for data in selected_datas if args.no_blacklist or data.get("domain") not in blacklist], key=key)
        for data in sorted_selected_datas:
            num_successes += 1
            domain = data["domain"]
            domain += "-1nut" if domain == "spanner" and data["planner"] == "siw+bfws" else ""
            domain = domain if args.no_translation else translation.get(domain, domain)
            if "--simplify_only_conditions" in data["call"]:
                domain += "*"
            size_Q = data["stats"]["learning"]["instances"]
            size_S = data.get("vertices", -1)
            size_F = data["stats"]["learning"]["features"]
            strategy: str = data["filepath"].parent.name.split(".")[-1]
            strategy: str = "$S_1$" if strategy == "forward" else "$S_2$"
            outer = data["stats"]["iterations"]["outer"]
            inner_global = data["stats"]["iterations"]["inner"]
            size_Qp = data["stats"]["learning"]["selected"]
            inner = data["stats"]["iterations"]["inner/last"]
            if "ext_edges" not in data["stats"]["last"]: print(data)
            good = len(data["stats"]["last"]["ext_edges"])
            bad = len(data["stats"]["last"]["deadend_ext_states"])
            size_H = sum(list(data["stats"]["last"]["requirements"].values()))
            size_G = data["stats"]["policy"]["features"]
            size_pi = data["stats"]["policy"]["rules"]
            max_complexity = data["stats"]["policy"]["max_complexity"]
            #prep = data["stats"]["last"]["pricing/preprocessing"]
            #alg = data["stats"]["last"]["pricing/algorithm"]
            prep = data["stats"]["time"]["feature-termination"]
            alg = data["stats"]["time"]["pricing/algorithm"]
            ver = data["stats"]["time"]["verification"]
            total = data["stats"]["time"]["total"]
            domain_str = f"\\Verb!{domain}!"
            print(f"{domain_str:40} & {size_Q:6,} & {size_S:6,} & {size_F:8,} & {strategy:5} & {outer:6} & {inner_global:6,} & {size_Qp:7} & {inner:6} & {good:8} & {bad:8} & {size_H:5,} & {size_G:6} & {size_pi:7} & {prep:10,.2f} & {alg:10,.2f} & {ver:10,.2f} & {total:10,.2f} \\\\")

    # Report failures
    if not args.only_successes:
        blacklist: Set[str] = {
            "barman-no-costs",
            "barman2-no-costs",
            "barman3-no-costs",
            "blocks_4",
            "depot2",
            "depot3",
            "hiking2",
            "hiking3",
            "logistics-indexicals",
            "logistics2",
            "zenotravel2",
            "zenotravel3",
        }
        print(f"Blacklist: {sorted(blacklist)}")
        print("")

        def key(item):
            strategy: str = item["filepath"].parent.name.split(".")[-1]
            outer = item["stats"].get("iterations", dict()).get("outer", -1)
            size_Qp =  -1 if "last" not in item["stats"] else len(item["stats"]["last"]["bundle"])
            inner = int(item["stats"]["iterations"]["inner/last"])
            domain = item["domain"]
            return (strategy, outer, size_Qp, inner, domain)

        num_failures: int = 0
        selected_datas: List[Dict[str, Any]] = [data for data in datas if data["success"] is None]
        sorted_selected_datas: List[Dict[str, Any]] = sorted([data for data in selected_datas if args.no_blacklist or data.get("domain") not in blacklist], key=key)
        for data in sorted_selected_datas:
            try:
                num_failures += 1
                domain = data["domain"]
                domain += "-1nut" if domain == "spanner" and data["planner"] == "siw+bfws" else ""
                domain = domain if args.no_translation else translation.get(domain, domain)
                if "--simplify_only_conditions" in data["call"]:
                    domain += "*"
                size_Q = -1 if data["instances"] == 0 else data["instances"]
                size_S = data.get("vertices", -1)
                size_F = data.get("features", dict()).get("numerical_features", 0) + data.get("features", dict()).get("boolean_features", 0)
                strategy: str = data["filepath"].parent.name.split(".")[-1]
                strategy: str = "$S_1$" if strategy == "forward" else "$S_2$"
                outer = data["stats"].get("iterations", dict()).get("outer", -1)
                inner_global = data["stats"]["iterations"]["inner"]
                size_Qp = -1 if "last" not in data["stats"] else len(data["stats"]["last"]["bundle"])
                inner = data["stats"].get("iterations", dict()).get("inner/last", -1)
                good = -1 if "last" not in data["stats"] else len(data["stats"]["last"]["ext_edges"])
                bad = -1 if "last" not in data["stats"] else len(data["stats"]["last"]["deadend_ext_states"])
                size_H = -1 if "last" not in data["stats"] else sum(list(data["stats"]["last"]["requirements"].values()))
                reason = data["failure"]["reason"]
                if reason == "Eligible":
                    keys: List[str] = sorted(set(data["failure"]["requirements"].keys()))
                    reason = "-".join(keys)

                #prep = -1 if "last" not in data["stats"] else data["stats"]["last"]["pricing/preprocessing"]
                #alg = -1 if "last" not in data["stats"] else data["stats"]["last"].get("pricing/algorithm", -1)
                prep = data["stats"]["time"].get("feature-termination", -1)
                alg = data["stats"]["time"].get("pricing/algorithm", -1)
                ver = data["stats"]["time"].get("verification", -1)
                total = data["stats"]["time"].get("total", -1)
                domain_str = f"\\Verb!{domain}!"
                if reason != "UNEXPLAINED":
                    print(f"{domain_str:40} & {size_Q:6,} & {size_S:6,} & {size_F:8,} & {strategy:5} & {outer:6} & {inner_global:6,} & {size_Qp:7} & {inner:6} & {good:8} & {bad:8} & {size_H:5,} & {reason:>16} & {prep:10,.2f} & {alg:10,.2f} & {ver:10,.2f} & {total:10,.2f} \\\\")
            except:
                print(f"Error: {data}")

    # Analyze feature pools
    largest_pool: Dict[str, Any] = None
    slowest_pool: Dict[str, Any] = None
    for data in datas:
        pool = data.get("features")
        if pool is not None:
            if largest_pool == None or pool.get("numerical_features") + pool.get("boolean_features") > largest_pool.get("numerical_features") + largest_pool.get("boolean_features"):
                largest_pool = pool
            if slowest_pool == None or pool.get("generation_time", -1) > slowest_pool.get("generation_time"):
                slowest_pool = pool

    print("")
    print(f"Largest pool: {largest_pool}")
    print(f"Slowest pool: {slowest_pool}")
