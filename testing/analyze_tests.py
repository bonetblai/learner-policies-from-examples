import sys
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Set, Any

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
    parser.add_argument("--output", type=Path, required=True, help="Path to output folder.")
    args = parser.parse_args()

    # Read dmains
    domains: List[str] = [domain for domain in args.output.glob("*") if domain.is_dir()]

    # Read output files
    outputs: Dict[str, Any] = dict()
    for domain in domains:
        datas: List[Dict[str, Any]] = []
        for output_file in domain.glob("*.output"):
            data: Dict[str, Any] = dict()
            with output_file.open("r") as fd:
                lines: List[str] = [line.strip("\n") for line in fd.readlines()]
            data["lines"] = lines
            data["valid"] = 1 if "Plan valid" in lines else 0
            max_eff_width: List[str] = [line for line in lines if line.startswith("Max ef. width:")]
            avg_eff_width: List[str] = [line for line in lines if line.startswith("Average ef. width:")]
            max_eff_width: float = float(max_eff_width[0].split(" ")[-1])
            avg_eff_width: float = float(avg_eff_width[0].split(" ")[-1])
            data["max_eff_width"] = max_eff_width
            data["avg_eff_width"] = avg_eff_width
            datas.append(data)
        outputs[domain.name] = datas

    # Domain translations
    translation: Dict[str, str] = {
        "8puzzle-1tile": "8puzzle-1tile-fixed",
        "8puzzle-1tile-v2": "8puzzle-1tile",
        "barman-1cocktail": "Barman-1cocktail",
        "barman-1cocktail-1shot": "Barman-1cocktail-1shot",
        "blocks_3": "Blocks3ops",
        "blocks_3_v2": "Blocks3ops",
        "blocks_4_v2": "Blocks4ops",
        "blocks_4_clear_no_constants": "Blocks4ops-clear",
        "blocks_4_on_no_constants": "Blocks4ops-on",
        "childsnack": "Childsnack",
        "childsnack_untyped": "Childsnack",
        "delivery": "Delivery",
        "delivery_untyped": "Delivery",
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
        "logistics2-indexicals": "Logistics-indexicals",
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

    blacklist: Set[str] = {
        "8puzzle-1tile-v2-X",
    }

    # Output
    for domain in sorted(outputs.keys()):
        datas = outputs[domain]
        if len(datas) > 0 and domain not in blacklist:
            num_instances = len(datas)
            coverage: float = sum([data["valid"] for data in datas]) / num_instances
            max_eff_width: float = max([data["max_eff_width"] for data in datas])
            avg_eff_width: float = max([data["avg_eff_width"] for data in datas])
            domain_str = f"\\Verb!{translation.get(domain, domain)}!"
            coverage_str = f"{coverage*100:6,.1f}\\%"
            coverage_str = coverage_str if coverage == 1.0 else f"\\cellcolor{{blue!25}}{coverage_str}"
            max_eff_width_str = f"{max_eff_width:4,.2f}"
            max_eff_width_str = max_eff_width_str if max_eff_width == 0.0 else f"\\cellcolor{{blue!25}}{max_eff_width_str}"
            avg_eff_width_str = f"{avg_eff_width:4,.2f}"
            avg_eff_width_str = avg_eff_width_str if avg_eff_width == 0.0 else f"\\cellcolor{{blue!25}}{avg_eff_width_str}"
            #print(f"{domain_str:40} & {num_instances:4} & {coverage*100:6,.1f} & {max_eff_width:4,.2f} & {avg_eff_width:4,.2f} \\\\")
            print(f"{domain_str:40} & {num_instances:4} & {coverage_str} & {max_eff_width_str} & {avg_eff_width_str} \\\\")



 
