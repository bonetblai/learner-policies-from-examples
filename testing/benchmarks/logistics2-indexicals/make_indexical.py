import sys
from pathlib import Path
from typing import List

if __name__ == "__main__":
    assert len(sys.argv) > 1
    filepath: Path = Path(sys.argv[1])

    lines: List[str] = []
    objects: List[str] = None
    with filepath.open("r") as fd:
        for line in fd.readlines():
            line = line.strip("\n")
            if line.startswith("          p0"):
                objects: List[str] = line.strip(" ").split(" ")
            elif line.startswith("(:init"):
                assert objects is not None
                line += " (no-marks) " + " ".join([f"(markable {obj})" for obj in objects])
            lines.append(line)

    for line in lines:
        print(line)

