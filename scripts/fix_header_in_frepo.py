from sys import argv
from pathlib import Path
from shutil import copy as fcopy

if __name__ == "__main__":
    filepath: Path = Path(argv[1])
    family: str = filepath.parent.name
    with filepath.open("r") as fd:
        for line in fd:
            line = line.strip("\n")
            if "UU" in line:
                uuid: str = line.split(" ")[-1]
            elif "Total concept elements" in line:
                num_concepts: int = int(line.split(" ")[-1])
            elif "Total role elements" in line:
                num_roles: int = int(line.split(" ")[-1])
            elif "Iteration:" in line:
                break

    #addendum = {"concepts": num_concepts, "roles": num_roles, "uuid": uuid, "family": family}

    repo: Path = filepath.parent / f"workspace/feature_repositories/repo_{uuid}.frepo"
    with repo.open("r") as fd:
        parameters = eval(fd.readline().strip("\n"))
        instance_names = eval(fd.readline().strip("\n"))
        statistics = eval(fd.readline().strip("\n"))
        features = [line.strip("\n") for line in fd.readlines()]

    addendum = dict()
    if "concepts" not in statistics:
        addendum["concepts"] = num_concepts
    if "roles" not in statistics:
        addendum["roles"] = num_roles
    if "uuid" not in statistics:
        addendum["uuid"] = uuid
    if "family" not in statistics:
        addendum["family"] = family

    if len(addendum) > 0:
        statistics.update(addendum)
        tmp_repo: Path = repo.with_suffix(".tmp")
        with tmp_repo.open("w") as fd:
            fd.write(f"{str(parameters)}\n")
            fd.write(f"{str(instance_names)}\n")
            fd.write(f"{str(statistics)}\n")
            for feature in features:
                fd.write(f"{feature}\n")

        fcopy(repo, repo.with_suffix(".bak"))
        tmp_repo.rename(repo)
        print(repo)

