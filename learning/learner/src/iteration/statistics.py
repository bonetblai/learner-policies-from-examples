import logging
from ..util import Timer
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


class PlainStatistics:
    def __init__(self):
        self._statistics: Dict[str, Any] = dict()

    def size(self) -> int:
        return len(self._statistics)

    def add(self, name: str, value: Any):
        self._statistics[name] = value

    def print(self, title: str = None, prefixes: Dict[str, str] = None, suffixes: Dict[str, str] = None, formatters: Dict[str, str] = None, logger: bool = False):
        if title is not None and title != "":
            if logger:
                logging.info(title)
            else:
                print(title)

        for name, value in self._statistics.items():
            prefix: str = name if prefixes is None else prefixes.get(name, name)
            suffix: str = None if suffixes is None else suffixes.get(name)
            formatter: str = "" if formatters is None else formatters.get(name, "")
            line: str = f"  {prefix}: {value:{formatter}} {'' if suffix is None else suffix}"
            if logger:
                logging.info(line)
            else:
                print(line)


class Statistics:
    def __init__(self):
        self._statistics = PlainStatistics()
        self._timers: Dict[str, Timer] = dict()
        self._formatting: Dict[str, Any] = dict()

    def size(self) -> int:
        return self._statistics.size() + len(self._timers)

    def add(self, name: str, value: Any):
        self._statistics.add(name, value)

    def add_timer(self, name, stopped: bool = True):
        if name not in self._timers:
            self._timers[name] = Timer(stopped=stopped)

    def get_timer(self, name) -> Timer:
        return self._timers.get(name)
        
    def add_timers(self, names: List[str]):
        for name in names:
            self.add_timer(name)

    def resume(self, name: str):
        timer: Timer = self._timers.get(name)
        if timer is not None:
            timer.resume()

    def stop(self, name: str):
        timer: Timer = self._timers.get(name)
        if timer is not None:
            timer.stop()

    def stop_all(self):
        for timer in self._timers.values():
            timer.stop()

    def get_elapsed_sec(self, name: str) -> float:
        timer: Timer = self._timers.get(name)
        return None if timer is None else timer.get_elapsed_sec()

    def get_elapsed_msec(self, name: str) -> float:
        timer: Timer = self._timers.get(name)
        return None if timer is None else timer.get_elapsed_msec()

    def register_formatting(self,
                            title: str = None,
                            subtitle1: str = None,
                            subtitle2: str = None,
                            prefixes: Dict[str, str] = None,
                            suffixes: Dict[str, str] = None,
                            formatters: Dict[str, str] = None):
        if title is not None: self._formatting["title"] = title
        if subtitle1 is not None: self._formatting["subtitle1"] = subtitle1
        if subtitle2 is not None: self._formatting["subtitle2"] = subtitle2
        if prefixes is not None: self._formatting["prefixes"] = prefixes
        if suffixes is not None: self._formatting["suffixes"] = suffixes
        if formatters is not None: self._formatting["formatters"] = formatters

    def print(self,
              title: str = None,
              subtitle1: str = None,
              subtitle2: str = None,
              prefixes: Dict[str, str] = None,
              suffixes: Dict[str, str] = None,
              formatters: Dict[str, str] = None,
              logger: bool = False):

        self.stop_all()
        title = title if title is not None else self._formatting.get("title")
        subtitle1 = subtitle1 if subtitle1 is not None else self._formatting.get("subtitle1")
        subtitle2 = subtitle2 if subtitle2 is not None else self._formatting.get("subtitle2")
        prefixes = prefixes if prefixes is not None else self._formatting.get("prefixes")
        suffixes = suffixes if suffixes is not None else self._formatting.get("suffixes")
        formatters = formatters if formatters is not None else self._formatting.get("formatters")

        time_stats: PlainStatistics = PlainStatistics()
        time_prefixes: Dict[str, str] = dict()
        time_suffixes: Dict[str, str] = dict()
        time_formatters: Dict[str, str] = dict()
        for name, timer in self._timers.items():
            key = f"timer/{name}"
            prefix: str = name if prefixes is None else prefixes.get(key, name)
            suffix: str = "seconds." if suffixes is None else suffixes.get(key, "seconds.")
            formatter: str = ".02f" if formatters is None else formatters.get(key, ".02f")
            value: float = timer.get_elapsed_sec()
            time_stats.add(name, value)
            time_prefixes[name] = prefix
            time_suffixes[name] = suffix
            time_formatters[name] = formatter

        if logger:
            if title is not None and title != "": logging.info(title)
        else:
            if title is not None and title != "": print(title)

        if self._statistics.size() > 0:
            self._statistics.print(title=subtitle1, prefixes=prefixes, suffixes=suffixes, formatters=formatters, logger=logger)
        if len(self._timers) > 0:
            time_stats.print(title=subtitle2, prefixes=time_prefixes, suffixes=time_suffixes, formatters=time_formatters, logger=logger)


@dataclass
class LearningStatistics:
    num_training_instances: int = 0
    num_selected_training_instances: int = 0
    num_states_in_selected_training_instances: int = 0
    num_states_in_complete_selected_training_instances: int = 0
    num_features_in_pool: int = 0

    def print(self):
        print("LearningStatistics:")
        print("    num_training_instances:", self.num_training_instances)
        print("    num_selected_training_instances (|P|):", self.num_selected_training_instances)
        print("    num_states_in_selected_training_instances (|S|):", self.num_states_in_selected_training_instances)
        print("    num_states_in_complete_selected_training_instances (|S|):", self.num_states_in_complete_selected_training_instances)
        print("    num_features_in_pool (|F|):", self.num_features_in_pool)

