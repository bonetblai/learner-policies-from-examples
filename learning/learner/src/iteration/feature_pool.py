from typing import Union
from dataclasses import dataclass

import dlplan.core as dlplan_core


@dataclass
class Feature:
    """ A single feature with custom complexity. """
    _dlplan_feature: Union[dlplan_core.Boolean, dlplan_core.Numerical]
    _complexity: int
    _max_depth_complexity: int

    @property
    def dlplan_feature(self):
        return self._dlplan_feature

    @property
    def complexity(self):
        return self._complexity

    @property
    def max_depth_complexity(self):
        return self._max_depth_complexity

    def __eq__(self, other: "Feature"):
        return self.dlplan_feature == other.dlplan_feature

    def __hash__(self):
        return hash(str(self.dlplan_feature))
