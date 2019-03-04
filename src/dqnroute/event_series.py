import pandas as pd

from typing import Callable, Dict, Optional, List

Aggregator = Callable[[Optional[float], float], float]

class EventSeries:
    def __init__(self, period: int, aggregators: Dict[str, Aggregator]):
        columns = ['time'] + list(aggregators.keys())

        self.records = pd.DataFrame(columns=columns)
        self.period = period
        self.aggregators = aggregators

    def logEvent(self, time, value):
        cur_period = int(time / self.period)
        avg_time = cur_period * self.period
        data_cols = self.records.columns[1:]

        try:
            data_vals = self.records.loc[cur_period, data_cols]
        except KeyError:
            data_vals = [None]*len(data_cols)

        new_data_vals = [self.aggregators[col](old_value, value)
                         for (col, old_value) in zip(data_cols, data_vals)]

        self.records.loc[cur_period] = [avg_time] + new_data_vals

    def getSeries(self):
        return self.records.sort_index()

    def reset(self):
        self.records = self.records.iloc[0:0]

def aggregator(f: Callable[[float, float], float], dv = None) -> Aggregator:
    """
    Make an aggregator from a simple function of two values
    """
    def _inner(a: Optional[float], b: float) -> float:
        if a is None:
            return dv if dv is not None else b
        return f(a, b)
    return _inner

def binary_func(name: str) -> Aggregator:
    """
    Helper for getting standard aggregator functions
    """
    if name == 'sum':
        return aggregator(lambda a, b: a + b)
    elif name == 'count':
        return aggregator(lambda a, _: a + 1, 1)
    elif name == 'max':
        return aggregator(max)
    elif name == 'min':
        return aggregator(min)
    else:
        raise Exception('Unknown aggregator function ' + name)

def event_series(period: int, aggr_names: List[str]) -> EventSeries:
    return EventSeries(period, {name: binary_func(name) for name in aggr_names})
