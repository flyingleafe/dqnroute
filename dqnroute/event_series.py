import pandas as pd

class EventSeries:
    def __init__(self, period):
        self.records = pd.DataFrame(columns=['time', 'count', 'avg', 'min', 'max'])
        self.period = period
        self.last_event_time = 0
        self.current_val = 0
        self.current_count = 0
        self.cur_max = 0
        self.cur_min = 1000000000000

    def _curAvg(self):
        return 0 if self.current_count == 0 else self.current_val / self.current_count

    def logEvent(self, time, value):
        last_period = round(self.last_event_time / self.period)
        cur_period = round(time / self.period)
        if cur_period == last_period:
            self.current_val += value
            self.current_count += 1
            self.cur_max = max(self.cur_max, value)
            self.cur_min = min(self.cur_min, value)
        else:
            self.records.loc[len(self.records)] = [last_period * self.period, self.current_count, self._curAvg(), self.cur_min, self.cur_max]
            for i in range(0, cur_period - last_period - 1):
                self.records.loc[len(self.records)] = [(last_period + i + 1) * self.period, 0, 0, 0, 0]
            self.current_val = value
            self.current_count = 1
            self.cur_min = value
            self.cur_max = value
        self.last_event_time = time

    def getSeries(self):
        return self.records
