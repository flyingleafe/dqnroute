import pandas as pd

def add_prefix(pref):
    if pref == '':
        return lambda x: x
    else:
        return lambda x: pref+'_'+x

class EventSeries:
    def __init__(self, period, prefix='', func='avg'):
        value_columns = list(map(add_prefix(prefix), ['count', 'avg', 'min', 'max']))

        self.prefix = prefix
        self.records = pd.DataFrame(columns=['time']+value_columns)
        self.func = func
        self.period = period
        self.last_event_time = 0
        self.current_val = 0
        self.current_count = 0
        self.cur_max = 0
        self.cur_min = 1000000000000

    def _curAvg(self):
        if self.func == 'avg':
            return 0 if self.current_count == 0 else self.current_val / self.current_count
        elif self.func == 'sum':
            return self.current_val
        else:
            raise Exception('mda kone4no nu ty i pososal')

    def logEvent(self, time, value):
        last_period = round(self.last_event_time / self.period)
        cur_period = round(time / self.period)
        if cur_period == last_period:
            self.current_val += value
            self.current_count += 1
            self.cur_max = max(self.cur_max, value)
            self.cur_min = min(self.cur_min, value)
            self.last_event_time = time
        elif cur_period > last_period:
            self.records.loc[len(self.records)] = [last_period * self.period, self.current_count, self._curAvg(), self.cur_min, self.cur_max]
            for i in range(0, cur_period - last_period - 1):
                self.records.loc[len(self.records)] = [(last_period + i + 1) * self.period, 0, 0, 0, 0]
            self.current_val = value
            self.current_count = 1
            self.cur_min = value
            self.cur_max = value
            self.last_event_time = time
        else:
            real_cur_time = cur_period * self.period
            idxs = self.records['time'] == real_cur_time
            pr = self.prefix
            count_col, avg_col, min_col, max_col = list(map(add_prefix(pr), ['count', 'avg', 'min', 'max']))
            self.records.loc[idxs, min_col] = min(self.records.loc[idxs, min_col].iloc[0], value)
            self.records.loc[idxs, max_col] = max(self.records.loc[idxs, max_col].iloc[0], value)
            self.records.loc[idxs, count_col] += 1
            if self.func == 'avg':
                cnt = self.records.loc[idxs, count_col]
                self.records.loc[idxs, avg_col] = (self.records.loc[idxs, avg_col] * (cnt - 1) + value) / cnt
            elif self.func == 'sum':
                self.records.loc[idxs, avg_col] += value
            else:
                raise Exception('eto 4to vashe takoe')

    def getSeries(self):
        return self.records
