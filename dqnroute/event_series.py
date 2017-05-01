class EventSeries:
    def __init__(self, period):
        self.period = period
        self.records = []
        self.last_event_time = 0
        self.current_val = 0
        self.current_count = 0

    def _curAvg(self):
        return 0 if self.current_count == 0 else self.current_val / self.current_count

    def logEvent(self, time, value):
        last_period = round(self.last_event_time / self.period)
        cur_period = round(time / self.period)
        if cur_period == last_period:
            self.current_val += value
            self.current_count += 1
        else:
            self.records.append(self._curAvg())
            for i in range(0, cur_period - last_period - 1):
                self.records.append(0)
            self.current_val = value
            self.current_count = 1
        self.last_event_time = time

    def getSeries(self):
        return self.records
