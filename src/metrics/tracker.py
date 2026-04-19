class MetricTracker:
    def __init__(self, *names):
        self._names = list(names)
        self.reset()

    def add(self, name):
        if name in self._totals:
            return
        self._names.append(name)
        self._totals[name] = 0.0
        self._counts[name] = 0

    def reset(self):
        self._totals = {name: 0.0 for name in self._names}
        self._counts = {name: 0 for name in self._names}

    def update(self, name, value, n=1):
        if name not in self._totals:
            self.add(name)
        self._totals[name] += float(value) * int(n)
        self._counts[name] += int(n)

    def avg(self, name):
        count = self._counts.get(name, 0)
        if count == 0:
            return None
        return self._totals[name] / count

    def result(self):
        return {
            name: self._totals[name] / self._counts[name]
            for name in self._names
            if self._counts[name] > 0
        }
