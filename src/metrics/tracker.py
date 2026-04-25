import pandas as pd


class MetricTracker:
    """
    Class to aggregate metrics from many batches.
    """

    def __init__(self, metrics, writer=None):
        """
        Args:
            *keys (list[str]): list (as positional arguments) of metric
                names (may include the names of losses)
            writer (WandBWriter | CometMLWriter | None): experiment tracker.
                Not used in this code version. Can be used to log metrics
                from each batch.
        """
        self.writer = writer
        self.metrics = metrics
        self.reset()

    def reset(self):
        """
        Reset all metrics after epoch end.
        """
        if self.metrics is None:
            return
        for met in self.metrics:
            met.reset()

    def update(self, batch):
        """
        Update metrics DataFrame with new value.

        Args:
            key (str): metric name.
            value (float): metric value on the batch.
            n (int): how many times to count this value.
        """
        if self.metrics is None:
            return
        for met in self.metrics:
            met.update(**batch)

    def result(self):
        """
        Return average value of each metric.

        Returns:
            average_metrics (dict): dict, containing average metrics
                for each metric name.
        """
        if self.metrics is None:
            return {}
        return {met.name: met.compute() for met in self.metrics}
