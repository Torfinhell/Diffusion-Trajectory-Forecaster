"""
Locust load test for the Diffusion Trajectory Forecaster Gradio app.

Usage:
    # 1. Start the app first:
    uv run python app.py

    # 2. Headless, 5 users spawned at 1/s, run for 60s:
    uv run locust -f tests/locustfile.py --headless -u 5 -r 1 -t 60s --host http://localhost:7860

    # 3. Web UI at http://localhost:8089:
    uv run locust -f tests/locustfile.py --host http://localhost:7860

Each simulated user picks a random scenario, calls Predict, then occasionally GIF.
Timings are reported to Locust's stats under "Predict" and "Create GIF".
"""
import random
import time

from locust import User, task, between, events
from gradio_client import Client


class GradioUser(User):
    """Simulates one browser session against the running Gradio app."""
    wait_time = between(1, 3)
    NUM_SCENARIOS = 40

    def on_start(self):
        self._client = Client(self.host, verbose=False, httpx_kwargs={"timeout": 120})

    def _call(self, name: str, api_name: str, *args, retries: int = 2):
        """Call a Gradio endpoint and report timing + errors to Locust."""
        t0 = time.monotonic()
        exc = None
        result = None
        for attempt in range(retries + 1):
            try:
                result = self._client.predict(*args, api_name=api_name)
                exc = None
                break
            except Exception as e:
                exc = e
                if attempt < retries:
                    time.sleep(0.5)
        elapsed_ms = (time.monotonic() - t0) * 1000
        events.request.fire(
            request_type="gradio",
            name=name,
            response_time=elapsed_ms,
            response_length=0,
            exception=exc,
        )
        return result

    @task(4)
    def predict(self):
        idx = random.randint(0, self.NUM_SCENARIOS - 1)
        self._call("Predict", "/on_predict", idx)

    @task(1)
    def predict_and_gif(self):
        idx = random.randint(0, self.NUM_SCENARIOS - 1)
        result = self._call("Predict", "/on_predict", idx)
        if result is not None:
            self._call("Create GIF", "/on_create_gif", idx)
