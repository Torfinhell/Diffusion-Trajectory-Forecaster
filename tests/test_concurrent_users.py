"""
Simulate N concurrent users hitting Predict and Create GIF.

Run:
    uv run python tests/test_concurrent_users.py
    uv run python tests/test_concurrent_users.py --users 5 --no-gif
"""
import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import (
    InferenceQueue,
    create_prediction_gif,
    load_cfg,
    load_model,
    load_scenarios,
    batch_transform_trajs_to_global_frame,
    render_scenario,
)
import numpy as np

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
N_SCENARIOS = 10
DATASET_CONFIG = "small_no_scenes"
CHECKPOINT_PATH = "checkpoints/best45/best.eqx"


def setup():
    cfg = load_cfg(DATASET_CONFIG)
    scenarios = load_scenarios(cfg, N_SCENARIOS)
    model, diffusion_sampler, status = load_model(cfg, CHECKPOINT_PATH)
    print(f"Model: {status}")
    pred_cache = {}
    infer_queue = InferenceQueue(model, diffusion_sampler, scenarios, pred_cache)
    return scenarios, pred_cache, infer_queue


# ---------------------------------------------------------------------------
# Single user simulation
# ---------------------------------------------------------------------------
def user_session(user_id, scenario_idx, scenarios, pred_cache, infer_queue,
                 do_gif, mpl_lock, results):
    t0 = time.monotonic()
    timings = {}

    # --- Predict ---
    neighbour_indices = [
        i for i in [scenario_idx - 1, scenario_idx, scenario_idx + 1]
        if 0 <= i < len(scenarios)
    ]
    fut = infer_queue.submit(neighbour_indices)
    fut.result()
    timings["predict"] = time.monotonic() - t0

    # --- GIF (optional) ---
    if do_gif and scenario_idx in pred_cache:
        t1 = time.monotonic()
        entry = pred_cache[scenario_idx]
        s = scenarios[scenario_idx]
        if s.get("scenario") is not None and entry[1] is not None:
            gif_path = create_prediction_gif(s, np.asarray(entry[1]), mpl_lock=mpl_lock)
            timings["gif"] = time.monotonic() - t1
            timings["gif_path"] = gif_path

    timings["total"] = time.monotonic() - t0
    results[user_id] = timings
    print(f"  user {user_id:2d} (scenario {scenario_idx}): "
          f"predict={timings['predict']:.1f}s"
          + (f"  gif={timings.get('gif', 0):.1f}s" if do_gif else "")
          + f"  total={timings['total']:.1f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(num_users: int, do_gif: bool):
    print(f"\n{'='*60}")
    print(f"Simulating {num_users} concurrent users (gif={do_gif})")
    print(f"{'='*60}")

    scenarios, pred_cache, infer_queue = setup()
    mpl_lock = threading.Lock()
    results = {}

    # Spread users across different scenarios
    user_scenarios = [i % N_SCENARIOS for i in range(num_users)]

    threads = [
        threading.Thread(
            target=user_session,
            args=(uid, user_scenarios[uid], scenarios, pred_cache,
                  infer_queue, do_gif, mpl_lock, results),
        )
        for uid in range(num_users)
    ]

    t_wall_start = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall_time = time.monotonic() - t_wall_start

    predict_times = [r["predict"] for r in results.values()]
    gif_times = [r["gif"] for r in results.values() if "gif" in r]
    total_times = [r["total"] for r in results.values()]

    print(f"\nSummary ({num_users} users):")
    print(f"  wall time:      {wall_time:.1f}s")
    print(f"  predict  avg={sum(predict_times)/len(predict_times):.1f}s  "
          f"max={max(predict_times):.1f}s")
    if gif_times:
        print(f"  gif      avg={sum(gif_times)/len(gif_times):.1f}s  "
              f"max={max(gif_times):.1f}s")
    print(f"  total    avg={sum(total_times)/len(total_times):.1f}s  "
          f"max={max(total_times):.1f}s")
    print(f"  speedup vs serial: {sum(total_times)/wall_time:.1f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=3)
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()
    run(args.users, do_gif=not args.no_gif)
