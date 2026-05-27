"""
Gradio web app for Diffusion Trajectory Forecaster.

HOW IT WORKS:
- At startup we read N scenarios from the WebDataset into a plain Python list.
- The user picks a scenario from a gallery 
- Clicking "Predict" runs the diffusion model and overlays predicted trajectories.
"""

import sys
import time
import queue
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import gradio as gr
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

sys.path.insert(0, str(Path(__file__).parent))
from src.data_module.data_module import instantiate_dataset_split, collate_fn
from src.trainers.base_trainer import BaseTrainer
from src.utils.data_utils import (
    batch_transform_trajs_to_global_frame,
    predictions_to_local_xy,
)
from src.visualization.plotting import plot_state, create_prediction_gif

import equinox as eqx
import jax


#use dataset with extract_scene:true and use_full_agent_info: true. (only val split is needed)
DATASET_CONFIG = "small_no_scenes"
CHECKPOINT_PATH = "checkpoints/best45/best.eqx"
NUM_SCENARIOS = 40


def load_cfg(dataset_config: str, model_config: str = "ddpm_attn"):
    """Load Hydra config once, combining dataset and model configs."""
    config_dir = str(Path(__file__).resolve().parent / "src" / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=model_config, overrides=[f"dataset={dataset_config}"])
    return cfg


def load_scenarios(cfg, n: int) -> list[dict]:
    """Read the first n samples from the val split into a list."""
    ds = instantiate_dataset_split(cfg.dataset.data, "val")
    scenarios = []
    for sample in ds:
        if len(scenarios) >= n:
            break
        scenarios.append(sample)
    print(f"Loaded {len(scenarios)} scenarios")
    return scenarios


# Step 2: Build the model from a Hydra config + load checkpoint weights
def load_model(cfg, checkpoint_path: str):
    model = instantiate(cfg.model, key=jr.PRNGKey(0))

    ckpt = Path(checkpoint_path)
    checkpoint_status = "random weights (no checkpoint)"
    if ckpt.exists():
        try:
            model = eqx.tree_deserialise_leaves(str(ckpt), model)
            checkpoint_status = f"loaded from {ckpt}"
            print(f"Loaded checkpoint from {ckpt}")
        except Exception as e:
            checkpoint_status = f"load FAILED — using random weights ({type(e).__name__})"
            print(f"WARNING: could not load checkpoint: {e}")
    else:
        print(f"WARNING: checkpoint not found at {ckpt}, using random weights")

    diffusion_sampler = instantiate(cfg.diffusion_sampler)
    return model, diffusion_sampler, checkpoint_status



# Step 3: Render a scenario image
def render_scenario(sample: dict, pred_xy_world: np.ndarray | None = None) -> np.ndarray:
    """
    Render a top-down scene image.

    Args:
        sample: one item from our cached scenario list
        pred_xy_world: optional (A, T, 2) predicted future xy in WORLD frame

    Returns:
        RGB image as numpy array (H, W, 3)
    """
    scenario = sample.get("scenario")

    if scenario is not None:
        img = plot_state(
            current_state=scenario,
            log_traj=True,
            traj_preds=pred_xy_world,   # (A, T, 2) or None
            dx=75,
            tick_off=True,
            img_size=(400, 400),
        )
        return img
    else:
        print("WARNING: no scenario stored for this sample (extract_scene=false)")
        return None


# Step 4: Run diffusion model inference on a batch of scenarios
def run_inference_batch(model, diffusion_sampler, samples: list[dict]) -> list[np.ndarray]:
    """
    Run inference on a list of scenarios using BaseTrainer.sample_batch_sol.

    Returns:
        list of (A, T_future, 2) arrays, one per input scenario, LOCAL frame.
    """

    batch = collate_fn(samples) 
    keys_to_stack = [k for k in batch if k != "scenario"]

    B = len(samples)
    data_shape = batch["actions_future"].shape[1:]   # (A, K, 2)
    key = jr.PRNGKey(int(time.time_ns() % (2**31)))
    sample_keys = jr.split(key, B)

    jax_batch = {k: jnp.asarray(batch[k]) for k in keys_to_stack}

    sample_fn = lambda k, b: BaseTrainer.sample_one_sol(
        None, model, diffusion_sampler, data_shape, b, key=k,
    )
    x_pred_batch = jax.vmap(sample_fn)(sample_keys, jax_batch)   # (B, A, K, 2)

    results = []
    for i, sample in enumerate(samples):
        b = {k: jax_batch[k][i] for k in keys_to_stack}
        pred_xy_local, _ = predictions_to_local_xy(
            x_pred_batch[i],
            agent_past=b["agent_past"],
            origin_vel=b["origin_vel"],
            agent_future=b["agent_future"],
            actions_future=b["actions_future"],
            accel_scale=1.0,
            yaw_rate_scale=0.15,
        )
        results.append(np.asarray(pred_xy_local))
    return results   # list of (A, T_future, 2)


# Step 5: Batching inference queue
# Each Predict click submits a request  with a Future.
# A background thread waits up to BATCH_TIMEOUT_MS for more requests, then runs
# one batched forward pass and resolves all futures.
BATCH_TIMEOUT_MS = 200
MAX_BATCH_SIZE = 32


class InferenceQueue:
    def __init__(self, model, diffusion_sampler, scenarios, pred_cache):
        self.model = model
        self.diffusion_sampler = diffusion_sampler
        self.scenarios = scenarios
        self.pred_cache = pred_cache
        self._q = queue.Queue()
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def submit(self, indices: list[int]) -> Future:
        """Submit a list of scenario indices. Returns a Future resolved with None."""
        fut = Future()
        self._q.put((indices, fut))
        return fut

    def _loop(self):
        while True:
            # Block until the first request arrives.
            requests = [self._q.get()]
            deadline = time.monotonic() + BATCH_TIMEOUT_MS / 1000

            # Drain the queue until deadline or max batch size.
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    requests.append(self._q.get(timeout=remaining))
                except queue.Empty:
                    break
                # Unique indices across all requests so far.
                total = len({i for idxs, _ in requests for i in idxs})
                if total >= MAX_BATCH_SIZE:
                    break

            # Collect unique uncached indices across all requests.
            all_indices = []
            seen = set()
            for idxs, _ in requests:
                for i in idxs:
                    if i not in seen and i not in self.pred_cache:
                        seen.add(i)
                        all_indices.append(i)

            if all_indices:
                try:
                    samples = [self.scenarios[i] for i in all_indices]
                    results = run_inference_batch(self.model, self.diffusion_sampler, samples)
                    for i, pred_xy_local in zip(all_indices, results):
                        s = self.scenarios[i]
                        if s.get("scenario") is not None:
                            pred_xy_plot = np.asarray(batch_transform_trajs_to_global_frame(
                                pred_xy_local,
                                origin_xy=np.asarray(s["origin_xy"]),
                                origin_theta=np.asarray(s["origin_theta"]),
                            ))
                        else:
                            pred_xy_plot = pred_xy_local
                        img = render_scenario(s, pred_xy_world=pred_xy_plot)
                        existing_gif = self.pred_cache.get(i, (None, None, None))[2]
                        self.pred_cache[i] = (img, pred_xy_plot, existing_gif)
                except Exception as e:
                    for _, fut in requests:
                        if not fut.done():
                            fut.set_exception(e)
                    continue

            for _, fut in requests:
                if not fut.done():
                    fut.set_result(None)



# Step 6: Build the Gradio UI
def build_app(scenarios: list[dict], model, diffusion_sampler, checkpoint_status: str = "") -> gr.Blocks:
    thumbnails = [render_scenario(s) for s in scenarios]

    pred_cache: dict[int, tuple] = {}  # idx → (pred_img_arr, pred_xy, gif_path_or_None)
    infer_queue = InferenceQueue(model, diffusion_sampler, scenarios, pred_cache)

    # Thread pool for GIF rendering.
    gif_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="gif")
    # matplotlib's global figure manager is not thread-safe
    _matplotlib_lock = threading.Lock()

    with gr.Blocks(title="Diffusion Trajectory Forecaster") as demo:
        selected_idx = gr.State(value=0)
        ckpt_note = f"  \n**Model:** {checkpoint_status}" if checkpoint_status else ""
        gr.Markdown(
            f"## Diffusion Trajectory Forecaster\n"
            f"Pick a scenario from the gallery, then click **Predict**.{ckpt_note}"
        )

        with gr.Row():
            # Left: gallery of scenario thumbnails
            with gr.Column(scale=1):
                gallery = gr.Gallery(
                    value=thumbnails,
                    label="Scenarios (click to select)",
                    columns=4,
                    rows=5,
                    height=420,
                    object_fit="cover",
                )
                selected_label = gr.Textbox(
                    value="Selected: scenario 0",
                    interactive=False,
                    show_label=False,
                )

            # Right: full scene view + prediction output
            with gr.Column(scale=2):
                scene_img = gr.Image(
                    value=thumbnails[0],
                    label="Scene — log trajectory (past + ground truth)",
                    height=400,
                )
                with gr.Row():
                    predict_btn = gr.Button("Predict", variant="primary")
                    gif_btn = gr.Button("Create GIF", variant="secondary")
                pred_img = gr.Image(
                    label="Prediction overlay",
                    height=400,
                )
                gif_out = gr.Image(
                    label="Animated prediction",
                    height=400,
                )

        # Event: user selects a scenario in the gallery
        def on_gallery_select(evt: gr.SelectData):
            idx = evt.index
            scene = render_scenario(scenarios[idx])
            if idx in pred_cache:
                cached_img, _, cached_gif = pred_cache[idx]
                return scene, f"Selected: scenario {idx}", idx, cached_img, cached_gif
            return scene, f"Selected: scenario {idx}", idx, None, None

        gallery.select(
            fn=on_gallery_select,
            inputs=None,
            outputs=[scene_img, selected_label, selected_idx, pred_img, gif_out],
        )

        # Event: user clicks Predict
        def on_predict(idx):
            if idx in pred_cache:
                return pred_cache[idx][0]

            # Submit idx + neighbours to the shared inference queue.
            neighbour_indices = [
                i for i in [idx - 1, idx, idx + 1]
                if 0 <= i < len(scenarios)
            ]
            try:
                fut = infer_queue.submit(neighbour_indices)
                fut.result()   # blocks until the batch loop resolves this request
                _enqueue_gif_prerender(idx)
                return pred_cache[idx][0]
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise gr.Error(f"Inference failed: {e}")

        predict_btn.click(
            fn=on_predict,
            inputs=[selected_idx],
            outputs=[pred_img],
        )

        def _render_gif(idx) -> str:
            """Render a GIF for scenario idx and store in pred_cache"""
            entry = pred_cache[idx]
            if entry[2] is not None:
                return entry[2]
            s = scenarios[idx]
            if s.get("scenario") is None:
                raise RuntimeError("No road graph (extract_scene=true required).")
            gif_path = create_prediction_gif(s, np.asarray(entry[1]), mpl_lock=_matplotlib_lock)
            pred_cache[idx] = (entry[0], entry[1], gif_path)
            return gif_path

        # Event: user clicks Create GIF — runs in thread pool, parallel across users
        def on_create_gif(idx):
            if idx not in pred_cache:
                raise gr.Error("Run Predict first.")
            if pred_cache[idx][2] is not None:
                return pred_cache[idx][2]
            try:
                return gif_executor.submit(_render_gif, idx).result()
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise gr.Error(f"GIF creation failed: {e}")

        def _enqueue_gif_prerender(idx):
            """After a Predict, silently pre-render GIFs for neighbours in background."""
            for i in [idx - 1, idx, idx + 1]:
                if 0 <= i < len(scenarios) and i in pred_cache and pred_cache[i][2] is None:
                    gif_executor.submit(_render_gif, i)  # fire and forget

        gif_btn.click(
            fn=on_create_gif,
            inputs=[selected_idx],
            outputs=[gif_out],
        )

    return demo



# Entry point
if __name__ == "__main__":
    cfg = load_cfg(DATASET_CONFIG)

    print("Loading scenarios...")
    scenarios = load_scenarios(cfg, NUM_SCENARIOS)

    print("Loading model...")
    model, diffusion_sampler, ckpt_status = load_model(cfg, CHECKPOINT_PATH)

    print("Building app...")
    demo = build_app(scenarios, model, diffusion_sampler, ckpt_status)

    demo.queue()
    # share=False = local only. Set share=True for a public gradio tunnel URL.
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Base(),
    )
