"""Gradio web app for Diffusion Trajectory Forecaster."""

import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import equinox as eqx
import gradio as gr
import hydra
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from hydra.utils import instantiate

from src.data_module import AgentPath
from src.data_module.data_module import collate_fn, instantiate_dataset_split
from src.trainers.base_trainer import BaseTrainer
from src.visualization.plotting import create_prediction_gif, plot_state


def load_scenarios(cfg, split: str, n: int) -> list[dict]:
    ds = instantiate_dataset_split(cfg.dataset.data, split)
    scenarios = []
    for sample in ds:
        if len(scenarios) >= n:
            break
        scenarios.append(sample)
    print(f"Loaded {len(scenarios)} scenarios from {split}")
    return scenarios


def load_model(cfg, checkpoint_path: str):
    model = instantiate(cfg.model, key=jr.PRNGKey(0))
    ckpt = Path(checkpoint_path)
    checkpoint_status = "random weights (no checkpoint)"
    if ckpt.exists():
        try:
            model = eqx.tree_deserialise_leaves(str(ckpt), model)
            checkpoint_status = f"loaded from {ckpt}"
        except Exception as exc:
            checkpoint_status = f"load FAILED — random weights ({type(exc).__name__})"
    diffusion_sampler = instantiate(cfg.diffusion_sampler)
    return model, diffusion_sampler, checkpoint_status


def render_scenario(
    sample: dict, pred_xy_world: np.ndarray | None = None
) -> np.ndarray:
    scenario = sample.get("scenario")
    if scenario is None:
        return None
    return plot_state(
        current_state=scenario,
        log_traj=True,
        traj_preds=pred_xy_world,
        dx=75,
        tick_off=True,
        img_size=(400, 400),
    )


def run_inference_batch(
    model, diffusion_sampler, samples: list[dict], app_cfg
) -> list[np.ndarray]:
    batch = collate_fn(samples)
    keys_to_stack = [k for k in batch if k != "scenario"]
    action_len = int(app_cfg.action_len)
    extract_actions = bool(app_cfg.extract_actions)
    sample0_past = jnp.asarray(batch["agent_past"][0])
    sample0_future = jnp.asarray(batch["agent_future"][0])
    past0 = AgentPath(sample0_past[0], action_len)
    future0 = AgentPath(sample0_future[0], action_len, ref_idx=0)
    data_shape = past0.denoise_shape(extract_actions)
    key = jr.PRNGKey(int(time.time_ns() % (2**31)))
    sample_keys = jr.split(key, len(samples))
    jax_batch = {k: jnp.asarray(batch[k]) for k in keys_to_stack}

    def infer_one(sample_key, single_batch):
        past_valid = jnp.any(single_batch["agent_past"][..., :2] != 0, axis=-1)
        model_batch = {
            k: v
            for k, v in single_batch.items()
            if k not in {"agent_future", "agent_past"}
        }
        past_path = AgentPath(single_batch["agent_past"][0], action_len)
        future_path = AgentPath(single_batch["agent_future"][0], action_len, ref_idx=0)
        model_batch["past_path"] = past_path
        sampled = BaseTrainer.sample_one_sol(
            model, diffusion_sampler, data_shape, model_batch, sample_key
        )
        if extract_actions:
            return past_path.decode_action_sample(
                sampled,
                accel_scale=float(app_cfg.accel_scale),
                yaw_rate_scale=float(app_cfg.yaw_rate_scale),
            )
        return future_path.decode_xy_sample(
            sampled,
            coord_scale=float(app_cfg.coord_scale),
            past_path=past_path,
        )

    pred_xy = jax.vmap(infer_one)(sample_keys, jax_batch)
    return [np.asarray(pred_xy[i]) for i in range(len(samples))]


class InferenceQueue:
    def __init__(self, model, diffusion_sampler, scenarios, pred_cache, app_cfg):
        self.model = model
        self.diffusion_sampler = diffusion_sampler
        self.scenarios = scenarios
        self.pred_cache = pred_cache
        self.app_cfg = app_cfg
        self._q = queue.Queue()
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, indices: list[int]) -> Future:
        fut = Future()
        self._q.put((indices, fut))
        return fut

    def _loop(self):
        while True:
            requests = [self._q.get()]
            deadline = time.monotonic() + self.app_cfg.batch_timeout_ms / 1000
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    requests.append(self._q.get(timeout=remaining))
                except queue.Empty:
                    break
                if (
                    len({i for idxs, _ in requests for i in idxs})
                    >= self.app_cfg.max_batch_size
                ):
                    break

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
                    results = run_inference_batch(
                        self.model, self.diffusion_sampler, samples, self.app_cfg
                    )
                    for i, pred_xy_local in zip(all_indices, results):
                        s = self.scenarios[i]
                        if s.get("scenario") is not None:
                            past_arr = jnp.asarray(s["agent_past"])
                            past_path = AgentPath(
                                past_arr[0], int(self.app_cfg.action_len), ref_idx=-1
                            )
                            pred_xy_plot = np.asarray(
                                past_path.xy_to_global(jnp.asarray(pred_xy_local))
                            )
                        else:
                            pred_xy_plot = pred_xy_local
                        img = render_scenario(s, pred_xy_world=pred_xy_plot)
                        existing_gif = self.pred_cache.get(i, (None, None, None))[2]
                        self.pred_cache[i] = (img, pred_xy_plot, existing_gif)
                except Exception as exc:
                    for _, fut in requests:
                        if not fut.done():
                            fut.set_exception(exc)
                    continue

            for _, fut in requests:
                if not fut.done():
                    fut.set_result(None)


def build_app(
    scenarios: list[dict],
    model,
    diffusion_sampler,
    app_cfg,
    checkpoint_status: str = "",
):
    thumbnails = [render_scenario(s) for s in scenarios]
    pred_cache: dict[int, tuple] = {}
    infer_queue = InferenceQueue(
        model, diffusion_sampler, scenarios, pred_cache, app_cfg
    )
    gif_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="gif")
    matplotlib_lock = threading.Lock()

    with gr.Blocks(title="Diffusion Trajectory Forecaster") as demo:
        selected_idx = gr.Number(value=0, visible=False, precision=0)
        ckpt_note = f"  \n**Model:** {checkpoint_status}" if checkpoint_status else ""
        gr.Markdown(
            f"## Diffusion Trajectory Forecaster\n"
            f"Pick a scenario from the gallery, then click **Predict**.{ckpt_note}"
        )

        with gr.Row():
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
            with gr.Column(scale=2):
                scene_img = gr.Image(value=thumbnails[0], label="Scene", height=400)
                with gr.Row():
                    predict_btn = gr.Button("Predict", variant="primary")
                    gif_btn = gr.Button("Create GIF", variant="secondary")
                pred_img = gr.Image(label="Prediction overlay", height=400)
                gif_out = gr.Image(label="Animated prediction", height=400)

        def on_gallery_select(evt: gr.SelectData):
            idx = evt.index
            scene = render_scenario(scenarios[idx])
            if idx in pred_cache:
                cached_img, _, cached_gif = pred_cache[idx]
                return scene, f"Selected: scenario {idx}", idx, cached_img, cached_gif
            return scene, f"Selected: scenario {idx}", idx, None, None

        gallery.select(
            fn=on_gallery_select,
            outputs=[scene_img, selected_label, selected_idx, pred_img, gif_out],
        )

        def on_predict(idx):
            idx = int(idx)
            if idx in pred_cache:
                return pred_cache[idx][0]
            neighbour_indices = [
                i for i in (idx - 1, idx, idx + 1) if 0 <= i < len(scenarios)
            ]
            try:
                infer_queue.submit(neighbour_indices).result()
                _enqueue_gif_prerender(idx)
                return pred_cache[idx][0]
            except Exception as exc:
                raise gr.Error(f"Inference failed: {exc}") from exc

        def _render_gif(idx) -> str:
            entry = pred_cache[idx]
            if entry[2] is not None:
                return entry[2]
            s = scenarios[idx]
            if s.get("scenario") is None:
                raise RuntimeError("No road graph (extract_scene=true required).")
            gif_path = create_prediction_gif(
                s, np.asarray(entry[1]), mpl_lock=matplotlib_lock
            )
            pred_cache[idx] = (entry[0], entry[1], gif_path)
            return gif_path

        def _enqueue_gif_prerender(idx):
            for i in (idx - 1, idx, idx + 1):
                if (
                    0 <= i < len(scenarios)
                    and i in pred_cache
                    and pred_cache[i][2] is None
                ):
                    gif_executor.submit(_render_gif, i)

        def on_create_gif(idx):
            idx = int(idx)
            if idx not in pred_cache:
                raise gr.Error("Run Predict first.")
            if pred_cache[idx][2] is not None:
                return pred_cache[idx][2]
            try:
                return gif_executor.submit(_render_gif, idx).result()
            except Exception as exc:
                raise gr.Error(f"GIF creation failed: {exc}") from exc

        predict_btn.click(fn=on_predict, inputs=[selected_idx], outputs=[pred_img])
        gif_btn.click(fn=on_create_gif, inputs=[selected_idx], outputs=[gif_out])

    return demo


@hydra.main(version_base=None, config_name="app", config_path="src/configs")
def main(cfg) -> None:
    app_cfg = cfg.app
    scenarios = load_scenarios(cfg, app_cfg.dataset_split, int(app_cfg.num_scenarios))
    model, diffusion_sampler, ckpt_status = load_model(cfg, app_cfg.checkpoint_path)
    demo = build_app(scenarios, model, diffusion_sampler, app_cfg, ckpt_status)
    demo.queue()
    demo.launch(
        server_name=app_cfg.server_name,
        server_port=int(app_cfg.server_port),
        share=bool(app_cfg.share),
        theme=gr.themes.Base(),
    )


if __name__ == "__main__":
    main()
