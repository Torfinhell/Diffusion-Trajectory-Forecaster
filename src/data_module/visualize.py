import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
from hydra.utils import instantiate
from jax import tree_util
from PIL import Image
from waymax import config as _config
from waymax import datatypes, dynamics
from waymax import env as _env
from waymax import visualization

from src.metrics import average_episode_summaries, summarize_episode_metrics


def grid_frame(images, cols=None):
    """images: list of HxWx3 arrays → one grid image"""
    pil = [Image.fromarray(np.asarray(im, dtype=np.uint8)) for im in images]
    w, h = pil[0].size
    n = len(pil)

    if cols is None:
        cols = int(math.ceil(math.sqrt(n)))
    rows = math.ceil(n / cols)

    canvas = Image.new("RGB", (cols * w, rows * h))
    for i, im in enumerate(pil):
        r, c = divmod(i, cols)
        canvas.paste(im, (c * w, r * h))
    return canvas


def save_grid_gif(imgs, path="batch.gif", fps=5):
    """
    imgs: list of lists
          imgs[b][t] = image
    Produces ONE GIF over time with grid per frame
    """

    B = len(imgs)
    T = len(imgs[0])

    frames = []
    for t in range(T):
        frame_imgs = [imgs[b][t] for b in range(B)]
        frames.append(grid_frame(frame_imgs))

    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
    )


def take_batch_item(state, b: int):
    return tree_util.tree_map(lambda x: x[b], state)


def rollout_one(env, ego_actor, init_state, render_every=5):
    """Roll out a single (unbatched) scenario. Returns (states, metrics_t, imgs)."""
    met_fn = jax.jit(env.metrics)
    jit_step = jax.jit(env.step)
    jit_select_action = jax.jit(ego_actor.select_action)

    state = env.reset(init_state)
    states = [state]
    metrics_t = [met_fn(state)]
    imgs = []

    t = 0
    while t < states[0].remaining_timesteps:
        current_state = states[-1]
        out = jit_select_action({}, current_state, None, None)
        next_state = jit_step(current_state, out.action)

        states.append(next_state)
        metrics_t.append(met_fn(next_state))

        # render every N steps
        if (t % render_every) == 0:
            imgs.append(
                visualization.plot_simulator_state(next_state, use_log_traj=False)
            )

        t += 1

    return states, metrics_t, imgs


def visualize_batch_rollouts(config, env, dynamics_model, dataloader, render_every=5):
    ego_actor = instantiate(
        config,
        is_controlled_func=lambda s: s.object_metadata.is_sdc,
        dynamics_model=dynamics_model,
    )

    # get one batch
    state0 = next(iter(dataloader))
    B = int(state0.timestep.shape[0])

    all_imgs = []
    all_metrics = []
    all_states = []

    for b in range(B):
        init_state = take_batch_item(state0, b)

        states, metrics_t, imgs = rollout_one(
            env=env,
            ego_actor=ego_actor,
            init_state=init_state,
            render_every=render_every,
        )

        all_states.append(states)
        all_metrics.append(metrics_t)
        all_imgs.append(imgs)

    return all_states, all_metrics, all_imgs


def visualize_data(dataloader, config, config_viz):
    dynamics_model = dynamics.StateDynamics()
    env = _env.BaseEnvironment(
        dynamics_model=dynamics_model,
        config=dataclasses.replace(
            _config.EnvironmentConfig(),
            max_num_objects=config_viz.max_num_objects,
            controlled_object=_config.ObjectType.SDC,
            sim_agents=None,
        ),
    )
    all_states, all_metrics, all_imgs = visualize_batch_rollouts(
        config, env, dynamics_model, dataloader, render_every=config_viz.render_every
    )

    episode_summaries = []
    B = len(all_states)
    for b in range(B):
        ego_mask = all_states[b][0].object_metadata.is_sdc
        episode_summaries.append(
            summarize_episode_metrics(all_metrics[b], ego_mask=ego_mask)
        )

    avg_metrics = average_episode_summaries(episode_summaries)
    for k, v in sorted(avg_metrics.items()):
        print(f"{k}: {v}")
    save_grid_gif(all_imgs, "batch_rollout.gif", fps=5)
