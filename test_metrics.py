import warnings

import hydra

from src.data_module import DiffusionTrackerDataModule, visualize_data

warnings.filterwarnings("ignore", category=UserWarning)
from mocked_model import NoiseModel
from src.metrics import ade, fde

import jax.numpy as jnp
import mediapy

def traj_to_xy_valid(traj):
    # traj.x, traj.y: (..., T)
    xy = jnp.stack([traj.x, traj.y], axis=-1)   # (..., T, 2)
    valid = getattr(traj, "valid", None)        # (..., T) or None
    return xy, valid

from src.visualization.plotting import plot_state
import src.visualization.viz as viz

@hydra.main(version_base=None, config_path="src/configs", config_name="visualize")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    data_module = DiffusionTrackerDataModule(config.datasets, config.dataloaders)
    loader = data_module.train_dataloader()
    model = NoiseModel(mean=0, std=0.5, t=1, history=1)
    a = ade.AdeMetric(name="ade")
    f = fde.FdeMetric(name="fde")
    for state, i in zip(loader, range(len(loader))): #loader:
        pred = model(state)
        pred_xy, _ = traj_to_xy_valid(pred.log_trajectory)
        gt_xy, valid = traj_to_xy_valid(state.log_trajectory)
        #state is SimulatorState here
        #TODO: Determine type for state
        img = viz.plot_simulator_state(
                state,
                batch_idx=0,
                plot_all_trajectories=True,
                pred_xy=pred_xy,   
            )
        mediapy.write_image(f"frame{i}.png", img)
        f.update(pred_xy, gt_xy, valid)
        print(f.compute())
        a.update(pred_xy, gt_xy, valid)
        print(a.compute())
        print(state.log_trajectory.x.shape)

if __name__ == "__main__":
    main()