import matplotlib.pyplot as plt
import jax.numpy as jnp

def plot_trajectory(pred_xy, gt_xy):

    pred_xy = jnp.asarray(pred_xy)[0]  # take first sample
    gt_xy = jnp.asarray(gt_xy)[0]

    fig = plt.figure()

    plt.plot(gt_xy[:,0], gt_xy[:,1], label="GT")
    plt.plot(pred_xy[:,0], pred_xy[:,1], label="Pred")

    plt.scatter(gt_xy[0,0], gt_xy[0,1])
    plt.legend()
    plt.axis("equal")
    #plt.title(f"step {self.global_step_}")

    plt.tight_layout()

    return fig