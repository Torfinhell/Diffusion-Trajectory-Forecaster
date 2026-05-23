import jax.numpy as jnp


def squeeze_batch(x, *, name: str):
    if x.ndim == 4:
        if x.shape[0] != 1:
            raise ValueError(
                f"{name} expected a single-sample batch or unbatched input, got {x.shape}"
            )
        return x[0]
    return x
