import equinox as eqx
import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jr


class CrossTransformerBlock(eqx.Module):
    """Post-norm block matching VBD CrossTransformer exactly."""
    attn: eqx.nn.MultiheadAttention
    ffn: eqx.nn.MLP
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    dropout_key: jax.Array

    def __init__(self, dim: int, num_heads: int, ffn_dim: int, drop_attn: float, key, kv_dim: int | None = None):
        attn_key, ffn_key, self.dropout_key = jr.split(key, 3)
        kv_dim = dim if kv_dim is None else kv_dim
        self.attn = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=dim,
            key_size=kv_dim,
            value_size=kv_dim,
            dropout_p=drop_attn,
            key=attn_key,
        )
        self.ffn = eqx.nn.MLP(
            in_size=dim, width_size=ffn_dim, depth=1, out_size=dim,
            activation=jnn.gelu, key=ffn_key,
        )
        self.norm1 = eqx.nn.LayerNorm(shape=dim)
        self.norm2 = eqx.nn.LayerNorm(shape=dim)

    def __call__(self, query, kv, attn_mask=None):
        # post-norm: add residual then normalize
        query = jax.vmap(self.norm1)(
            query + self.attn(query, kv, kv, mask=attn_mask, key=self.dropout_key)
        )
        return jax.vmap(self.norm2)(query + jax.vmap(self.ffn)(query))


class VBDDenoiser(eqx.Module):
    """
    JAX/Equinox port of VBD TransformerDecoder.

    Interface: __call__(t_noise, x_t, batch) → predicted x0, shape (a, T, 2)
      - t_noise: float in [0, 1], normalized diffusion timestep
      - x_t: (a, T, 2) noisy future xy trajectories
      - batch: dict with key "agent_past" (1, a, t_past, f) and "agents_valid" (a,)

    Internal flow (matching VBD):
      1. x_t per-token MLP projection: (a, T, 2) → (a, T, dim)
      2. Add time_embedding (T, dim) and noise_level_embedding (dim,)
      3. Two rounds of:
           - self-attn over flattened (a*T, dim)
           - cross-attn to scene context (n_ctx, kv_dim)
      4. Output MLP: (a, T, dim) → (a, T, 2)
    """
    input_proj: eqx.nn.Sequential       # (2,) → (dim,) applied per token
    time_embedding: eqx.nn.Embedding    # (T,) → (dim,)
    noise_level_embedding: eqx.nn.Embedding  # (num_steps,) → (dim,)
    agent_encoder: eqx.nn.Sequential    # per-agent past MLP: (t_past*f,) → (kv_dim,)
    sa1: CrossTransformerBlock
    ca1: CrossTransformerBlock
    sa2: CrossTransformerBlock
    ca2: CrossTransformerBlock
    out_proj: eqx.nn.Sequential
    future_len: int
    num_steps: int

    def __init__(
        self,
        future_len: int = 80,
        num_agents: int = 32,
        time_past: int = 11,
        num_feat: int = 9,
        dim: int = 256,
        ffn_dim: int = 512,
        num_heads: int = 8,
        kv_dim: int = 256,
        num_steps: int = 50,
        drop_attn: float = 0.1,
        key = None,
    ):
        keys = jr.split(key, 10)
        self.future_len = future_len
        self.num_steps = num_steps

        # per-token input projection: 2 → dim (VBD: Linear(input_dim,128) → ReLU → Linear(128,256))
        self.input_proj = eqx.nn.Sequential([
            eqx.nn.Linear(2, dim // 2, key=keys[0]),
            eqx.nn.Lambda(jnn.relu),
            eqx.nn.Linear(dim // 2, dim, key=keys[1]),
        ])

        self.time_embedding = eqx.nn.Embedding(future_len, dim, key=keys[2])
        self.noise_level_embedding = eqx.nn.Embedding(num_steps, dim, key=keys[3])

        # simple per-agent past encoder: flatten past → kv_dim
        self.agent_encoder = eqx.nn.Sequential([
            eqx.nn.Linear(time_past * num_feat, kv_dim, key=keys[4]),
            eqx.nn.Lambda(jnn.relu),
            eqx.nn.Linear(kv_dim, kv_dim, key=keys[5]),
        ])

        # two rounds of (self-attn, cross-attn)
        self.sa1 = CrossTransformerBlock(dim, num_heads, ffn_dim, drop_attn, keys[6])
        self.ca1 = CrossTransformerBlock(dim, num_heads, ffn_dim, drop_attn, keys[7], kv_dim=kv_dim)
        self.sa2 = CrossTransformerBlock(dim, num_heads, ffn_dim, drop_attn, keys[8])
        self.ca2 = CrossTransformerBlock(dim, num_heads, ffn_dim, drop_attn, keys[9], kv_dim=kv_dim)

        out_key1, out_key2 = jr.split(keys[0])  # reuse key slot, different split
        self.out_proj = eqx.nn.Sequential([
            eqx.nn.Linear(dim, dim // 2, key=out_key1),
            eqx.nn.Lambda(jnn.elu),
            eqx.nn.Linear(dim // 2, 2, key=out_key2),
        ])

    def __call__(self, t_noise, x_t, batch):
        if x_t.ndim == 4:
            x_t = x_t[0]  # (a, T, 2)

        agent_past = batch["agent_past"]
        if agent_past.ndim == 4:
            agent_past = agent_past[0]  # (a, t_past, f)
        a, T, _ = x_t.shape

        # 1. project each token
        query = jax.vmap(jax.vmap(self.input_proj))(x_t)  # (a, T, dim)

        # 2. add time embedding and noise level embedding
        time_emb = jax.vmap(self.time_embedding)(jnp.arange(T))  # (T, dim)
        t_int = jnp.round(t_noise * (self.num_steps - 1)).astype(jnp.int32)
        noise_emb = self.noise_level_embedding(t_int)   # (dim,)
        query = query + time_emb[None, :, :] + noise_emb[None, None, :]  # (a, T, dim)

        # 3. encode past context: (a, t_past*f) → (a, kv_dim)
        kv = jax.vmap(self.agent_encoder)(agent_past.reshape(a, -1))  # (a, kv_dim)

        # self-attn over time per agent (T, dim) — avoids (a*T)^2 memory
        query = jax.vmap(lambda qi: self.sa1(qi, qi))(query)  # (a, T, dim)
        query = jax.vmap(lambda qi, ki: self.ca1(qi, ki[None, :]))(query, kv)

        query = jax.vmap(lambda qi: self.sa2(qi, qi))(query)
        q_per_agent = jax.vmap(lambda qi, ki: self.ca2(qi, ki[None, :]))(query, kv)

        # 4. output projection
        return jax.vmap(jax.vmap(self.out_proj))(q_per_agent)  # (a, T, 2)
