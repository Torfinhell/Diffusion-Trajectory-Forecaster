import equinox as eqx
import jax
import jax.random as jr
from einops import rearrange

from src.models.encoders.agent.position import RoPETimePosEmb, build_time_pos_emb
from src.models.encoders.agent.temporal import LSTMTemporal, build_temporal


class AgentEncoder(eqx.Module):
    temporal: LSTMTemporal | eqx.Module
    pos_emb: eqx.Module
    sa_agents: eqx.nn.MultiheadAttention
    mlp: eqx.nn.MLP
    dropout_key: jax.random.PRNGKey
    expected_in_dim: int

    def __init__(
        self,
        rnn_type: str,
        rnn_num_heads: int,
        sa_num_heads: int,
        drop_attn: float,
        mlp_dim: int,
        num_mlp_layers: int,
        num_agents: int,
        time_len: int,
        num_feat: int,
        out_dim: int,
        pos_emb_type: str,
        rope_theta: float,
        key,
    ):
        rnn_key, sa_key, mlp_key, self.dropout_key, embed_key = jr.split(key, 5)
        in_dim = num_agents * num_feat
        self.expected_in_dim = in_dim

        self.temporal = build_temporal(
            rnn_type,
            in_dim=in_dim,
            num_heads=rnn_num_heads,
            drop_attn=drop_attn,
            key=rnn_key,
        )
        self.pos_emb = build_time_pos_emb(
            pos_emb_type,
            time_len=time_len,
            embedding_size=in_dim,
            rope_theta=rope_theta,
            key=embed_key,
        )
        if isinstance(self.temporal, LSTMTemporal) and isinstance(
            self.pos_emb, RoPETimePosEmb
        ):
            raise ValueError(
                "RoPE positional embedding is not supported with LSTM temporal encoder"
            )

        sa_dim = time_len * num_feat
        assert (
            sa_dim % sa_num_heads == 0
        ), "time_len * num_feat must be divisible by sa_num_heads"
        self.sa_agents = eqx.nn.MultiheadAttention(
            num_heads=sa_num_heads,
            query_size=sa_dim,
            dropout_p=drop_attn,
            key=sa_key,
        )
        self.mlp = eqx.nn.MLP(
            in_size=sa_dim,
            width_size=mlp_dim,
            depth=max(num_mlp_layers - 1, 0),
            out_size=out_dim,
            key=mlp_key,
        )

    def __call__(self, x):
        if x.ndim == 3:
            x = x[None, ...]
        elif x.ndim != 4:
            raise ValueError(f"AgentEncoder expected 3D or 4D input, got {x.shape}")

        _, a, t, f = x.shape
        actual_in_dim = a * f
        if actual_in_dim != self.expected_in_dim:
            raise ValueError(
                "AgentEncoder input shape mismatch: expected A*F="
                f"{self.expected_in_dim} from model config, got A*F={actual_in_dim} "
                f"(A={a}, F={f}). Align model.num_agents/num_feat with dataset preprocessing."
            )

        x = rearrange(x, "1 a t f -> t (a f)")
        x = self.temporal(x, self.pos_emb)
        x = self.pos_emb.after_temporal(x, t)
        x = rearrange(x, "t (a f) -> a (t f)", a=a)
        x = self.sa_agents(x, x, x, key=self.dropout_key)
        return jax.vmap(self.mlp)(x).reshape(a, -1)
