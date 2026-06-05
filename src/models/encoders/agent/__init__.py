from src.models.encoders.agent.full import AgentEncoder
from src.models.encoders.agent.simple import SimpleAgentEncoder

_AGENT_ENCODERS = {
    "simple": SimpleAgentEncoder,
    "full": AgentEncoder,
}

_SIMPLE_KEYS = frozenset({"time_len", "num_feat", "out_dim"})
_FULL_KEYS = frozenset(
    {
        "rnn_type",
        "rnn_num_heads",
        "sa_num_heads",
        "drop_attn",
        "mlp_dim",
        "num_mlp_layers",
        "num_agents",
        "time_len",
        "num_feat",
        "out_dim",
        "pos_emb_type",
        "rope_theta",
    }
)


def build_agent_encoder(name: str, *, key, **kwargs):
    encoder_name = name.strip().lower()
    if encoder_name not in _AGENT_ENCODERS:
        raise ValueError(
            f"Unknown agent encoder {name!r}; choose from {sorted(_AGENT_ENCODERS)}"
        )
    allowed = _SIMPLE_KEYS if encoder_name == "simple" else _FULL_KEYS
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return _AGENT_ENCODERS[encoder_name](key=key, **filtered)
