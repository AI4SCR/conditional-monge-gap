import pytest
import jax
import jax.numpy as jnp

from cmonge.models.nn import ConditionalPerturbationNetwork

# (context_bonds, dim_cond, num_contexts)
CONTEXT_BOND_CONFIGS = [
    pytest.param(
        ((0, 10), (10, 20)),
        20,
        2,
        id="non_overlapping_2_modalities",
    ),
    pytest.param(
        ((0, 10), (0, 10)),
        10,
        2,
        id="overlapping_2_modalities",
    ),
    pytest.param(
        ((0, 10), (10, 20), (20, 30)),
        30,
        3,
        id="non_overlapping_3_modalities",
    ),
]

DIM_DATA = 16
DIM_HIDDEN = [32, 32]
DIM_COND_MAP = (8,)
BATCH_SIZE = 4


def _make_model(context_bonds, attention_pooling, dropout_rate=0.1):
    return ConditionalPerturbationNetwork(
        dim_hidden=DIM_HIDDEN,
        dim_data=DIM_DATA,
        dim_cond=max(stop for _, stop in context_bonds),
        dim_cond_map=DIM_COND_MAP,
        embed_cond_equal=True,
        attention_pooling=attention_pooling,
        num_heads=4,
        dropout_rate=dropout_rate,
        context_entity_bonds=context_bonds,
    )


def _make_inputs(rng, dim_cond):
    rng_x, rng_c = jax.random.split(rng)
    x = jax.random.normal(rng_x, (BATCH_SIZE, DIM_DATA))
    c = jax.random.normal(rng_c, (BATCH_SIZE, dim_cond))
    return x, c


class TestAttentionPooling:
    """Tests for attention pooling in ConditionalPerturbationNetwork."""

    @pytest.mark.parametrize(
        "context_bonds,dim_cond,num_contexts", CONTEXT_BOND_CONFIGS
    )
    def test_attention_pooling_forward_pass(
        self, context_bonds, dim_cond, num_contexts
    ):
        """Test that attention pooling produces correct output shape."""
        model = _make_model(context_bonds, attention_pooling=True)
        rng = jax.random.PRNGKey(0)
        x, c = _make_inputs(rng, dim_cond)

        rng_params, rng_dropout = jax.random.split(rng)
        params = model.init({"params": rng_params, "dropout": rng_dropout}, x=x, c=c)[
            "params"
        ]

        out = model.apply({"params": params}, x, c, num_contexts)
        assert out.shape == (BATCH_SIZE, DIM_DATA)
        assert not jnp.allclose(out, 0.0)

    @pytest.mark.parametrize(
        "context_bonds,dim_cond,num_contexts", CONTEXT_BOND_CONFIGS
    )
    def test_both_pooling_modes_same_output_shape(
        self, context_bonds, dim_cond, num_contexts
    ):
        """Test that mean and attention pooling produce the same output shape."""
        rng = jax.random.PRNGKey(42)
        x, c = _make_inputs(rng, dim_cond)

        model_mean = _make_model(context_bonds, attention_pooling=False)
        rng_p1, rng_d1, rng_p2, rng_d2 = jax.random.split(rng, 4)
        params_mean = model_mean.init({"params": rng_p1, "dropout": rng_d1}, x=x, c=c)[
            "params"
        ]
        out_mean = model_mean.apply({"params": params_mean}, x, c, num_contexts)

        model_attn = _make_model(context_bonds, attention_pooling=True)
        params_attn = model_attn.init({"params": rng_p2, "dropout": rng_d2}, x=x, c=c)[
            "params"
        ]
        out_attn = model_attn.apply({"params": params_attn}, x, c, num_contexts)

        assert out_mean.shape == out_attn.shape == (BATCH_SIZE, DIM_DATA)

    @pytest.mark.parametrize(
        "context_bonds,dim_cond,num_contexts", CONTEXT_BOND_CONFIGS
    )
    def test_dropout_deterministic_vs_stochastic(
        self, context_bonds, dim_cond, num_contexts
    ):
        """Test that deterministic=False produces different outputs across runs
        while deterministic=True is consistent."""
        model = _make_model(context_bonds, attention_pooling=True, dropout_rate=0.5)
        rng = jax.random.PRNGKey(7)
        x, c = _make_inputs(rng, dim_cond)

        rng_params, rng_dropout = jax.random.split(rng)
        params = model.init({"params": rng_params, "dropout": rng_dropout}, x=x, c=c)[
            "params"
        ]

        # Deterministic mode: two calls should be identical
        out_eval_1 = model.apply(
            {"params": params}, x, c, num_contexts, deterministic=True
        )
        out_eval_2 = model.apply(
            {"params": params}, x, c, num_contexts, deterministic=True
        )
        assert jnp.allclose(out_eval_1, out_eval_2)

        # Stochastic mode: two calls with different dropout keys should differ
        key1, key2 = jax.random.split(jax.random.PRNGKey(99))
        out_train_1 = model.apply(
            {"params": params},
            x,
            c,
            num_contexts,
            deterministic=False,
            rngs={"dropout": key1},
        )
        out_train_2 = model.apply(
            {"params": params},
            x,
            c,
            num_contexts,
            deterministic=False,
            rngs={"dropout": key2},
        )
        assert not jnp.allclose(out_train_1, out_train_2)
