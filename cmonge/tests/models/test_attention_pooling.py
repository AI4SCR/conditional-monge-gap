import jax
import jax.numpy as jnp
import optax

from cmonge.models.nn import ConditionalPerturbationNetwork


class TestAttentionPooling:
    """Tests for attention pooling in ConditionalPerturbationNetwork."""

    # Shared config for a model using embed_cond_equal (deep set path)
    DIM_DATA = 16
    DIM_COND = 20  # 2 contexts of size 10 each
    DIM_HIDDEN = [32, 32]
    DIM_COND_MAP = (8,)
    CONTEXT_BONDS = ((0, 10), (10, 20))
    BATCH_SIZE = 4
    NUM_CONTEXTS = 2

    def _make_model(self, attention_pooling: bool, dropout_rate: float = 0.1):
        return ConditionalPerturbationNetwork(
            dim_hidden=self.DIM_HIDDEN,
            dim_data=self.DIM_DATA,
            dim_cond=self.DIM_COND,
            dim_cond_map=self.DIM_COND_MAP,
            embed_cond_equal=True,
            attention_pooling=attention_pooling,
            num_heads=4,
            dropout_rate=dropout_rate,
            context_entity_bonds=self.CONTEXT_BONDS,
        )

    def _make_inputs(self, rng):
        rng_x, rng_c = jax.random.split(rng)
        x = jax.random.normal(rng_x, (self.BATCH_SIZE, self.DIM_DATA))
        c = jax.random.normal(rng_c, (self.BATCH_SIZE, self.DIM_COND))
        return x, c

    def test_attention_pooling_forward_pass(self):
        """Test that attention pooling produces correct output shape."""
        model = self._make_model(attention_pooling=True)
        rng = jax.random.PRNGKey(0)
        x, c = self._make_inputs(rng)

        rng_params, rng_dropout = jax.random.split(rng)
        params = model.init(
            {"params": rng_params, "dropout": rng_dropout}, x=x, c=c
        )["params"]

        out = model.apply({"params": params}, x, c, self.NUM_CONTEXTS)
        assert out.shape == (self.BATCH_SIZE, self.DIM_DATA)
        # Output should be a residual: x + f(x, c), so not all zeros
        assert not jnp.allclose(out, 0.0)

    def test_both_pooling_modes_same_output_shape(self):
        """Test that mean pooling and attention pooling produce the same output shape."""
        rng = jax.random.PRNGKey(42)
        x, c = self._make_inputs(rng)

        # Mean pooling (default)
        model_mean = self._make_model(attention_pooling=False)
        rng_p1, rng_d1, rng_p2, rng_d2 = jax.random.split(rng, 4)
        params_mean = model_mean.init(
            {"params": rng_p1, "dropout": rng_d1}, x=x, c=c
        )["params"]
        out_mean = model_mean.apply({"params": params_mean}, x, c, self.NUM_CONTEXTS)

        # Attention pooling
        model_attn = self._make_model(attention_pooling=True)
        params_attn = model_attn.init(
            {"params": rng_p2, "dropout": rng_d2}, x=x, c=c
        )["params"]
        out_attn = model_attn.apply({"params": params_attn}, x, c, self.NUM_CONTEXTS)

        assert out_mean.shape == out_attn.shape == (self.BATCH_SIZE, self.DIM_DATA)

    def test_dropout_deterministic_vs_stochastic(self):
        """Test that deterministic=False (training) produces different outputs across runs
        while deterministic=True (eval) is consistent."""
        model = self._make_model(attention_pooling=True, dropout_rate=0.5)
        rng = jax.random.PRNGKey(7)
        x, c = self._make_inputs(rng)

        rng_params, rng_dropout = jax.random.split(rng)
        params = model.init(
            {"params": rng_params, "dropout": rng_dropout}, x=x, c=c
        )["params"]

        # Deterministic mode: two calls should be identical
        out_eval_1 = model.apply({"params": params}, x, c, self.NUM_CONTEXTS, deterministic=True)
        out_eval_2 = model.apply({"params": params}, x, c, self.NUM_CONTEXTS, deterministic=True)
        assert jnp.allclose(out_eval_1, out_eval_2)

        # Stochastic mode: two calls with different dropout keys should differ
        key1, key2 = jax.random.split(jax.random.PRNGKey(99))
        out_train_1 = model.apply(
            {"params": params}, x, c, self.NUM_CONTEXTS,
            deterministic=False, rngs={"dropout": key1},
        )
        out_train_2 = model.apply(
            {"params": params}, x, c, self.NUM_CONTEXTS,
            deterministic=False, rngs={"dropout": key2},
        )
        assert not jnp.allclose(out_train_1, out_train_2)
