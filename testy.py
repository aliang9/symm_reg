if __name__ == "__main__":
    from typing import NamedTuple
    import jax
    import jax.numpy as jnp
    from jaxopt import GradientDescent

    # 1) Define your parameter container as a NamedTuple
    class Params(NamedTuple):
        w: jnp.ndarray
        b: jnp.ndarray

    # 2) Define a loss that accepts this pytree
    def loss(params: Params, x, y):
        preds = jnp.dot(x, params.w) + params.b
        return jnp.mean((preds - y) ** 2)

    # 3) Create some data
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (100, 10))
    y = jnp.dot(x, jnp.arange(10.0)) + 1.0

    # 4) Initialize parameters as a NamedTuple
    init_params = Params(
        w=jnp.zeros((10,)),  # weight vector
        b=jnp.zeros(()),  # scalar bias
    )

    # 5) Instantiate and run the solver
    gd = GradientDescent(fun=loss, stepsize=0.1)
    opt_params, opt_state = gd.run(init_params, x, y)

    # opt_params is again a Params instance
    print(opt_params)
    # Params(w=DeviceArray([...]), b=DeviceArray(…))
