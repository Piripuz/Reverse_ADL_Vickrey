import jax.numpy as jnp


def asymm_gaussian(sigma_l=0.9, sigma_r=0.2, mu=9.5):
    def left_gau(x):
        return jnp.exp(-((x - mu) ** 2) / sigma_l**2)

    def right_gau(x):
        return jnp.exp(-((x - mu) ** 2) / sigma_r**2)

    return lambda t: jnp.where(t < mu, left_gau(t), right_gau(t))


def asymm_gaussian_plateau(sigma_l=0.9, sigma_r=0.2, mu=9.5, plateau_len=3):
    def left_gau(x):
        return jnp.exp(-((x - mu) ** 2) / sigma_l**2)

    def right_gau(x):
        return jnp.exp(-((x - mu) ** 2) / sigma_r**2)

    def right_fun(t):
        return jnp.where(
            t - mu > plateau_len / 2, right_gau(t - plateau_len / 2), 1
        )

    return lambda t: jnp.where(
        t - mu < -plateau_len / 2, left_gau(t + plateau_len / 2), right_fun(t)
    )
