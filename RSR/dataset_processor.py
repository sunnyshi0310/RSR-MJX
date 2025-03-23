import jax
import jax.numpy as jnp
from brax.envs.base import Env, PipelineEnv, State
from datetime import datetime
from etils import epath
import os
import numpy as np

def load_dataset_from_path(path):
    path = epath.Path(path)
    loaded_data = np.load(path, allow_pickle=True)
    print("====== dataset load from npz file ======")
    return np.array(loaded_data["states"]), np.array(loaded_data["actions"]), np.array(loaded_data["next_states"])


def evaluate_kde(data, grid, bandwidth=0.1):
      """
      Evaluate the KDE on a grid of points to approximate the distribution.
      Args:
          data: Observations (N, D).
          samples(grid): Points to estimate density at (M, D).
          bandwidth: Bandwidth of the Gaussian kernel.
      Returns:
          Normalized probabilities on the grid (G,).
      """
      samples = grid
      diffs = jnp.expand_dims(samples, axis=1) - jnp.expand_dims(data, axis=0)  # (M, N, D)
      kernel_vals = jnp.exp(-jnp.sum(diffs**2, axis=-1) / (2 * bandwidth**2))  # (M, N)
      pdf = jnp.mean(kernel_vals, axis=-1)  # Average over data points (M,)

      return pdf / jnp.sum(pdf)  # Normalize to form a valid probability distribution

def kl_divergence(p, q):
    """Compute KL divergence between two discrete distributions."""
    return jnp.sum(p * jnp.log((p + 1e-10) / (q + 1e-10)))

def wasserstein_distance(p, q):
    """Compute Wasserstein distance between two discrete distributions."""
    return jnp.sum(jnp.abs(jnp.cumsum(p) - jnp.cumsum(q)))
  
def concatenate_states(states):

    def get_obs(state: State):
        return state.obs
    
    c_states = get_obs(states[0])
    for i in range(1, len(states)):
        c_states = jnp.vstack([c_states, get_obs(states[i])])

    return c_states