"""Registry for all environments."""
from typing import Any, Callable, Dict, Optional, Tuple, Union
import jax
import ml_collections
from mujoco import mjx
from mujoco_playground._src import dm_control_suite
from mujoco_playground._src import locomotion
from mujoco_playground._src import manipulation
from mujoco_playground._src import mjx_env
DomainRandomizer = Optional[
    Callable[[mjx.Model, jax.Array], Tuple[mjx.Model, mjx.Model]]
]
ALL_ENVS = (
    dm_control_suite.ALL_ENVS
    + locomotion.ALL_ENVS
    + manipulation.ALL_ENVS
)
def get_default_config(env_name: str):
  if env_name in manipulation.ALL_ENVS:
    return manipulation.get_default_config(env_name)
  elif env_name in locomotion.ALL_ENVS:
    return locomotion.get_default_config(env_name)
  elif env_name in dm_control_suite.ALL_ENVS:
    return dm_control_suite.get_default_config(env_name)
  raise ValueError(f"Env '{env_name}' not found in default configs.")
def load(
    env_name: str,
    config: Optional[ml_collections.ConfigDict] = None,
    config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
) -> mjx_env.MjxEnv:
  if env_name in manipulation.ALL_ENVS:
    return manipulation.load(env_name, config, config_overrides)
  elif env_name in locomotion.ALL_ENVS:
    return locomotion.load(env_name, config, config_overrides)
  elif env_name in dm_control_suite.ALL_ENVS:
    return dm_control_suite.load(env_name, config, config_overrides)
  raise ValueError(f"Env '{env_name}' not found. Available envs: {ALL_ENVS}")
def get_domain_randomizer(env_name: str) -> Optional[DomainRandomizer]:
  if env_name in manipulation.ALL_ENVS:
    return manipulation.get_domain_randomizer(env_name)
  if env_name in locomotion.ALL_ENVS:
    return locomotion.get_domain_randomizer(env_name)
  return None
