"""Locomotion environments available in this repository."""

import functools
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import jax
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go2 import getup as go2_getup
from mujoco_playground._src.locomotion.go2 import handstand as go2_handstand
from mujoco_playground._src.locomotion.go2 import joystick as go2_joystick
from mujoco_playground._src.locomotion.go2 import randomize as go2_randomize

_envs = {
    "Go2JoystickFlatTerrain": functools.partial(
        go2_joystick.Joystick, task="flat_terrain"
    ),
    "Go2JoystickRoughTerrain": functools.partial(
        go2_joystick.Joystick, task="rough_terrain"
    ),
    "Go2Getup": go2_getup.Getup,
    "Go2Handstand": go2_handstand.Handstand,
    "Go2Footstand": go2_handstand.Footstand,
}

_cfgs = {
    "Go2JoystickFlatTerrain": go2_joystick.default_config,
    "Go2JoystickRoughTerrain": go2_joystick.default_config,
    "Go2Getup": go2_getup.default_config,
    "Go2Handstand": go2_handstand.default_config,
    "Go2Footstand": go2_handstand.default_config,
}

_randomizer = {
    "Go2JoystickFlatTerrain": go2_randomize.domain_randomize,
    "Go2JoystickRoughTerrain": go2_randomize.domain_randomize,
    "Go2Getup": go2_randomize.domain_randomize,
    "Go2Handstand": go2_randomize.domain_randomize,
    "Go2Footstand": go2_randomize.domain_randomize,
}


def __getattr__(name):
  if name == "ALL_ENVS":
    return tuple(_envs.keys())
  raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def register_environment(
    env_name: str,
    env_class: Type[mjx_env.MjxEnv],
    cfg_class: Callable[[], config_dict.ConfigDict],
) -> None:
  _envs[env_name] = env_class
  _cfgs[env_name] = cfg_class


def get_default_config(env_name: str) -> config_dict.ConfigDict:
  if env_name not in _cfgs:
    raise ValueError(
        f"Env '{env_name}' not found in default configs. Available configs:"
        f" {list(_cfgs.keys())}"
    )
  return _cfgs[env_name]()


def load(
    env_name: str,
    config: Optional[config_dict.ConfigDict] = None,
    config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
) -> mjx_env.MjxEnv:
  if env_name not in _envs:
    raise ValueError(f"Env '{env_name}' not found. Available envs: {_cfgs.keys()}")
  config = config or get_default_config(env_name)
  return _envs[env_name](config=config, config_overrides=config_overrides)


def get_domain_randomizer(
    env_name: str,
) -> Optional[Callable[[mjx.Model, jax.Array], Tuple[mjx.Model, mjx.Model]]]:
  if env_name not in _randomizer:
    print(
        f"Env '{env_name}' does not have a domain randomizer in the locomotion"
        " registry."
    )
    return None
  return _randomizer[env_name]
