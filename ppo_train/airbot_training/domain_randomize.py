from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
from brax.base import System

_FRICTION_TABLE_CUBE = (0.68, 1.32)
_MASS_CUBE = (0.84, 1.16)
_FRICTION_FINGER = (0.76, 1.24)
_JOINT_SCALE = (0.92, 1.08)

_ARM_DOF_SLICE = slice(0, 8)


def _geom_ids_for_bodies(mj_model, body_names: tuple[str, ...]) -> tuple[int, ...]:
  body_ids = {mj_model.body(name).id for name in body_names}
  return tuple(
      int(i)
      for i in range(mj_model.ngeom)
      if int(mj_model.geom_bodyid[i]) in body_ids
  )


def domain_randomize(sys: System, rng: jax.Array) -> Tuple[System, System]:

  mj_model = sys.mj_model
  table_geom_id = int(mj_model.geom('table-b').id)
  cube_geom_id = int(mj_model.geom('geom_for_push').id)
  cube_body_id = int(mj_model.body('cube_for_push').id)
  finger_geom_ids = jnp.asarray(
      _geom_ids_for_bodies(mj_model, ('left', 'right')), dtype=jnp.int32
  )

  @jax.vmap
  def rand_dynamics(rng):
    rng, key = jax.random.split(rng)
    table_scale = jax.random.uniform(
        key, minval=_FRICTION_TABLE_CUBE[0], maxval=_FRICTION_TABLE_CUBE[1]
    )
    rng, key = jax.random.split(rng)
    cube_friction_scale = jax.random.uniform(
        key, minval=_FRICTION_TABLE_CUBE[0], maxval=_FRICTION_TABLE_CUBE[1]
    )
    rng, key = jax.random.split(rng)
    cube_mass_scale = jax.random.uniform(
        key, minval=_MASS_CUBE[0], maxval=_MASS_CUBE[1]
    )
    rng, key = jax.random.split(rng)
    finger_scale = jax.random.uniform(
        key, minval=_FRICTION_FINGER[0], maxval=_FRICTION_FINGER[1]
    )
    rng, key = jax.random.split(rng)
    damping_scale = jax.random.uniform(
        key, minval=_JOINT_SCALE[0], maxval=_JOINT_SCALE[1]
    )
    rng, key = jax.random.split(rng)
    frictionloss_scale = jax.random.uniform(
        key, minval=_JOINT_SCALE[0], maxval=_JOINT_SCALE[1]
    )

    geom_friction = sys.geom_friction
    geom_friction = geom_friction.at[table_geom_id].multiply(table_scale)
    geom_friction = geom_friction.at[cube_geom_id].multiply(cube_friction_scale)
    geom_friction = geom_friction.at[finger_geom_ids].multiply(finger_scale)

    body_mass = sys.body_mass.at[cube_body_id].multiply(cube_mass_scale)
    dof_damping = sys.dof_damping.at[_ARM_DOF_SLICE].multiply(damping_scale)
    dof_frictionloss = sys.dof_frictionloss.at[_ARM_DOF_SLICE].multiply(
        frictionloss_scale
    )

    return geom_friction, body_mass, dof_damping, dof_frictionloss

  geom_friction, body_mass, dof_damping, dof_frictionloss = rand_dynamics(rng)

  in_axes = jax.tree_util.tree_map(lambda _: None, sys)
  in_axes = in_axes.tree_replace({
      'geom_friction': 0,
      'body_mass': 0,
      'dof_damping': 0,
      'dof_frictionloss': 0,
  })
  sys = sys.tree_replace({
      'geom_friction': geom_friction,
      'body_mass': body_mass,
      'dof_damping': dof_damping,
      'dof_frictionloss': dof_frictionloss,
  })
  return sys, in_axes
