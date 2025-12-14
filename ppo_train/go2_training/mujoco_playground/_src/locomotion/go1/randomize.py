"""Domain randomization for the Go1 environment."""
import jax
from mujoco import mjx
FLOOR_GEOM_ID = 0
TORSO_BODY_ID = 1
def domain_randomize(model: mjx.Model, rng: jax.Array):
  @jax.vmap
  def rand_dynamics(rng):
    rng, key = jax.random.split(rng)
    geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(
        jax.random.uniform(key, minval=0.4, maxval=1.0)
    )
    rng, key = jax.random.split(rng)
    frictionloss = model.dof_frictionloss[6:] * jax.random.uniform(
        key, shape=(12,), minval=0.9, maxval=1.1
    )
    dof_frictionloss = model.dof_frictionloss.at[6:].set(frictionloss)
    rng, key = jax.random.split(rng)
    armature = model.dof_armature[6:] * jax.random.uniform(
        key, shape=(12,), minval=1.0, maxval=1.05
    )
    dof_armature = model.dof_armature.at[6:].set(armature)
    rng, key = jax.random.split(rng)
    kp_scale = jax.random.uniform(
        key, shape=(12,), minval=0.95, maxval=1.05
    )
    actuator_gainprm = model.actuator_gainprm.at[:, 0].set(
        model.actuator_gainprm[:, 0] * kp_scale
    )
    actuator_biasprm = model.actuator_biasprm.at[:, 1].set(
        model.actuator_biasprm[:, 1] * kp_scale
    )
    rng, key = jax.random.split(rng)
    kd_scale = jax.random.uniform(
        key, shape=(12,), minval=0.95, maxval=1.05
    )
    dof_damping = model.dof_damping.at[6:].set(
        model.dof_damping[6:] * kd_scale
    )
    rng, key = jax.random.split(rng)
    dpos_x = jax.random.uniform(key, (), minval=-0.2, maxval=0.2)
    rng, key = jax.random.split(rng)
    dpos_yz = jax.random.uniform(key, (2,), minval=-0.2, maxval=0.2)
    dpos = jax.numpy.concatenate([jax.numpy.array([dpos_x]), dpos_yz])
    body_ipos = model.body_ipos.at[TORSO_BODY_ID].set(
        model.body_ipos[TORSO_BODY_ID] + dpos
    )
    rng, key = jax.random.split(rng)
    dmass = jax.random.uniform(
        key, shape=(model.nbody,), minval=0.9, maxval=1.1
    )
    body_mass = model.body_mass.at[:].set(model.body_mass * dmass)
    rng, key = jax.random.split(rng)
    dmass = jax.random.uniform(key, minval=-3.0, maxval=3.0)
    body_mass = body_mass.at[TORSO_BODY_ID].set(
        body_mass[TORSO_BODY_ID] + dmass
    )
    rng, key = jax.random.split(rng)
    qpos0 = model.qpos0
    qpos0 = qpos0.at[7:].set(
        qpos0[7:]
        + jax.random.uniform(key, shape=(12,), minval=-0.05, maxval=0.05)
    )
    return (
        geom_friction,
        body_ipos,
        body_mass,
        qpos0,
        dof_frictionloss,
        dof_armature,
        actuator_gainprm,
        actuator_biasprm,
        dof_damping,
    )
  (
      friction,
      body_ipos,
      body_mass,
      qpos0,
      dof_frictionloss,
      dof_armature,
      actuator_gainprm,
      actuator_biasprm,
      dof_damping,
  ) = rand_dynamics(rng)
  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({
      "geom_friction": 0,
      "body_ipos": 0,
      "body_mass": 0,
      "qpos0": 0,
      "dof_frictionloss": 0,
      "dof_armature": 0,
      "actuator_gainprm": 0,
      "actuator_biasprm": 0,
      "dof_damping": 0,
  })
  model = model.tree_replace({
      "geom_friction": friction,
      "body_ipos": body_ipos,
      "body_mass": body_mass,
      "qpos0": qpos0,
      "dof_frictionloss": dof_frictionloss,
      "dof_armature": dof_armature,
      "actuator_gainprm": actuator_gainprm,
      "actuator_biasprm": actuator_biasprm,
      "dof_damping": dof_damping,
  })
  return model, in_axes
