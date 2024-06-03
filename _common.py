
import jax.numpy as jnp
import jax

import numpy as np

from functools import partial

import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm

from trajax import integrators

from trajax.experimental.sqp import util
from importlib import reload

from trajax.optimizers import ilqr
from flax import struct
import chex

import os

Array = jax.Array
Scalar = chex.Scalar

# Do angle wrapping on theta1 and theta2
s1_ind = (0, 1)
state_wrap = util.get_s1_wrapper(s1_ind)

n = 4
m = 1

@struct.dataclass
class AcrobotDynamicsParams:
    
    LINK_MASS_1: float = 1.0
    LINK_MASS_2: float = 1.0
    LINK_LENGTH_1: float = 1.0
    LINK_COM_POS_1: float = 0.5
    LINK_COM_POS_2: float = 0.5
    LINK_MOI_1: float = 1.0
    LINK_MOI_2: float = 1.0


def acrobot(
    x: Array,
    u: Array,
    t: Scalar,
    params: AcrobotDynamicsParams
) -> Array:
    """Classic Acrobot system.

    Note this implementation emulates the OpenAI gym implementation of
    Acrobot-v2, which itself is based on Stutton's Reinforcement Learning book.

    https://gymnasium.farama.org/environments/classic_control/acrobot/

    Args:
      x: state, (4, ) array
      u: control, (1, ) array
      t: scalar time. Disregarded because system is time-invariant.
      params: tuple of (LINK_MASS_1, LINK_MASS_2, LINK_LENGTH_1, LINK_COM_POS_1,
        LINK_COM_POS_2 LINK_MOI_1, LINK_MOI_2)

    Returns:
      xdot: state time derivative, (4, )
    """
    del t  # Unused

    m1, m2, l1, lc1, lc2, I1, I2 = jax.flatten_util.ravel_pytree(params)[0]
    g = 9.8
    a = u[0] #+ 10*(np.random.rand()-0.5)#jax.random.uniform(jax.random.PRNGKey(0),minval=-5,maxval=5,shape=(1,1))[0]
    theta1 = x[0]
    theta2 = x[1]
    dtheta1 = x[2]
    dtheta2 = x[3]
    d1 = (
        m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * jnp.cos(theta2)) + I1 +
        I2)
    d2 = m2 * (lc2**2 + l1 * lc2 * jnp.cos(theta2)) + I2
    phi2 = m2 * lc2 * g * jnp.cos(theta1 + theta2 - jnp.pi / 2.)
    phi1 = (-m2 * l1 * lc2 * dtheta2**2 * jnp.sin(theta2) -
            2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * jnp.sin(theta2) +
            (m1 * lc1 + m2 * l1) * g * jnp.cos(theta1 - jnp.pi / 2) + phi2)
    ddtheta2 = ((a + d2 / d1 * phi1 -
                 m2 * l1 * lc2 * dtheta1**2 * jnp.sin(theta2) - phi2) /
                (m2 * lc2**2 + I2 - d2**2 / d1))
    ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
    return jnp.array([dtheta1, dtheta2, ddtheta1, ddtheta2])


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import matplotlib.animation as animation

def animate_acrobot(x, t, params: AcrobotDynamicsParams, filename='test_scp', dt=0.1):
    m1, m2, l1, lc1, lc2, I1, I2 = jax.flatten_util.ravel_pytree(params)[0]
    l2 = l1
    
    ## Convert Theta to Position
    pos_x_elbow = l1*jnp.sin(x[:,0]) 
    pos_y_elbow = -l1*jnp.cos(x[:,0]) 
    
    pos_x_end = pos_x_elbow+l2*jnp.sin(x[:,0]+x[:,1]) 
    pos_y_end = pos_y_elbow-l2*jnp.cos(x[:,0]+x[:,1]) 

    # Figure and axis
    fig, ax = plt.subplots(dpi=100)
    x_min, x_max, y_min, y_max = -1.1 * (l1+l2), 1.1 * (l1+l2), -1.1 * (l1+l2), 1.1 * (l1+l2)
    ax.plot(0.0, 0.0, "X", linewidth=0.1, color="k")#[0]
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_yticks([])
    ax.set_aspect(1.0)
    
    # Artists
    link1 = ax.plot([], [], "-", linewidth=3, color="b")[0]
    link2 = ax.plot([], [], "-", linewidth=3, color="g")[0]
    elbow = ax.plot([], [], "ro", linewidth=3)[0]
    trace = ax.plot([], [], "--", linewidth=2, color="tab:orange")[0]
    timestamp = ax.text(0.1, 0.9, "", transform=ax.transAxes)
    
    def animate(k, t):
        # Geometry
        link1.set_data([0, pos_x_elbow[k]], [0, pos_y_elbow[k]])
        link2.set_data([pos_x_elbow[k], pos_x_end[k]], [pos_y_elbow[k], pos_y_end[k]])
        elbow.set_data([pos_x_elbow[k], pos_x_elbow[k]], [pos_y_elbow[k],pos_y_elbow[k]])
        trace.set_data(pos_x_end[:k], pos_y_end[:k])
    
        # Time-stamp
        timestamp.set_text("t = {:.1f} s".format(t[k]))
    
        artists = (link1, link2, elbow, trace, timestamp)
        return artists
    
    ani = animation.FuncAnimation(
        fig, animate, t.size, fargs=(t,), interval=dt * 1000, blit=True
    )

    # filename = 'test_scp_fine'

    ani.save(f"{filename}.mp4", writer="ffmpeg")
    fig.savefig(f'{filename}.png') 
            

def validate_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name