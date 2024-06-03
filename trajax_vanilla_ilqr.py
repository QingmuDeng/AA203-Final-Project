from _common import *

@struct.dataclass
class AcrobotCostParams:
    stage_cost_x: float = 0.1
    stage_cost_u: float = 0.01
    term_cost_x: float = 1000.0


def acrobot_cost(
    x: Array,
    u: Array,
    t: Scalar,
    params: AcrobotCostParams
) -> float:
    # delta = state_wrap(x - goal)
    delta = x - goal
    terminal_cost = 0.5 * params.term_cost_x * jnp.dot(delta, delta)
    stage_cost = 0.5 * params.stage_cost_x * jnp.dot(
        delta, delta) + 0.5 * params.stage_cost_u * jnp.dot(u, u)
    return jnp.where(t == T, terminal_cost, stage_cost)


def acrobot_soln(
    dynamics_params: AcrobotDynamicsParams, 
    cost_params: AcrobotCostParams,
) -> float:
    dynamics = integrators.euler(
        partial(acrobot, params=dynamics_params), dt=0.1)
    x0 = jnp.zeros(4)#np.random.rand(4)
    U = jnp.zeros((T, 1))
    return ilqr(
        partial(acrobot_cost, params=cost_params),
        dynamics, x0, U,
        maxiter=100, make_psd=False, vjp_method='explicit')


dt = 0.1
T = 50
goal = jnp.array([jnp.pi, 0.0, 0.0, 0.0])

dynamics_params = AcrobotDynamicsParams()
cost_params = AcrobotCostParams()
soln = acrobot_soln(dynamics_params, cost_params)

filename = 'trajax_ilqr'
outpath = f'./{filename}/'
output = validate_dir(outpath)
animate_acrobot(soln[0], np.arange(0, T+1)*0.1, dynamics_params,filename=outpath+f'{filename}')


# Plot and Save State and Control Traj
fig, ax = plt.subplots(1, n, figsize=(15, 2), dpi=150)

for i in range(n):
    ax[i].plot(soln[0][:,i], alpha=2/3)
fig.savefig(outpath+f'{filename}_state_traj.png')

fig, ax = plt.subplots(1, 1, figsize=(5, 3), dpi=150)
ax.plot(soln[1])
fig.savefig(outpath+f'{filename}_control_traj.png')

plt.close()
