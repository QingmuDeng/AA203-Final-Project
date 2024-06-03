from _common import *

@struct.dataclass
class AcrobotCostParams:
    stage_cost_x: float = 0.1
    stage_cost_u: float = 0.01
    term_cost_x: float = 1000.0


def acrobot_cost_mpc(
    x: Array,
    u: Array,
    t: Scalar,
    # T: Scalar,
    params: AcrobotCostParams
) -> float:
    delta = x - goal
    terminal_cost = 0.5 * params.term_cost_x * jnp.dot(delta, delta)
    stage_cost = 0.5 * params.stage_cost_x * jnp.dot(
        delta, delta) + 0.5 * params.stage_cost_u * jnp.dot(u, u)
    return jnp.where(t == N_mpc, terminal_cost, stage_cost) # In MPC, the terminal cost should be the end of the horizon instead of end of simulation time


@jax.jit
def acrobot_solve(
    dynamics_params: AcrobotDynamicsParams, 
    cost_params: AcrobotCostParams,
    x0,
    U,
    lqr_inter
) -> float:
    dynamics = integrators.euler(
        partial(acrobot, params=dynamics_params), dt=0.1)
    return ilqr(
        partial(acrobot_cost_mpc, params=cost_params,),
        dynamics, x0, U,
        maxiter=lqr_inter, make_psd=False, vjp_method='explicit')


goal = jnp.array([jnp.pi, 0.0, 0.0, 0.0])

dynamics_params = AcrobotDynamicsParams()
cost_params = AcrobotCostParams()


N_mpc = 50  # MPC horizon; horizon length is related to how fast the system is going to converge;for some disturbances we should have a fast convergence and short horizon
N_ilqr = 20  # maximum number of LQR iterations

dynamics_params = AcrobotDynamicsParams()
cost_params = AcrobotCostParams()

myT = 60  # total simulation time step
dt = 0.1
t = np.arange(0.0, (myT)*dt, dt)
x_mpc = np.zeros((myT , N_mpc+1, n))
u_mpc = np.zeros((myT , N_mpc, m))


########### Solve the MPC Problem ###########
x0 = jnp.zeros(4)
U = jnp.zeros((N_mpc, 1))

x0_rand = jnp.zeros(4)
U_rand = jnp.zeros((N_mpc, 1))

np.random.seed(1)

dynamics = integrators.euler(partial(acrobot, params=dynamics_params), dt=0.1)
for k in tqdm(range(myT)):
    
    
    soln = acrobot_solve(dynamics_params, cost_params, x0, U, N_ilqr)

    x_mpc[k,:,:] = soln[0]
    u_mpc[k,:,:] = soln[1]


    # ax[i].plot(x_mpc[6,:,i]
    x0 = dynamics(x_mpc[k,0,:], u_mpc[k,0,:], 0) #+ (np.random.rand(1)-0.5)*5
    # x0 += (np.random.rand(4)-0.5)*0.0005
    # print(x0)#+ (np.random.rand(1,4)-0.5)*0.001
    U = np.concatenate([soln[1][1:], np.zeros((1,1))])

########### Solve the MPC Problem ###########



filename = 'trajax_ilqr_mpc'
outpath = f'./{filename}/'
output = validate_dir(outpath)
animate_acrobot(x_mpc[:,0,:], np.arange(0, myT+1)*0.1, dynamics_params,filename=outpath+f'{filename}')


# Plot and Save State and Control Traj
fig, ax = plt.subplots(n//2, n//2, figsize=(15, 6))


colors = plt.cm.jet(np.linspace(0,1,len(t)))

for i in range(n):
    for k, t_k in enumerate(t):
        t_series = np.linspace(t_k, t_k+(N_mpc)*dt, num=N_mpc+1)#np.arange(t_k, t_k+(N_mpc+1)*dt, dt)[:11]
        # print(t_series , len(t_series))
        ax[i//2,i%2].plot(t_series, x_mpc[k,:,i],':', color=colors[k],alpha=0.5)

    ax[i//2,i%2].plot(t,x_mpc[:,0,i],'k.-')
fig.savefig(outpath+f'{filename}_state_traj.png')


fig, ax = plt.subplots( figsize=(15, 6))

for k, t_k in enumerate(t):
    t_series = np.linspace(t_k, t_k+(N_mpc)*dt, num=N_mpc)
    ax.plot(t_series, u_mpc[k,:],':', color=colors[k], alpha=0.5)
ax.plot(t,u_mpc[:,0],'k*')
ax.grid()
fig.savefig(outpath+f'{filename}_control_traj.png')

plt.close()
