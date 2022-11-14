from PDP import PDP
from JinEnv import JinEnv
import numpy as np
import time
import scipy.io as sio
import matplotlib.pyplot as plt

# --------------------------- load environment ----------------------------------------
uav = JinEnv.Quadrotor()
Jx, Jy, Jz, mass, win_len = 1, 1, 1, 1, 0.4
uav.initDyn(Jx=Jx, Jy=Jy, Jz=Jz, mass=mass, l=win_len, c=0.01)
wr, wv, wq, ww = 1, 1, 5, 1
uav.initCost(wr=wr, wv=wv, wq=wq, ww=ww, wthrust=0.1)

# --------------------------- create PDP Control/Planning object ----------------------------------------
dt = 0.1
uavoc = PDP.ControlPlanning()
uavoc.setStateVariable(uav.X)
uavoc.setControlVariable(uav.U)
dyn = uav.X + dt * uav.f
uavoc.setDyn(dyn)
uavoc.setPathCost(uav.path_cost)
uavoc.setFinalCost(uav.final_cost)
# set the horizon and initial condition
horizon = 35 # Original repo: 35
# set initial state
ini_r_I = [5, 5, 5]
ini_v_I = [0.0, 0.0, 0.0]
ini_q = JinEnv.toQuaternion(0, [1, -1, 1])
ini_w = [0.0, 0.0, 0.0]
ini_state = ini_r_I + ini_v_I + ini_q + ini_w

# --------------------------- create PDP true oc solver ----------------------------------------
true_uavoc = PDP.OCSys()
true_uavoc.setStateVariable(uav.X)
true_uavoc.setControlVariable(uav.U)
true_uavoc.setDyn(dyn)
true_uavoc.setPathCost(uav.path_cost)
true_uavoc.setFinalCost(uav.final_cost)
true_sol = true_uavoc.ocSolver(ini_state=ini_state, horizon=horizon)
print('Optimal cost:')
print(true_sol['cost'])

# true_state_traj = true_sol['state_traj_opt']
# true_control_traj = true_sol['control_traj_opt']
# plt.figure(1)
# plt.plot(true_control_traj) # Each thruster
# plt.show()
# uav.play_animation(wing_len=1.5, state_traj=true_sol['state_traj_opt'], save_option=1, title='T2')

# ---------------------------- start learning the control policy -------------------------------------

for j in range(5): # trial loop
    start_time = time.time()
    # learning rate
    lr = 1e-4
    # initialize
    loss_trace, parameter_trace = [], []
    uavoc.init_step(horizon)
    current_parameter = np.random.randn(uavoc.n_auxvar)
    max_iter = 5000 # Original repo: 5000
    # iteration
    for k in range(int(max_iter)):
        # one iteration of PDP
        loss, dp = uavoc.step(ini_state, horizon, current_parameter)
        # update
        current_parameter -= lr * dp
        loss_trace += [loss]
        # print
        if k % 100 == 0:
            print('Trial:', j, 'Iter:', k, 'loss:', loss)

    # solve the trajectory
    sol = uavoc.integrateSys(ini_state, horizon, current_parameter)

    # learned_state = sol['state_traj']
    # learned_control = sol['control_traj']
    # plt.figure(1)
    # plt.plot(learned_control)
    # plt.show()
    uav.play_animation(wing_len=1.5, state_traj=sol['state_traj'], save_option=1, title='UAV_trial_' + str(j))

    # # save the results
    # save_data = {'trial_no': j,
    #              'parameter_trace': parameter_trace,
    #              'loss_trace': loss_trace,
    #              'learning_rate': lr,
    #              'solved_solution': sol,
    #              'true_solution': true_sol,
    #              'time_passed': time.time() - start_time,
    #              'pendulum': {'Jx': Jx,
    #                           'Jy': Jy,
    #                           'Jz': Jz,
    #                           'm': mass,
    #                           'len': win_len,
    #                           'wr': wr,
    #                           'wv': wv,
    #                           'wq': wq,
    #                           'ww': ww},
    #              'dt': dt,
    #              'horizon': horizon
    #              }
    # # # sio.savemat('./data/PDP_OC_results_trial_' + str(j) + '.mat', {'results': save_data}) # Unix format
    # path = os.getcwd() + '\\data'
    # sio.savemat(path + '\\PDP_ArthurNN_trial_' + str(j) + '.mat', {'results': save_data}) # Format for Windows