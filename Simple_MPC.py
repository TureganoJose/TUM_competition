import os
import matplotlib.pyplot as plt
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.draw_dispatch_cr import draw_object
from vehiclemodels import parameters_vehicle3, vehicle_dynamics_ks

try:
    import matplotlib.pyplot as plt
except ImportError:
    print('Matplotlib not installed. Please use pip(3) to install required package!')

try:
    import numpy as np
except ImportError:
    print('Numpy not installed. Please use pip(3) to install required package!')

try:
    import vehiclemodels
except ImportError:
    print('commonroad-vehicle-models not installed. Please use pip install to install required package!')

try:
    import pkg_resources
    pkg_resources.require("scipy>=1.1.0")
    pkg_resources.require("cvxpy>=1.0.0")
    from cvxpy import *
except ImportError:
    print('CVXPy not installed or wrong version. Please use pip(3) to install required package!')


class TIConstraints:
    a_min = -8
    a_max = 15
    s_min = 0
    s_max = 150
    v_min = 0
    v_max = 35
    j_min = -30
    j_max = 30



# load the CommonRoad scenario that has been created in the CommonRoad tutorial
file_path = os.path.join(os.getcwd(), 'ZAM_Tutorial-1_2_T-1.xml')

scenario, planning_problem_set = CommonRoadFileReader(file_path).open()

# states
# x1 = x-position in a global coordinate system
# x2 = y-position in a global coordinate system
# x3 = steering angle of front wheels
# x4 = velocity in x-direction
# x5 = yaw angle
# u1 = steering angle velocity of front wheels
# u2 = longitudinal acceleration
# problem data
N = 40  # number of time steps
n = 5   # length of state vector
m = 2   # length of input vector
dt = scenario.dt # time step

# set up variables
x = Variable(shape=(n, N+1)) # optimization vector x contains n states per time step
u = Variable(shape=(m, N)) # optimization vector u contains 2 state

# set up constraints
c = TIConstraints()
c.a_min = -6 # Minimum feasible acceleration of vehicle
c.a_max = 6 # Maximum feasible acceleration of vehicle
c.s_min = 0 # Minimum allowed position
c.s_max = 100 # Maximum allowed position
c.v_min = 0 # Minimum allowed velocity (no driving backwards!)
c.v_max = 35 # Maximum allowed velocity (speed limit)
c.j_min = -15 # Minimum allowed jerk
c.j_max = 15 # Maximum allowed jerk

# vehicle
ego_params = parameters_vehicle3.parameters_vehicle3()

###################
# Test dynamics
###################

time_simulation = 4
N = int(time_simulation/dt)
# initial states
# u1 = steering angle velocity of front wheels
# u2 = longitudinal acceleration
u_init = np.array([0, 0])
u_input = np.zeros((N, m))
u_input[0, :] = u_init
u_input[:, 1] = 1  # constant acceleration
u_input[50:, 0] = 0.1
# states
# x1 = x-position in a global coordinate system
# x2 = y-position in a global coordinate system
# x3 = steering angle of front wheels
# x4 = velocity in x-direction
# x5 = yaw angle
x_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
x_out = np.zeros((N, n))
x_d_out = np.zeros((N, n))
x_out[0, :] = x_init
x_d_out[0, :] = x_init # discretized states

# time and distance
time = np.zeros(N)
distance = np.zeros(N)
for i in range(1, N):
    f = vehicle_dynamics_ks.vehicle_dynamics_ks(x_out[i-1, :], u_input[i, :], ego_params)
    x_out[i, :] = x_out[i-1, :] + dt * np.asarray(f)
    time[i] = time[i-1]+dt
    distance[i] = distance[i-1] + np.sqrt((x_out[i, 0]-x_out[i-1, 0])**2 + (x_out[i, 1]-x_out[i-1, 1])**2)

    [Ad, Bd, gd] = vehicle_dynamics_ks.jacobian_ks(x_d_out[i-1, :], u_input[i, :], ego_params, dt, f)
    x = np.expand_dims(x_out[i - 1, :], axis=0).T
    u = np.expand_dims(u_input[i, :], axis=0).T
    x_next = np.dot(Ad, x) + np.dot(Bd, u) + gd
    x_d_out[i, :] = x_next.T


fig, axs = plt.subplots(2, 1, figsize=(5, 5))
fig.suptitle("Car states")
axs[0].plot(x_out[:, 0], x_out[:, 1], 'b', label='dynamics')
axs0 = axs[0].twinx()
axs0.plot(x_d_out[:, 0], x_d_out[:, 1], 'r', label='Linearised-discretised vehicle')
axs[0].set_title('Car position')
axs[0].set_xlabel('x pos')
axs[0].set_ylabel('y pos', color="black", fontsize=14)
axs[0].grid(True)
axs[0].legend()
axs0.legend()
axs[1].plot(distance, x_out[:, 3], 'b')
axs1 = axs[1].twinx()
axs1.plot(distance, x_d_out[:, 3], 'r', label='Linearised-discretised vehicle')
axs[1].set_title('Car speed')
axs[1].set_xlabel('distance')
axs[1].set_ylabel('car speed', color="black", fontsize=14)
axs[1].grid(True)

plt.show()


####################
## Testing QP
####################


# Set up optimization problem
cost = 0
constr = [x[:,0] == x_0]
for k in range(N):
    # cost function
    cost += quad_form(x[:,k+1] - npy.array([0,v_ref,0,0],), Q)\
           + R * u[k] ** 2
    # single state and input constraints
    constr.append(x[:,k+1] == A*x[:,k] + B*u[k])
    # add obstacle constraint
    constr.append(x[0,k+1] <= s_max[k])
    constr.append(x[0,k+1] >= s_min[k])

