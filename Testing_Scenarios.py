import os
import matplotlib.pyplot as plt
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.draw_dispatch_cr import draw_object
from matplotlib import animation
import numpy as npy
from cvxpy import *
from vehiclemodels.init_ks import init_ks
from vehiclemodels.parameters_vehicle1 import parameters_vehicle1
from vehiclemodels.vehicle_dynamics_ks import vehicle_dynamics_ks, jacobian_ks

class TIConstraints:
    a_min = -8
    a_max = 15
    s_min = 0
    s_max = 150
    v_min = 0
    v_max = 35
    j_min = -30
    j_max = 30

# generate path of the file to be opened
mydir = "C:\\Users\\jmartinez\\PycharmProjects\\TUM_Competition\\commonroad-scenarios\\scenarios\\hand-crafted\\"
myfile = "USA_US101-7_2_T-1.xml"
file_path = os.path.join(mydir, myfile)


# read in the scenario and planning problem set
scenario, planning_problem_set = CommonRoadFileReader(file_path).open()

# Get all the planning problem ids
planning_problem_ids=[]
for idPlan in planning_problem_set.planning_problem_dict:
    planning_problem_ids.append(idPlan)
if len(planning_problem_ids)>1:
    f'More than one planning problems. Number of problems: { len(planning_problem_ids) }'
for problem_id in planning_problem_ids:
    planning_problem = planning_problem_set.find_planning_problem_by_id(problem_id)

# Extract final time from planning problem
t_final_problem = planning_problem.goal.state_list[0].time_step.start

def init():
    # initialize an empty list of cirlces
    return []

def animate(i):
    # draw circles, select to color for the circles based on the input argument i.
    plt.clf()
    draw_object(scenario, draw_params={'time_begin': i})
    draw_object(planning_problem_set)
    plt.gca().set_aspect('equal')

# Goal lanelet id
goal_lanelet_id = planning_problem.goal.lanelets_of_goal_position[0][0]

# Find goal center lane
for ilanelets in scenario.lanelet_network.lanelets:
    if ilanelets.lanelet_id == goal_lanelet_id:
        goal_center_vertices = ilanelets.center_vertices

# states
# x1 = x-position in a global coordinate system
# x2 = y-position in a global coordinate system
# x3 = steering angle of front wheels
# x4 = velocity in x-direction
# x5 = yaw angle

# u1 = steering angle velocity of front wheels
# u2 = longitudinal acceleration


# problem data
N  = 40  # number of time steps
n  = 5   # length of state vector
m  = 2   # length of input vector
dT = scenario.dt # time step


# set up variables
x = Variable(shape=(n,N+1)) # optimization vector x contains n states per time step
u = Variable(shape=(m,N)) # optimization vector u contains 1 state

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

# weights for cost function
w_x = 0
w_y = 0
w_v = 8
w_a = 2
w_j = 2
Q = npy.eye(n)*npy.transpose(npy.array([w_x,w_y,w_v,w_a,w_j]))
w_u = 1
R = w_u

A = npy.array([[1,dT,(dT**2)/2,(dT**3)/6],
               [0,1,dT,(dT**2)/2],
               [0,0,1,dT],
               [0,0,0,1]])
B = npy.array([(dT**4)/24,
                (dT**4)/24,
               (dT**3)/6,
               (dT**2)/2,
               dT]).reshape([n,])

initial_state = planning_problem.initial_state

# initial state of vehicle for the optimization problem (longitudinal position, velocity, acceleration, jerk)
x_0 = npy.array([initial_state.position[0],
                 initial_state.position[0],
                 initial_state.velocity,
                 0.0,
                 0.0]).reshape([n,]) # initial state


# extract obstacle from scenario
dyn_obstacles = scenario.dynamic_obstacles

# create constraints for minimum and maximum position
x_min = [] # minimum position constraint
x_max = [] # maximum position constraint
y_min = [] # minimum position constraint
y_max = [] # maximum position constraint

# go through obstacle list and distinguish between following and leading vehicle
for o in dyn_obstacles:
    if o.initial_state.position[0] < x_0[0]:
        print('Following vehicle id={}'.format(o.obstacle_id))
        prediction = o.prediction.trajectory.state_list
        for p in prediction:
            x_min.append(p.position[0]+o.obstacle_shape.length/2.+2.5)
            y_min.append(p.position[1]+o.obstacle_shape.width/2.+2.5)
    else:
        print('Leading vehicle id={}'.format(o.obstacle_id))
        prediction = o.prediction.trajectory.state_list
        for p in prediction:
            x_max.append(p.position[0]-o.obstacle_shape.length/2.-2.5)
            y_max.append(p.position[0] - o.obstacle_shape.width / 2. - 2.5)

tiConstraints = [x[1,:] <= c.v_max, x[1,:] >= c.v_min] # velocity
tiConstraints += [x[2,:] <= c.a_max, x[2,:] >= c.a_min] # acceleration
tiConstraints += [x[3,:] <= c.j_max, x[3,:] >= c.j_min] # jerk


# Set up optimization problem
states = []
cost = 0
# initial state constraint
constr = [x[:,0] == x_0]

# load vehicle parameters
p = parameters_vehicle1()

v_ref = 30

u_0 = npy.array([1, 2])

f = []
f = vehicle_dynamics_ks(x_0, u_0, p)
Ak, Bk, gk = jacobian_ks(x_0, u_0, p, dT, f)

for k in range(N):
    # cost function
    cost += quad_form(x[:,k+1] - npy.array([0,0,v_ref,0,0],), Q)\
           + R * u[k] ** 2
    # single state and input constraints
    f = vehicle_dynamics_ks.vehicle_dynamics_ks(x[:,k], u[:, k], p)
    Ak, Bk, gk = vehicle_dynamics_ks.jacobian_ks(x, u, p, dT, f)
    constr.append(x[:,k+1] == Ak*x[:,k] + Bk*u[k]+gk)
    # add obstacle constraint
    constr.append(x[0,k+1] <= x_max[k])
    constr.append(x[0,k+1] >= x_min[k])

# sums problem objectives and concatenates constraints.
prob = sum(states)
# add constraints for all states & inputs
prob = Problem(Minimize(cost), constr + tiConstraints)



# show initial objective
print('Initial SSE Objective: ' + str(objective(x0)))

# optimize
b = (1.0,5.0)
bnds = (b, b, b, b)
con1 = {'type': 'ineq', 'fun': constraint1}
con2 = {'type': 'eq', 'fun': constraint2}
cons = ([con1,con2])
solution = minimize(objective,x0,method='SLSQP',\
                    bounds=bnds,constraints=cons)
x = solution.x


# Solve optimization problem
prob.solve(verbose=True)

print("Problem is convex:",prob.is_dcp())
print("Problem solution is "+prob.status)




fig = plt.figure(figsize=(15, 10))
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=t_final_problem, interval=20, blit=False)

plt.show()