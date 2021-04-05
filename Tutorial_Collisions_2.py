import os
import matplotlib.pyplot as plt

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad_dc.collision.visualization.draw_dispatch import draw_object

# load the exemplary CommonRoad scenario using the CommonRoad file reader
scenario, planning_problem_set = CommonRoadFileReader('ZAM_Tutorial-1_2_T-1.xml').open()

# plot the scenario
plt.figure(figsize=(25, 10))
draw_object(scenario)
draw_object(planning_problem_set)
plt.autoscale()
plt.gca().set_aspect('equal')
plt.show()
