import sys
sys.path.append('..')

import rospy
import numpy as np
from control.tm_control import *


rospy.init_node('move_tm')

tm = Tm()
tm.moveJointPtp([np.pi, .0, np.pi/2, .0, np.pi/2, 0])
