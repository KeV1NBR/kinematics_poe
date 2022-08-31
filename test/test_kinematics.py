import sys
sys.path.append('..')


import numpy as np
import tensorflow as tf
import modern_robotics as mr

from kinematics.kinematics import *
from robot_model import tm5_model
M, Slist = tm5_model.getModel()

km = Kinematics(M, Slist, "/device:GPU:0")

thetalist = tf.constant([np.pi, 0, np.pi/2, 0, np.pi/2, 0], dtype = tf.float16)
initial = tf.constant([np.pi, 0., 0., 0., 0., 0.], dtype = tf.float16)

pos = km.forward(thetalist)
print(pos)

t, e = mr.IKinSpace(Slist, M, pos, initial, 0.001,0.001)
print(t)
print (km.forward(t))

#km.maniEllips(thetalist)
