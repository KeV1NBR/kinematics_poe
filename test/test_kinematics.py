import sys
sys.path.append('..')


import numpy as np
import tensorflow as tf
import modern_robotics as mr

from kinematics.kinematics import *
from robot_model import tm5_model
kwargs = {
        'dtype': float,
    }
M, Slist = tm5_model.getModel()

km = Kinematics(M, Slist, "/device:GPU:0")

thetalist = tf.Tensor([np.pi, 0, np.pi/2, 0, np.pi/2, 0], **kwargs)
initial = tf.Tensor([np.pi, 0., 0., 0., 0., 0.], **kwargs)

pos = km.forward(thetalist)
#print(pos)

t, e = mr.IKinSpace(Slist, M, pos, initial, 0.001,0.001)
#print(t)
#print (km.forward(t))

km.maniEllips(thetalist)
