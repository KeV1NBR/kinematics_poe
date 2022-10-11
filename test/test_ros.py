import sys
sys.path.append('..')


import rospy
import tf
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker

from kinematics.kinematics import *
from robot_model import tm5_model

v = [0.1, 0.1, 0.1]
Lambda = [[1, 0, 0],
          [0, 1, 0],
          [0 ,0, 1]]

def callback(data):
    global v
    global Lambda
    (v, Lambda) = km.maniEllips(data.position[:5])
    if v[0] == 0:
        v[0] = 0.01
    if v[1] == 0:
        v[1] = 0.01
    if v[2] == 0:
        v[2] = 0.01
    print(v.max()/v.min())

#print(data.position)
M, Slist = tm5_model.getModel()

km = Kinematics(M, Slist, "/device:GPU:0")

if __name__ == '__main__':
    pub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
    rospy.init_node('manipulability_ellipsoid', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    rospy.Subscriber("/joint_states", JointState, callback)

    while not rospy.is_shutdown():
        msg = Marker()
        msg.header.frame_id = "Link6"
        msg.header.stamp = rospy.Time.now()
        msg.id = Marker.ADD
        msg.type = Marker.SPHERE
        msg.scale.x = v[0] * 0.1
        msg.scale.y = v[1] * 0.1
        msg.scale.z = v[2] * 0.1


        msg.pose.orientation.w = 1
        msg.color.a = 1
        msg.color.r = 1
        msg.lifetime = rospy.Duration(10)

        pub.publish(msg)
        rate.sleep()

    rospy.spin()
