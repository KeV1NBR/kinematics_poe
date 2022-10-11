import rospy
from tm_msgs.srv import SetPositions

from kinematics.kinematics import *
from robot_model import tm5_model

class Tm:
    def __init__(self):
        M, Slist = tm5_model.getModel()
        self.km = Kinematics(M, Slist, "/device:GPU:0")
        self.speed = 20
        self.accel = 20
        rospy.wait_for_service('tm_driver/set_positions')
        self.client = rospy.ServiceProxy('tm_driver/set_positions', SetPositions)
    def setSpeed(self, speed):
        self.speed = speed
    def setAccel(self, accel):
        self.accel = accel
    def moveJointPtp(self, thetaList):
        srv = SetPositions()
        srv._request_class.positions = thetaList
        srv._request_class.acc_time = 10 / self.accel
        srv._request_class.velocity = self.speed / 100
        srv._request_class.motion_type = srv._request_class.PTP_J
        srv._request_class.blend_percentage = 10
        srv._request_class.fine_goal = False

        self.client(srv)
