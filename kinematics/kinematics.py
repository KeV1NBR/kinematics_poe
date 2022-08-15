import numpy as np
import modern_robotics as mr
import torch
from . import lie_algebra


class Kinematics:

    def __init__(self, M, Slist, device='cpu', eomg=1e-3, ev=1e-3):
        self.eomg = eomg
        self.ev = ev

        self.device = torch.device(device)
        self.M = torch.tensor(M, dtype=torch.float, device=self.device)
        self.Slist = torch.tensor(Slist, dtype=torch.float, device=self.device)

    def forward(self, thetalist):
        thetalist = torch.as_tensor(thetalist,
                                    device=self.device,
                                    dtype=torch.float)
        is_batch = not (len(thetalist.shape) == 1)
        M = self.M
        Slist = self.Slist
        T = M.clone()
        n = len(thetalist)
        if is_batch:
            T = torch.repeat_interleave(T.unsqueeze(dim=0),
                                        len(thetalist),
                                        dim=0)
            n = thetalist.shape[1]
        for i in range(n - 1, -1, -1):
            if is_batch:
                theta_for_joint = thetalist[:, i:i + 1]
                Slist_for_joint = Slist[:, i].unsqueeze(dim=0)
            else:
                theta_for_joint = thetalist[i]
                Slist_for_joint = Slist[:, i]
            s_theta = Slist_for_joint * theta_for_joint
            se3 = lie_algebra.VecTose3(s_theta)
            exp_res = lie_algebra.MatrixExp6(se3)
            T = torch.matmul(exp_res, T)
        return T

    def inverse(self, goal, initial_guess=None):
        T = goal
        thetalist0 = initial_guess if initial_guess is not None else np.zeros(6)
        M = self.M.cpu().numpy()
        Slist = self.Slist.cpu().numpy()
        eomg = self.eomg
        ev = self.ev
        thetalist = np.array(thetalist0).copy()
        i = 0
        maxiterations = 20
        Vs, err = self.compute_Vs(M, Slist, T, eomg, ev, thetalist)
        while err and i < maxiterations:
            thetalist = thetalist + np.dot(np.linalg.pinv(mr.JacobianSpace(Slist, thetalist)), Vs)
            i = i + 1
            Vs, err = self.compute_Vs(M, Slist, T, eomg, ev, thetalist)
        return thetalist, not err

    def compute_Vs(self, M, Slist, T, eomg, ev, thetalist):
        Tsb = mr.FKinSpace(M, Slist, thetalist)
        Vs = np.dot(mr.Adjoint(Tsb), mr.se3ToVec(mr.MatrixLog6(np.dot(mr.TransInv(Tsb), T))))
        err = np.linalg.norm([Vs[0], Vs[1], Vs[2]]) > eomg or np.linalg.norm([Vs[3], Vs[4], Vs[5]]) > ev
        return Vs, err

    def maniEllips(self, thetaList):
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        SList = self.Slist.cpu().numpy()
        J = mr.JacobianSpace(SList, thetaList.cpu().numpy())

        A = np.dot(J[:3], J[:3].transpose())
        Lambda, v = np.linalg.eig(A)

        print(Lambda)
        print(v)
        return Lambda, v
