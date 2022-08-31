import tensorflow as tf
import numpy as np

def VecToso3(omg):
    kwargs = {
        'dtype': omg.dtype,
    }
    if len(omg.shape) == 1:
        return tf.convert_to_tensor([[0., -omg[2], omg[1]],
                                    [omg[2], 0., -omg[0]],
                                    [-omg[1], omg[0], 0.]])
    assert len(omg.shape) == 2
    res = tf.zeros(len(omg), 3, 3, **kwargs)

    res[:, 1, 2] = -omg[:, 0]
    res[:, 2, 1] = omg[:, 0]
    res[:, 2, 0] = -omg[:, 1]
    res[:, 0, 2] = omg[:, 1]
    res[:, 0, 1] = -omg[:, 2]
    res[:, 1, 0] = omg[:, 2]
    return res


def VecTose3(V):
    kwargs = {
        'dtype': V.dtype,
    }
    if len(V.shape) == 1:
        so3_vec = VecToso3(V[:3])
        next_v = tf.expand_dims(V[3:], 1)
        omega_and_v = tf.concat((so3_vec, next_v), 1)
        row = tf.zeros((1,4), **kwargs)
        res = tf.concat((omega_and_v, row), 0)
        return res

    res = tf.zeros(len(V), 4, 4, **kwargs)
    res[:, :3, :3] = VecToso3(V[:, :3])
    res[:, :3, 3] = V[:, 3:]
    return res


def so3ToVec(so3mat):
    kwargs = {
        'dtype': so3mat.dtype,
    }

    if len(so3mat.shape) == 2:
        return tf.convert_to_tensor([so3mat[2][1],
                             so3mat[0][2],
                             so3mat[1][0]],
                            **kwargs)
    return tf.stack((so3mat[:, 2, 1],
                        so3mat[:, 0, 2],
                        so3mat[:, 1, 0]),
                       1)


def MatrixExp3(so3mat, eps=1e-6):
    omgtheta = so3ToVec(so3mat)
    kwargs = {
        'dtype': omgtheta.dtype,
    }
    if len(omgtheta.shape) == 1:
        if tf.norm(omgtheta) < eps:
            return tf.eye(3, **kwargs)

        theta = tf.norm(omgtheta)
        omgmat = so3mat / theta
        initial = tf.eye(3, **kwargs)
        sin_term = np.sin(theta) * omgmat
        cos_term = (1 - np.cos(theta)) * tf.matmul(omgmat, omgmat)
        return initial + sin_term + cos_term

    res = tf.zeros(len(so3mat), 3, 3, **kwargs)
    theta = tf.norm(omgtheta, axis=1)
    res[theta < eps] = tf.eye(3, **kwargs)
    base_mask = ~(theta < eps)
    theta_u = theta.unsqueeze(dim=-1).unsqueeze(dim=-1)
    omgmat = so3mat[base_mask] / theta_u[base_mask]
    res[base_mask] = tf.eye(3, **kwargs)
    res[base_mask] += np.sin(theta_u[base_mask]) * omgmat
    cos_term = (1 - np.cos(theta_u[base_mask]))
    omg_term = tf.matmul(omgmat, omgmat)
    res[base_mask] += cos_term * omg_term
    return res


def MatrixExp6(se3mat, eps=1e-6):
    kwargs = {
        'dtype': se3mat.dtype,
    }
    if len(se3mat.shape) == 2:
        omgtheta = so3ToVec(se3mat[:3, :3])
        if tf.norm(omgtheta) < eps:
            I = tf.eye(3, **kwargs)
            v = se3mat[:3, 3:]
            res = tf.concat((I, v), 1)
            last_row = tf.constant([[0, 0, 0, 1]], **kwargs)
            return tf.concat((res, last_row), 0)
        theta = tf.norm(omgtheta)
        omgmat = se3mat[:3, :3] / theta
        exp3_mat = MatrixExp3(se3mat[:3, :3])

        initial = tf.eye(3, **kwargs) * theta
        cos_term = (1 - np.cos(theta)) * omgmat
        omg_term = tf.matmul(omgmat, omgmat)
        sin_term = (theta - np.sin(theta)) * omg_term
        composite_term = initial + cos_term + sin_term
        v = se3mat[:3, 3]
        res = tf.linalg.matvec(composite_term, v) / theta
        res = tf.expand_dims(res, 1)
        res = tf.concat((exp3_mat, res), 1)
        last_row = tf.constant([[0, 0, 0, 1]], **kwargs)
        res = tf.concat((res, last_row), 0)
        return res

    omgtheta = so3ToVec(se3mat[:, :3, :3])

    res = tf.zeros(len(se3mat), 4, 4, **kwargs)
    theta = tf.norm(omgtheta, axis=1)

    res[theta < eps, :3, :3] = tf.eye(3, **kwargs)
    res[theta < eps, :3, 3] = se3mat[theta < eps, :3, 3]
    res[:, 3, 3] = 1.

    mask = ~(theta < eps)
    theta_u = theta[mask].unsqueeze(dim=-1).unsqueeze(dim=-1)
    omgmat = se3mat[mask, :3, :3] / theta_u
    exp3_mat = MatrixExp3(se3mat[mask, :3, :3])

    eye = tf.eye(3, **kwargs)
    eye = eye.unsqueeze(dim=0)
    initial = eye * theta_u
    cos_term = (1 - np.cos(theta_u)) * omgmat
    omg_term = tf.matmul(omgmat, omgmat)
    sin_term = (theta_u - np.sin(theta_u)) * omg_term
    composite_term = initial + cos_term + sin_term
    v = se3mat[mask, :3, 3].unsqueeze(dim=-1)
    mul_term = tf.matmul(composite_term, v) / theta_u

    res[mask, :3, :3] = exp3_mat
    res[mask, :3, 3:] = mul_term
    return res
