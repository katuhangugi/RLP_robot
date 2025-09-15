import baselines.common.tf_util as U
import numpy as np
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

MPI = None

import tensorflow as tf

class TfAdamOptimizer(tf.compat.v1.train.Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name="MpiAdam"):
        super(TfAdamOptimizer, self).__init__(use_locking, name)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def _create_slots(self, var_list):
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _apply_dense(self, grad, var):
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        t = tf.compat.v1.train.get_or_create_global_step()
        t_plus_1 = tf.compat.v1.assign_add(t, 1)
        t_float = tf.cast(t_plus_1, tf.float32)
        lr_t = self.learning_rate * tf.sqrt(1 - tf.pow(self.beta2, t_float)) / (1 - tf.pow(self.beta1, t_float))
        m_t = m.assign(self.beta1 * m + (1 - self.beta1) * grad)
        v_t = v.assign(self.beta2 * v + (1 - self.beta2) * tf.square(grad))
        var_update = tf.compat.v1.assign_sub(var, lr_t * m_t / (tf.sqrt(v_t) + self.epsilon))
        return tf.group(var_update, m_t, v_t, t_plus_1)

    def _resource_apply_dense(self, grad, handle):
        return self._apply_dense(grad, handle)

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        raise NotImplementedError("Sparse gradient updates are not supported.")


MPI = None

import tensorflow as tf

class TfAdamOptimizer(tf.compat.v1.train.Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name="MpiAdam"):
        super(TfAdamOptimizer, self).__init__(use_locking, name)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def _create_slots(self, var_list):
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _apply_dense(self, grad, var):
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        t = tf.compat.v1.train.get_or_create_global_step()
        t_plus_1 = tf.compat.v1.assign_add(t, 1)
        t_float = tf.cast(t_plus_1, tf.float32)
        lr_t = self.learning_rate * tf.sqrt(1 - tf.pow(self.beta2, t_float)) / (1 - tf.pow(self.beta1, t_float))
        m_t = m.assign(self.beta1 * m + (1 - self.beta1) * grad)
        v_t = v.assign(self.beta2 * v + (1 - self.beta2) * tf.square(grad))
        var_update = tf.compat.v1.assign_sub(var, lr_t * m_t / (tf.sqrt(v_t) + self.epsilon))
        return tf.group(var_update, m_t, v_t, t_plus_1)

    def _resource_apply_dense(self, grad, handle):
        return self._apply_dense(grad, handle)

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        raise NotImplementedError("Sparse gradient updates are not supported.")


MPI = None

import tensorflow as tf

class TfAdamOptimizer(tf.compat.v1.train.Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name="MpiAdam"):
        super(TfAdamOptimizer, self).__init__(use_locking, name)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def _create_slots(self, var_list):
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _apply_dense(self, grad, var):
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        t = tf.compat.v1.train.get_or_create_global_step()
        t_plus_1 = tf.compat.v1.assign_add(t, 1)
        t_float = tf.cast(t_plus_1, tf.float32)
        lr_t = self.learning_rate * tf.sqrt(1 - tf.pow(self.beta2, t_float)) / (1 - tf.pow(self.beta1, t_float))
        m_t = m.assign(self.beta1 * m + (1 - self.beta1) * grad)
        v_t = v.assign(self.beta2 * v + (1 - self.beta2) * tf.square(grad))
        var_update = tf.compat.v1.assign_sub(var, lr_t * m_t / (tf.sqrt(v_t) + self.epsilon))
        return tf.group(var_update, m_t, v_t, t_plus_1)

    def _resource_apply_dense(self, grad, handle):
        return self._apply_dense(grad, handle)

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        raise NotImplementedError("Sparse gradient updates are not supported.")


MPI = None

import tensorflow as tf

class TfAdamOptimizer(tf.compat.v1.train.Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name="MpiAdam"):
        super(TfAdamOptimizer, self).__init__(use_locking, name)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def _create_slots(self, var_list):
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _apply_dense(self, grad, var):
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        t = tf.compat.v1.train.get_or_create_global_step()
        t_plus_1 = tf.compat.v1.assign_add(t, 1)
        t_float = tf.cast(t_plus_1, tf.float32)
        lr_t = self.learning_rate * tf.sqrt(1 - tf.pow(self.beta2, t_float)) / (1 - tf.pow(self.beta1, t_float))
        m_t = m.assign(self.beta1 * m + (1 - self.beta1) * grad)
        v_t = v.assign(self.beta2 * v + (1 - self.beta2) * tf.square(grad))
        var_update = tf.compat.v1.assign_sub(var, lr_t * m_t / (tf.sqrt(v_t) + self.epsilon))
        return tf.group(var_update, m_t, v_t, t_plus_1)

    def _resource_apply_dense(self, grad, handle):
        return self._apply_dense(grad, handle)

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        raise NotImplementedError("Sparse gradient updates are not supported.")


MPI = None

import tensorflow as tf

class TfAdamOptimizer(tf.compat.v1.train.Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name="MpiAdam"):
        super(TfAdamOptimizer, self).__init__(use_locking, name)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def _create_slots(self, var_list):
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _apply_dense(self, grad, var):
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        t = tf.compat.v1.train.get_or_create_global_step()
        t_plus_1 = tf.compat.v1.assign_add(t, 1)
        t_float = tf.cast(t_plus_1, tf.float32)
        lr_t = self.learning_rate * tf.sqrt(1 - tf.pow(self.beta2, t_float)) / (1 - tf.pow(self.beta1, t_float))
        m_t = m.assign(self.beta1 * m + (1 - self.beta1) * grad)
        v_t = v.assign(self.beta2 * v + (1 - self.beta2) * tf.square(grad))
        var_update = tf.compat.v1.assign_sub(var, lr_t * m_t / (tf.sqrt(v_t) + self.epsilon))
        return tf.group(var_update, m_t, v_t, t_plus_1)

    def _resource_apply_dense(self, grad, handle):
        return self._apply_dense(grad, handle)

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        raise NotImplementedError("Sparse gradient updates are not supported.")


MPI = None

import tensorflow as tf

class TfAdamOptimizer(tf.compat.v1.train.Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name="MpiAdam"):
        super(TfAdamOptimizer, self).__init__(use_locking, name)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def _create_slots(self, var_list):
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _apply_dense(self, grad, var):
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        t = tf.compat.v1.train.get_or_create_global_step()
        t_plus_1 = tf.compat.v1.assign_add(t, 1)
        t_float = tf.cast(t_plus_1, tf.float32)
        lr_t = self.learning_rate * tf.sqrt(1 - tf.pow(self.beta2, t_float)) / (1 - tf.pow(self.beta1, t_float))
        m_t = m.assign(self.beta1 * m + (1 - self.beta1) * grad)
        v_t = v.assign(self.beta2 * v + (1 - self.beta2) * tf.square(grad))
        var_update = tf.compat.v1.assign_sub(var, lr_t * m_t / (tf.sqrt(v_t) + self.epsilon))
        return tf.group(var_update, m_t, v_t, t_plus_1)

    def _resource_apply_dense(self, grad, handle):
        return self._apply_dense(grad, handle)

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        raise NotImplementedError("Sparse gradient updates are not supported.")


MPI = None

import tensorflow as tf

class TfAdamOptimizer(tf.compat.v1.train.Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name="MpiAdam"):
        super(TfAdamOptimizer, self).__init__(use_locking, name)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def _create_slots(self, var_list):
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _apply_dense(self, grad, var):
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        t = tf.compat.v1.train.get_or_create_global_step()
        t_plus_1 = tf.compat.v1.assign_add(t, 1)
        t_float = tf.cast(t_plus_1, tf.float32)
        lr_t = self.learning_rate * tf.sqrt(1 - tf.pow(self.beta2, t_float)) / (1 - tf.pow(self.beta1, t_float))
        m_t = m.assign(self.beta1 * m + (1 - self.beta1) * grad)
        v_t = v.assign(self.beta2 * v + (1 - self.beta2) * tf.square(grad))
        var_update = tf.compat.v1.assign_sub(var, lr_t * m_t / (tf.sqrt(v_t) + self.epsilon))
        return tf.group(var_update, m_t, v_t, t_plus_1)

    def _resource_apply_dense(self, grad, handle):
        return self._apply_dense(grad, handle)

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        raise NotImplementedError("Sparse gradient updates are not supported.")


MPI = None

import tensorflow as tf

class TfAdamOptimizer(tf.compat.v1.train.Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name="MpiAdam"):
        super(TfAdamOptimizer, self).__init__(use_locking, name)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def _create_slots(self, var_list):
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _apply_dense(self, grad, var):
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        t = tf.compat.v1.train.get_or_create_global_step()
        t_plus_1 = tf.compat.v1.assign_add(t, 1)
        t_float = tf.cast(t_plus_1, tf.float32)
        lr_t = self.learning_rate * tf.sqrt(1 - tf.pow(self.beta2, t_float)) / (1 - tf.pow(self.beta1, t_float))
        m_t = m.assign(self.beta1 * m + (1 - self.beta1) * grad)
        v_t = v.assign(self.beta2 * v + (1 - self.beta2) * tf.square(grad))
        var_update = tf.compat.v1.assign_sub(var, lr_t * m_t / (tf.sqrt(v_t) + self.epsilon))
        return tf.group(var_update, m_t, v_t, t_plus_1)

    def _resource_apply_dense(self, grad, handle):
        return self._apply_dense(grad, handle)

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        raise NotImplementedError("Sparse gradient updates are not supported.")

MPI = None


import tensorflow as tf


class TfAdamOptimizer(tf.compat.v1.train.Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name="MpiAdam"):
        super(TfAdamOptimizer, self).__init__(use_locking, name)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def _create_slots(self, var_list):
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _apply_dense(self, grad, var):
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        
        t = tf.compat.v1.train.get_or_create_global_step()
        
        t_plus_1 = tf.compat.v1.assign_add(t, 1)  # 直接更新 t
        t_float = tf.cast(t_plus_1, tf.float32)
        lr_t = self.learning_rate * tf.sqrt(1 - tf.pow(self.beta2, t_float)) / (1 - tf.pow(self.beta1, t_float))

        m_t = m.assign(self.beta1 * m + (1 - self.beta1) * grad)
        v_t = v.assign(self.beta2 * v + (1 - self.beta2) * tf.square(grad))

        # 计算参数的更新
        var_update = tf.compat.v1.assign_sub(var, lr_t * m_t / (tf.sqrt(v_t) + self.epsilon))

        return tf.group(var_update, m_t, v_t, t_plus_1)

    def _resource_apply_dense(self, grad, handle):
        return self._apply_dense(grad, handle)

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        raise NotImplementedError("Sparse gradient updates are not supported.")



class MpiAdam(object):
    def __init__(self, var_list, *, beta1=0.9, beta2=0.999, epsilon=1e-08, scale_grad_by_procs=True, comm=None):
        self.var_list = var_list
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.scale_grad_by_procs = scale_grad_by_procs
        size = sum(U.numel(v) for v in var_list)
        self.m = np.zeros(size, 'float32')
        self.v = np.zeros(size, 'float32')
        self.t = 0
        self.setfromflat = U.SetFromFlat(var_list)
        self.getflat = U.GetFlat(var_list)
        self.comm = MPI.COMM_WORLD if comm is None and MPI is not None else comm

    def update(self, localg, stepsize):
        if self.t % 100 == 0:
            self.check_synced()
        localg = localg.astype('float32')
        if self.comm is not None:
            globalg = np.zeros_like(localg)
            self.comm.Allreduce(localg, globalg, op=MPI.SUM)
            if self.scale_grad_by_procs:
                globalg /= self.comm.Get_size()
        else:
            globalg = np.copy(localg)

        self.t += 1
        a = stepsize * np.sqrt(1 - self.beta2**self.t)/(1 - self.beta1**self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = (- a) * self.m / (np.sqrt(self.v) + self.epsilon)
        self.setfromflat(self.getflat() + step)

    def sync(self):
        if self.comm is None:
            return
        theta = self.getflat()
        self.comm.Bcast(theta, root=0)
        self.setfromflat(theta)

    def check_synced(self):
        if self.comm is None:
            return
        if self.comm.Get_rank() == 0: # this is root
            theta = self.getflat()
            self.comm.Bcast(theta, root=0)
        else:
            thetalocal = self.getflat()
            thetaroot = np.empty_like(thetalocal)
            self.comm.Bcast(thetaroot, root=0)
            assert (thetaroot == thetalocal).all(), (thetaroot, thetalocal)
