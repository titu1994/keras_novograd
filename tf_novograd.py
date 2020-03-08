import functools

import six
import tensorflow as tf
from tensorflow.python.distribute import distribution_strategy_context as distribute_ctx
from tensorflow.python.keras import backend
from tensorflow.python.keras import initializers
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2, _var_key
from tensorflow.python.ops import variables as tf_variables


class NovoGrad(OptimizerV2):
    """NovoGrad optimizer.

    Default parameters follow those provided in the original paper.

    # Arguments
        lr: float >= 0. Learning rate.
        final_lr: float >= 0. Final learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        gamma: float >= 0. Convergence speed of the bound function.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        weight_decay: Weight decay weight.
        amsgrad: boolean. Whether to apply the AMSBound variant of this
            algorithm.

    # References
        - [Stochastic Gradient Methods with Layer-wise Adaptive Moments for Training of Deep Networks]
          (https://arxiv.org/abs/1905.11286)
        - [Adam - A Method for Stochastic Optimization]
          (https://arxiv.org/abs/1412.6980v8)
    """

    def __init__(self,
                 learning_rate=0.001,
                 beta_1=0.95,
                 beta_2=0.5,
                 epsilon=None,
                 weight_decay=0.0,
                 amsgrad=False,
                 grad_averaging=False,
                 name='NovoGrad', **kwargs):
        super(NovoGrad, self).__init__(name, **kwargs)

        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self._set_hyper('decay', self._initial_decay)
        self.epsilon = epsilon or tf.keras.backend.epsilon()
        self.amsgrad = amsgrad
        self.grad_averaging = grad_averaging
        self.weight_decay = weight_decay

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'm')

        for var in var_list:
            self.add_scalar_slots(var, 'vhat')
            self.add_scalar_slots(var, 'grads_ema')

    def add_scalar_slots(self, var, slot_name, initializer="zeros"):
        """Add a new slot variable for `var`."""
        if slot_name not in self._slot_names:
            self._slot_names.append(slot_name)
        var_key = _var_key(var)
        slot_dict = self._slots.setdefault(var_key, {})
        weight = slot_dict.get(slot_name, None)
        if weight is None:
            if isinstance(initializer, six.string_types) or callable(initializer):
                initializer = initializers.get(initializer)
                initial_value = functools.partial(
                    initializer, shape=[], dtype=var.dtype)
            else:
                initial_value = initializer
            strategy = distribute_ctx.get_strategy()
            with strategy.extended.colocate_vars_with(var):
                weight = tf_variables.Variable(
                    name="%s/%s" % (var._shared_name, slot_name),  # pylint: disable=protected-access
                    dtype=var.dtype,
                    trainable=False,
                    initial_value=initial_value)
            backend.track_variable(weight)
            slot_dict[slot_name] = weight
            self._restore_slot_variable(
                slot_name=slot_name, variable=var,
                slot_variable=weight)
            self._weights.append(weight)
        return weight

    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)

        m = self.get_slot(var, 'm')
        vhat = self.get_slot(var, 'vhat')
        grad_ema = self.get_slot(var, 'grads_ema')

        beta_1_t = self._get_hyper('beta_1', var_dtype)
        beta_2_t = self._get_hyper('beta_2', var_dtype)
        epsilon_t = tf.convert_to_tensor(self.epsilon, var_dtype)
        t = tf.cast(self.iterations + 1, var_dtype)

        # compute ema for grads^2 for each layer
        g_2 = tf.reduce_sum(tf.square(x=tf.cast(grad, tf.float32)))
        g_ema_new = tf.cond(tf.equal(grad_ema, 0.),
                            lambda: g_2,
                            lambda: grad_ema * beta_2_t + g_2 * (1.0 - beta_2_t))

        if self.amsgrad:
            g_ema_new = tf.maximum(vhat, g_ema_new)
            v_t = tf.compat.v1.assign(vhat, g_ema_new)
        else:
            v_t = tf.compat.v1.assign(vhat, vhat)

        grad *= 1.0 / (tf.sqrt(g_ema_new) + epsilon_t)

        # weight decay
        if self.weight_decay > 0.0:
            grad += (self.weight_decay * var)

        # Momentum --> SAG
        if self.grad_averaging:
            grad *= (1.0 - beta_1_t)

        m_t = beta_1_t * m + grad  # velocity
        m_t = tf.compat.v1.assign(m, m_t)

        grad_t = tf.compat.v1.assign(grad_ema, g_ema_new)

        with tf.control_dependencies([m_t, v_t, grad_t]):
            p_t = var - lr_t * m_t
            param_update = tf.compat.v1.assign(var, p_t)

            return tf.group(*[param_update, m_t, v_t, grad_t])

    def _resource_apply_sparse(self, grad, handle, indices):
        raise NotImplementedError("Sparse data is not supported yet")

    def get_config(self):
        config = super(NovoGrad, self).get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'decay': self._serialize_hyperparameter('decay'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'epsilon': self.epsilon,
            'weight_decay': self.weight_decay,
            'amsgrad': self.amsgrad,
            'grad_averaging': self.grad_averaging,
        })
        return config
