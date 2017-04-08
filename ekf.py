import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.training import optimizer
from tensorflow.python.training.optimizer import Optimizer
from tensorflow.python.ops import gradients
from tensorflow.python.ops.logging_ops import Print
import logging
logger = logging.getLogger('main')



def _as_list(x):
    return x if isinstance(x, (list, tuple)) else [x]


class AdamOptimizerM(optimizer.Optimizer):
    """Optimizer that implements the Adam algorithm.

    See [Kingma et. al., 2014](http://arxiv.org/abs/1412.6980)
    ([pdf](http://arxiv.org/pdf/1412.6980.pdf)).
    
    This is basically the same as Adam implemented in Tensorflow,
    the only difference being that Adam is implemented explicitely in
    in _apply_dense rather than calling some non-Python code which 
    might have special optimizations. This is to be able to distinguish between
    speed-up due to the algorithm vs. speed-up due to CUDA or other optimizations.
    
    @@__init__
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 use_locking=False, name="Adam"):
        """Construct a new Adam optimizer.

        Initialization:

        ```
        m_0 <- 0 (Initialize initial 1st moment vector)
        v_0 <- 0 (Initialize initial 2nd moment vector)
        t <- 0 (Initialize timestep)
        ```

        The update rule for `variable` with gradient `g` uses an optimization
        described at the end of section2 of the paper:

        ```
        t <- t + 1
        lr_t <- learning_rate * sqrt(1 - beta2^t) / (1 - beta1^t)

        m_t <- beta1 * m_{t-1} + (1 - beta1) * g
        v_t <- beta2 * v_{t-1} + (1 - beta2) * g * g
        variable <- variable - lr_t * m_t / (sqrt(v_t) + epsilon)
        ```

        The default value of 1e-8 for epsilon might not be a good default in
        general. For example, when training an Inception network on ImageNet a
        current good choice is 1.0 or 0.1.

        Note that in dense implement of this algorithm, m_t, v_t and variable will
        update even if g is zero, but in sparse implement, m_t, v_t and variable
        will not update in iterations g is zero.

        Args:
          learning_rate: A Tensor or a floating point value.  The learning rate.
          beta1: A float value or a constant float tensor.
            The exponential decay rate for the 1st moment estimates.
          beta2: A float value or a constant float tensor.
            The exponential decay rate for the 2nd moment estimates.
          epsilon: A small constant for numerical stability.
          use_locking: If True use locks for update operations.
          name: Optional name for the operations created when applying gradients.
            Defaults to "Adam".
        """
        super(AdamOptimizerM, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None

        # Variables to accumulate the powers of the beta parameters.
        # Created in _create_slots when we know the variables to optimize.
        self._beta1_power = None
        self._beta2_power = None

        # Created in SparseApply if needed.
        self._updated_lr = None

    def _get_beta_accumulators(self):
        return self._beta1_power, self._beta2_power

    def _create_slots(self, var_list):
        # Create the beta1 and beta2 accumulators on the same device as the first
        # variable.
        if (self._beta1_power is None or
                    self._beta1_power.graph is not var_list[0].graph):
            with ops.colocate_with(var_list[0]):
                self._beta1_power = variables.Variable(self._beta1,
                                                       name="beta1_power",
                                                       trainable=False)
                self._beta2_power = variables.Variable(self._beta2,
                                                       name="beta2_power",
                                                       trainable=False)
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")
        self._epsilon_t = ops.convert_to_tensor(self._epsilon, name="epsilon")

    def _apply_dense(self, grad, var):
        beta1_power = math_ops.cast(self._beta1_power, var.dtype.base_dtype)
        beta2_power = math_ops.cast(self._beta2_power, var.dtype.base_dtype)
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
        lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))
        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * (1 - beta1_t)
        m_t = state_ops.assign(m, m * beta1_t,
                               use_locking=self._use_locking)
        m_t = state_ops.assign_add(m_t, m_scaled_g_values,
                                   use_locking=self._use_locking)
        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_scaled_g_values = (grad * grad) * (1 - beta2_t)
        v_t = state_ops.assign(v, v * beta2_t, use_locking=self._use_locking)
        v_t = state_ops.assign_add(v_t, v_scaled_g_values,
                                   use_locking=self._use_locking)
        v_sqrt = math_ops.sqrt(v_t)
        var_update = state_ops.assign_sub(var,
                                          lr * m_t / (v_sqrt + epsilon_t),
                                          use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, m_t, v_t])

    def _apply_sparse(self, grad, var):
        beta1_power = math_ops.cast(self._beta1_power, var.dtype.base_dtype)
        beta2_power = math_ops.cast(self._beta2_power, var.dtype.base_dtype)
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
        lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))
        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = grad.values * (1 - beta1_t)
        m_t = state_ops.assign(m, m * beta1_t,
                               use_locking=self._use_locking)
        m_t = state_ops.scatter_add(m_t, grad.indices, m_scaled_g_values,
                                    use_locking=self._use_locking)
        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_scaled_g_values = (grad.values * grad.values) * (1 - beta2_t)
        v_t = state_ops.assign(v, v * beta2_t, use_locking=self._use_locking)
        v_t = state_ops.scatter_add(v_t, grad.indices, v_scaled_g_values,
                                    use_locking=self._use_locking)
        v_sqrt = math_ops.sqrt(v_t)
        var_update = state_ops.assign_sub(var,
                                          lr * m_t / (v_sqrt + epsilon_t),
                                          use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, m_t, v_t])

    def _finish(self, update_ops, name_scope):
        # Update the power accumulators.
        with ops.control_dependencies(update_ops):
            with ops.colocate_with(self._beta1_power):
                update_beta1 = self._beta1_power.assign(
                    self._beta1_power * self._beta1_t,
                    use_locking=self._use_locking)
                update_beta2 = self._beta2_power.assign(
                    self._beta2_power * self._beta2_t,
                    use_locking=self._use_locking)
        return control_flow_ops.group(*update_ops + [update_beta1, update_beta2],
                                      name=name_scope)


class EKFWeightOptimizer(optimizer.Optimizer):
    """Optimizer that implements the EKF algorithms to train weights.
    Each Tensor is treated independently from other tensors (otherwise the
    memory requirements would be too much). 

    @@__init__
    """

    def __init__(self, y_shape, learning_rate=0.01, P0=100, Q=1e-2,
                 use_locking=False, name="EKFWeight"):
        """Construct a new Adam optimizer.

        Initialization:

        Initializes P (the initial covariance of the tensor) 
        P <- Identity * P0
        Initializes the Dynamic noise Q
        Q <- Identity * self.Q
        Initializes the measurment noise R:
        R <- Identity / learning_rate

        Args:
            y_shape: A tensor, the shape of the network output Tensor
          learning_rate: A float, R = Identity / learning_rate
          P0: A float, prior convariance
          Q: A float, the covariance of the noise in the dynamics
          use_locking: If True use locks for update operations.
          name: Optional name for the operations created when applying gradients.
            Defaults to "Adam".
        """
        super(EKFWeightOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self.P0 = P0
        self.Q = Q
        self.y_shape = y_shape
        self.y_dim = np.prod(y_shape)
        #self.dim_y = math_ops.to_int32(math_ops.reduce_sum(self.y_shape))

        # Tensor versions of the constructor arguments, created in _prepare().
        self._Rt = None

    def _create_slots(self, var_list):
        # Create the beta1 and beta2 accumulators on the same device as the first
        # variable.
        if (self._Rt is None or
                    self._Rt.graph is not var_list[0].graph):
            with ops.colocate_with(var_list[0]):
                self._Rt = linalg_ops.eye(self.y_dim,
                                       name="R") / self._lr
                print("R shape; {}".format(self._Rt.get_shape().as_list()))

        # Create slots for the first and second moments.
        for v in var_list:
            self._get_or_make_slot(v, linalg_ops.eye(gen_array_ops.size(v.initialized_value()),
                                                  dtype=v.dtype.base_dtype) * self.Q, "Q", self._name)
            self._get_or_make_slot(v, linalg_ops.eye(gen_array_ops.size(v.initialized_value()),
                                                  dtype=v.dtype.base_dtype) * self.P0, "P", self._name)

    def minimize(self, y_target, y_pred, global_step=None, var_list=None,
                     gate_gradients=optimizer.Optimizer.GATE_OP, aggregation_method=None,
                     colocate_gradients_with_ops=False, name=None,
                     grad_loss=None):

        """
        Applies the EKF optimization using measurement y_target and prediction y_pred

        Args:
          y_target: The target tensor which we would like the trained network to output
          y_pred: The actual output Tensor of the network
          global_step: Optional `Variable` to increment by one after the
            variables have been updated.
          var_list: Optional list of `Variable` objects to update to minimize
            `loss`.  Defaults to the list of variables collected in the graph
            under the key `GraphKeys.TRAINABLE_VARIABLES`.
          gate_gradients: How to gate the computation of gradients.  Can be
            `GATE_NONE`, `GATE_OP`, or  `GATE_GRAPH`.
          aggregation_method: Specifies the method used to combine gradient terms.
            Valid values are defined in the class `AggregationMethod`.
          colocate_gradients_with_ops: If True, try colocating gradients with
            the corresponding op.
          name: Optional name for the returned operation.
          grad_loss: Optional. A `Tensor` holding the gradient computed for `loss`.

        Returns:
          An Operation that updates the variables in `var_list`.  If `global_step`
          was not `None`, that operation also increments `global_step`.

        Raises:
          ValueError: If some of the variables are not `Variable` objects.
        """
        if var_list is None:
            var_list = (
                variables.trainable_variables() +
                ops.get_collection(ops.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))

        real_grads = gradients.gradients(y_pred, var_list)
        filter_none_gs = [g for g in real_grads if g is not None]
        n_nones = len([g for g in real_grads if g is None])
        logger.info("Number of identically zero gradients: {}".format(n_nones))
        Hs_and_vars = self.compute_gradient_Hs(
            y_pred, var_list=var_list, gate_gradients=gate_gradients,
            aggregation_method=aggregation_method,
            colocate_gradients_with_ops=colocate_gradients_with_ops,
            grad_loss=grad_loss)

        vars_with_grad = [v for H, v in Hs_and_vars if H is not None]
        if not filter_none_gs:
            raise ValueError(
                "No gradients provided for any variable, check your graph for ops"
                " that do not support gradients, between variables %s and y_pred %s." %
                ([str(v) for _, v in Hs_and_vars], y_pred))
        error = gen_array_ops.reshape(y_target-y_pred, [-1, 1], name="error")
        return self.apply_gradients(Hs_and_vars, error, global_step=global_step,
                                    name=name)

    def apply_gradients(self, grads_and_vars, error, global_step=None, name=None):
        """
        Updates the weights of the network using EKF using grads_and_vars, the linearized measurment
        Matrices H and the variables, as well as the error = y_target-y_pred.

        Args:
          grads_and_vars: List of (H, variable) pairs as returned by
            `compute_gradient_Hs()`.
            error: the tensor y_target - y_pred
          global_step: Optional `Variable` to increment by one after the
            variables have been updated.
          name: Optional name for the returned operation.  Default to the
            name passed to the `Optimizer` constructor.

        Returns:
          An `Operation` that applies the specified gradients. If `global_step`
          was not None, that operation also increments `global_step`.

        Raises:
          TypeError: If `grads_and_vars` is malformed.
          ValueError: If none of the variables have gradients.
        """
        # This is a default implementation of apply_gradients() that can be shared
        # by most optimizers.  It relies on the subclass implementing the following
        # methods: _create_slots(), _prepare(), _apply_dense(), and _apply_sparse().

        grads_and_vars = tuple(grads_and_vars)  # Make sure repeat iteration works.
        if not grads_and_vars:
            raise ValueError("No variables provided.")
        converted_grads_and_vars = []
        for g, v in grads_and_vars:
            if g is not None:
                try:
                    # Convert the grad to Tensor or IndexedSlices if necessary.
                    g = ops.convert_to_tensor_or_indexed_slices(g)
                except TypeError:
                    raise TypeError(
                        "Gradient must be convertible to a Tensor"
                        " or IndexedSlices, or None: %s" % g)
                if not isinstance(g, (ops.Tensor, ops.IndexedSlices)):
                    raise TypeError(
                        "Gradient must be a Tensor, IndexedSlices, or None: %s" % g)
            converted_grads_and_vars.append((g, v))

        converted_grads_and_vars = tuple(converted_grads_and_vars)
        var_list = [v for g, v in converted_grads_and_vars if g is not None]
        if not var_list:
            raise ValueError("No gradients provided for any variable: %s." %
                             ([str(v) for  _, v in converted_grads_and_vars],))
        with ops.control_dependencies(None):
            self._create_slots(var_list)
        update_ops = []
        with ops.name_scope(name, self._name) as name:
            self._prepare()
            for grad, var in converted_grads_and_vars:
                if grad is None:
                    continue
                # We colocate all ops created in _apply_dense or _apply_sparse
                # on the same device as the variable.
                with ops.name_scope("update_" + var.op.name), ops.colocate_with(var):
                    update_ops.append(self._apply_dense(grad, var, error))
            if global_step is None:
                apply_updates = self._finish(update_ops, name)
            else:
                with ops.control_dependencies([self._finish(update_ops, "update")]):
                    with ops.colocate_with(global_step):
                        apply_updates = state_ops.assign_add(global_step, 1, name=name).op

            train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
            if apply_updates not in train_op:
                train_op.append(apply_updates)

            return apply_updates

    def _apply_dense(self, H, var, error):
        Q = self.get_slot(var, "Q")  # Process noise
        P = self.get_slot(var, "P")  # Covariance matrix
        S = self._Rt + math_ops.matmul(math_ops.matmul(H, P), H, transpose_b=True)
        Sinv = linalg_ops.matrix_inverse(S, name="Sinv")
        K = math_ops.matmul(math_ops.matmul(P, H, transpose_b=True), Sinv)

        #debugP = math_ops.trace(P)/math_ops.cast(gen_array_ops.shape(P)[0], dtype=np.float32)
        #debugK = math_ops.sqrt(math_ops.reduce_sum(math_ops.square(K))/math_ops.cast(gen_array_ops.shape(K)[1], dtype=np.float32))
        #K = Print(K, [debugP, debugK], message="P, K : ")

        dW = math_ops.matmul(K, error)
        update_weights = state_ops.assign_add(var, gen_array_ops.reshape(dW, gen_array_ops.shape(var)), use_locking=self._use_locking)
        update_P = state_ops.assign_add(P, Q - math_ops.matmul(math_ops.matmul(K, S), K, transpose_b=True), use_locking=self._use_locking)

        return control_flow_ops.group(*[update_weights, update_P])

    @staticmethod
    def calc_H(y, xs, y_size,
               colocate_gradients_with_ops=False,
               gate_gradients=False,
               aggregation_method=None):
        """
        Calculates the H matrix of EKF.
        :param y: the flattened measurment error y_true - y_prediction
        :param xs: a list of the flattened tensors of the weights of the variables.
        :param y_size: the number of elements in y
        :return: The H matrix of EKF.
        """
        xs = _as_list(xs)
        derivs = [gradients.gradients(y[i], xs,
                            gate_gradients=(gate_gradients == Optimizer.GATE_OP),
                            aggregation_method=aggregation_method,
                            colocate_gradients_with_ops=colocate_gradients_with_ops) for i in range(y_size)]
        #Transpose the list of lists
        derivs = list(map(list, zip(*derivs)))
        flat_d = [e for ls in derivs for e in ls]
        logger.info("Inside calc_H n derivs: {}".format(len(flat_d)))
        logger.info("Inside calc_H None derivs: {}".format(len([e for e in flat_d if e is None])))

        def nones2nulls(parts, shape):
            def none2nulls(part):
                return part if part is not None else array_ops.zeros(shape)
            return list(map(none2nulls, parts))

        #Glue the parts of the H matrices together
        #Hs = [gen_array_ops.reshape(array_ops.stack(nones2nulls(parts, gen_array_ops.shape(xs[i])), axis=0),
        #                            [y_size, -1]) for i, parts in enumerate(derivs)]
        Hs = [gen_array_ops.reshape(array_ops.stack(parts, axis=0),
                                    [y_size, -1]) for i, parts in enumerate(derivs)]
        #Hs[-1] = Print(Hs[-1], [math_ops.reduce_max(math_ops.abs(hh)) for hh in Hs], message="Hs:  \n {} \n".format([x.name for x in xs]))
        #Hs[0] = Print(Hs[0], [math_ops.reduce_max(math_ops.abs(hh)) for hh in flat_d],
        #               message="Calc H Primordial derivatives: ")
        return Hs



    def compute_gradient_Hs(self, y_pred, var_list=None,
                          gate_gradients=Optimizer.GATE_OP,
                          aggregation_method=None,
                          colocate_gradients_with_ops=False,
                          grad_loss=None):
        """Computes the EKF linearize measurment matrix H for the variables in `var_list`.

        This is the first part of `minimize()`.  It returns a list
        of (H, variable) pairs where "H" is the derivative of the measurment 
        function h with respect to "variable"
        for "variable".  

        Args:
          y_pred: The prediction of the network
            NOTE THAT THIS SHOULD BE A TENSOR AND NOT A SCALAR LIKE IN OTHER OPTIMIZERS.
          var_list: Optional list of `tf.Variable` to update to minimize
            `loss`.  Defaults to the list of variables collected in the graph
            under the key `GraphKey.TRAINABLE_VARIABLES`.
          gate_gradients: How to gate the computation of gradients.  Can be
            `GATE_NONE`, `GATE_OP`, or `GATE_GRAPH`.
          aggregation_method: Specifies the method used to combine gradient terms.
            Valid values are defined in the class `AggregationMethod`.
          colocate_gradients_with_ops: If True, try colocating gradients with
            the corresponding op.
          grad_loss: Optional. A `Tensor` holding the gradient computed for `loss`.

        Returns:
          A list of (H, variable) pairs. Variable is always present, but
          H can be `None`.

        Raises:
          TypeError: If `var_list` contains anything else than `Variable` objects.
          ValueError: If some arguments are invalid.
        """
        if gate_gradients not in [Optimizer.GATE_NONE, Optimizer.GATE_OP,
                                  Optimizer.GATE_GRAPH]:
            raise ValueError("gate_gradients must be one of: Optimizer.GATE_NONE, "
                             "Optimizer.GATE_OP, Optimizer.GATE_GRAPH.  Not %s" %
                             gate_gradients)
        self._assert_valid_dtypes([y_pred])
        if grad_loss is not None:
            self._assert_valid_dtypes([grad_loss])
        if var_list is None:
            var_list = (
                variables.trainable_variables() +
                ops.get_collection(ops.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))
        if not var_list:
            raise ValueError("No variables to optimize.")
        Hs = self.calc_H(gen_array_ops.reshape(y_pred, [-1]), var_list, self.y_dim,
            gate_gradients=(gate_gradients == Optimizer.GATE_OP),
            aggregation_method=aggregation_method,
            colocate_gradients_with_ops=colocate_gradients_with_ops)
        if gate_gradients == Optimizer.GATE_GRAPH:
            Hs = control_flow_ops.tuple(Hs)
        grads_and_vars = list(zip(Hs, var_list))
        self._assert_valid_dtypes([v for g, v in grads_and_vars if g is not None])
        return grads_and_vars