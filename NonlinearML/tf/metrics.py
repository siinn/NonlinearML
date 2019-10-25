import six
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.keras import backend as K
from tensorflow.python.ops.losses import util as tf_losses_utils

class MeanMetricWrapper(tf.keras.metrics.Mean):
  """Wraps a stateless metric function with the Mean metric."""

  def __init__(self, fn, name=None, dtype=None, **kwargs):
    """Creates a `MeanMetricWrapper` instance.
    Args:
      fn: The metric function to wrap, with signature
        `fn(y_true, y_pred, **kwargs)`.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
      **kwargs: The keyword arguments that are passed on to `fn`.
    """
    super(MeanMetricWrapper, self).__init__(name=name, dtype=dtype)
    self._fn = fn
    self._fn_kwargs = kwargs

  def update_state(self, y_true, y_pred, sample_weight=None):
    """Accumulates metric statistics.
    `y_true` and `y_pred` should have the same shape.
    Args:
      y_true: The ground truth values.
      y_pred: The predicted values.
      sample_weight: Optional weighting of each example. Defaults to 1. Can be
        a `Tensor` whose rank is either 0, or the same rank as `y_true`,
        and must be broadcastable to `y_true`.
    Returns:
      Update op.
    """
    y_true = math_ops.cast(y_true, self._dtype)
    y_pred = math_ops.cast(y_pred, self._dtype)
    [y_true, y_pred], sample_weight = \
        metrics_utils.ragged_assert_compatible_and_get_flat_values(
            [y_true, y_pred], sample_weight)
    y_pred, y_true = tf_losses_utils.squeeze_or_expand_dimensions(
        y_pred, y_true)

    matches = self._fn(y_true, y_pred, **self._fn_kwargs)
    return super(MeanMetricWrapper, self).update_state(
        matches, sample_weight=sample_weight)

  def get_config(self):
    config = {}
    for k, v in six.iteritems(self._fn_kwargs):
      config[k] = K.eval(v) if is_tensor_or_variable(v) else v
    base_config = super(MeanMetricWrapper, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))



class MeanLogSquaredError(MeanMetricWrapper):
    """Computes the mean of log squared difference between labels and
    predictions.
    loss = ln((y_pred - y_true)^2 + 1)
    """
    def mean_log_squared_error(self, y_true, y_pred):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        return K.mean(math_ops.log(math_ops.squared_difference(y_pred, y_true)+ 1.), axis=-1)
    
    def __init__(self, name='mean_log_squared_error', dtype=None):
      super(MeanLogSquaredError, self).__init__(
          self.mean_log_squared_error, name, dtype=dtype)






