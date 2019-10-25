import six
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.losses import util as tf_losses_util

class LossFunctionWrapper(tf.keras.losses.Loss):
  """Wraps a loss function in the `Loss` class.
  https://github.com/tensorflow/tensorflow/blob/r2.0/
    tensorflow/python/keras/losses.py#L180
  Args:
    fn: The loss function to wrap, with signature `fn(y_true, y_pred,
      **kwargs)`.
    reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to loss.
      Default value is `AUTO`. `AUTO` indicates that the reduction option will
      be determined by the usage context. For almost all cases this defaults to
      `SUM_OVER_BATCH_SIZE`.
      When used with `tf.distribute.Strategy`, outside of built-in training
      loops such as `tf.keras` `compile` and `fit`, using `AUTO` or
      `SUM_OVER_BATCH_SIZE` will raise an error. Please see
      https://www.tensorflow.org/alpha/tutorials/distribute/training_loops
      for more details on this.
    name: (Optional) name for the loss.
    **kwargs: The keyword arguments that are passed on to `fn`.
  """
  def __init__(self,
               fn,
               reduction=losses_utils.ReductionV2.AUTO,
               name=None,
               **kwargs):
    super(LossFunctionWrapper, self).__init__(reduction=reduction, name=name)
    self.fn = fn
    self._fn_kwargs = kwargs
  
  def call(self, y_true, y_pred):
    """Invokes the `LossFunctionWrapper` instance.
    Args:
      y_true: Ground truth values.
      y_pred: The predicted values.
    Returns:
      Loss values per sample.
    """
    if tensor_util.is_tensor(y_pred) and tensor_util.is_tensor(y_true):
      y_pred, y_true = tf_losses_util.squeeze_or_expand_dimensions(
          y_pred, y_true)
    return self.fn(y_true, y_pred, **self._fn_kwargs)
    
    def get_config(self):
        config = {}
        for k, v in six.iteritems(self._fn_kwargs):
          config[k] = K.eval(v) if tf_utils.is_tensor_or_variable(v) else v
        base_config = super(LossFunctionWrapper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class MeanSquaredError(LossFunctionWrapper):
    """Computes the mean of squares of errors between labels and predictions.
    `loss = square(y_true - y_pred)`
    Usage:
    ```python
    mse = tf.keras.losses.MeanSquaredError()
    loss = mse([0., 0., 1., 1.], [1., 1., 1., 0.])
    print('Loss: ', loss.numpy())  # Loss: 0.75
    ```
    Usage with the `compile` API:
    ```python
    model = tf.keras.Model(inputs, outputs)
    model.compile('sgd', loss=tf.keras.losses.MeanSquaredError())
    ```
    """
    def mean_squared_error(self, y_true, y_pred):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        return K.mean(math_ops.squared_difference(y_pred, y_true), axis=-1)
    
    def __init__(self,
                   reduction=losses_utils.ReductionV2.AUTO,
                   name='mean_squared_error'):
        super(MeanSquaredError, self).__init__(
            self.mean_squared_error, name=name, reduction=reduction)

class MeanLogSquaredError(LossFunctionWrapper):
    """Computes the mean of log squared difference between labels and
    predictions.
    loss = ln((y_pred - y_true)^2 + 1)
    """
    def mean_log_squared_error(self, y_true, y_pred):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        return K.mean(math_ops.log(math_ops.squared_difference(y_pred, y_true)+ 1.), axis=-1)
    def __init__(self,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name='mean_log_squared_error'):
      super(MeanLogSquaredError, self).__init__(
          self.mean_log_squared_error, name=name, reduction=reduction)

