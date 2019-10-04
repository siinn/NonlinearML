import tensorflow as tf
from datetime import datetime

#-------------------------------------------------------------------------------
# Model class
#-------------------------------------------------------------------------------
class TensorflowModel:
    """ Tensorflow model class to perform compile, fit, and evaluation. 
    Attributes:
        model: tf.keras.Sequential model
        params: MOdel parameters given in dictionary.
    """
    def __init__(self, model, params, log_path):
        """ Initialize variables."""
        print("Building model..")
        # parameter set
        self.model = model
        self.params = params
        self.log_path = log_path

    def get_param_string(self):
        """ Get name as a string."""
        names = []
        for key in self.params:
            if 'name' in dir(self.params[key]):
                names.append(key+'='+self.params[key].name)
            else:
                names.append(key+'='+str(self.params[key]))
        return ",".join(names)

    def get_tfboard(self):
        """ Return tfboard with updated log path."""
        path = "_".join([
            self.log_path,
            self.get_param_string(),
            str(datetime.now()).replace(' ', '_')]).strip('\'')
        return tf.keras.callbacks.TensorBoard(
            log_dir=path,
            histogram_freq=False, update_freq='epoch')

    def get_earlystop(self):
        """ Add early stopping and tfboard to callback"""
        return tf.keras.callbacks.EarlyStopping(
            monitor=self.params['metrics'].name,
            patience=self.params["patience"],
            mode='auto')

    def set_params(self, **params):
        """ Set model parameters. """
        self.params = params
        self.model = self.params['model']
        return self

    def compile(self):
        """ Compile model."""
        if not None:
            self.model.compile(
                #loss=tf.losses.SparseCategoricalCrossentropy(),
                loss=self.params["loss"],
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=self.params["learning_rate"]),
                metrics=[self.params['metrics']])
        else:
            print("Set hyperparameters first by .set_params method")
        return self

    def fit(self, X, y):
        """ Train model."""
        if not None:
            self.compile()
            history = self.model.fit(
                x=X.values, y=y.values,
                epochs=self.params["epochs"],
                batch_size=self.params['batch_size'],
                validation_split=self.params["validation_split"],
                callbacks=[self.get_earlystop(), self.get_tfboard()],
                verbose=1)
        else:
            print("Set hyperparameters first by .set_params method")
        return self

    def predict(self, x):
        """ Make prediction."""
        #y_prob = self.model.predict(x.values, batch_size=self.params['batch_size'])
        y_prob = self.model.predict_on_batch(x.values)
        if 'numpy' in dir(y_prob):
            y_prob = y_prob.numpy() # Convert to numpy if trained on GPU
        y_classes = y_prob.argmax(axis=-1)
        return y_classes


def extract_metrics(class_report, num_class):
    ''' Extract metrics from sklearn classification report.
    Args:
        class_report: Classification report from sklearn.metrics.
        num_class: Number of classes
    Return:
        Dictionary containing evaluation metrics.
    '''
    # Dictionary to hold results
    results = {}
    # List of classes = classificatin labels + macro and micro average
    list_classes = [str(float(i)) for i in range(num_class)] + \
                   ['macro avg', 'weighted avg']
    # Get metrics for each classes 
    for cls in list_classes:
        results['train_%s_precision' % cls] = class_report[cls]['precision']
        results['train_%s_recall' % cls]    = class_report[cls]['recall']
        results['train_%s_f1-score' % cls]  = class_report[cls]['f1-score']
    # Get overall accuracy
    results['accuracy'] = class_report['accuracy']
    return results

