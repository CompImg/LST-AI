import tensorflow as tf
import numpy as np

def load_custom_model(model_path, compile=False):
    """
    Loads a custom TensorFlow Keras model from the specified path.

    This function is specifically designed to handle models that originally used the
    `tfa.InstanceNormalization` layer from TensorFlow Addons (tfa). Since tfa is no
    longer maintained, this function replaces the `InstanceNormalization` layer with a
    custom layer, `CustomGroupNormalization`, to ensure compatibility and avoid the need
    for installing tfa.

    Args:
    model_path (str): The file path to the saved Keras model.
    compile (bool): If True, compiles the model after loading. Defaults to False.

    Returns:
    tf.keras.Model: The loaded Keras model with `InstanceNormalization` layers replaced
    by `CustomGroupNormalization`.

    Example:
    >>> model = load_custom_model('path/to/model.h5', compile=True)
    """
    custom_objects = {
        'Addons>InstanceNormalization': CustomGroupNormalization,
    }
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=compile)



class CustomGroupNormalization(tf.keras.layers.Layer):
    """
    Custom Group Normalization layer for TensorFlow Keras models.

    This class provides an alternative to the `tfa.InstanceNormalization` layer found in
    TensorFlow Addons (tfa), which is no longer maintained and not available for MAC ARM platforms.
    It facilitates the use of group normalization in models without the dependency on tfa, ensuring
    compatibility and broader platform support.

    Args:
    groups (int): Number of groups for Group Normalization. Default is -1.
    **kwargs: Additional keyword arguments for layer configuration.
    """
    def __init__(self, groups=-1, **kwargs):
        # Extract necessary arguments from kwargs
        self.groups = kwargs.pop('groups', -1)
        self.epsilon = kwargs.pop('epsilon', 0.001)
        self.center = kwargs.pop('center', True)
        self.scale = kwargs.pop('scale', True)
        self.beta_initializer = kwargs.pop('beta_initializer', 'zeros')
        self.gamma_initializer = kwargs.pop('gamma_initializer', 'ones')
        self.beta_regularizer = kwargs.pop('beta_regularizer', None)
        self.gamma_regularizer = kwargs.pop('gamma_regularizer', None)
        self.beta_constraint = kwargs.pop('beta_constraint', None)
        self.gamma_constraint = kwargs.pop('gamma_constraint', None)

        # 'axis' argument is not used in GroupNormalization, so we remove it
        kwargs.pop('axis', None)

        super(CustomGroupNormalization, self).__init__(**kwargs)
        self.group_norm = tf.keras.layers.GroupNormalization(
            groups=self.groups,
            epsilon=self.epsilon,
            center=self.center,
            scale=self.scale,
            beta_initializer=self.beta_initializer,
            gamma_initializer=self.gamma_initializer,
            beta_regularizer=self.beta_regularizer,
            gamma_regularizer=self.gamma_regularizer,
            beta_constraint=self.beta_constraint,
            gamma_constraint=self.gamma_constraint,
            **kwargs
        )

    def call(self, inputs, training=None):
        return self.group_norm(inputs, training=training)

    def get_config(self):
        config = super(CustomGroupNormalization, self).get_config()
        config.update({
            'groups': self.groups,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': self.beta_initializer,
            'gamma_initializer': self.gamma_initializer,
            'beta_regularizer': self.beta_regularizer,
            'gamma_regularizer': self.gamma_regularizer,
            'beta_constraint': self.beta_constraint,
            'gamma_constraint': self.gamma_constraint
        })
        return config