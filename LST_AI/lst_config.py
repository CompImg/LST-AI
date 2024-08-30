import numpy as np
import tensorflow as tf
import random
import os

def lst_set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
