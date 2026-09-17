import numpy as np
from tensorflow.keras import layers

def concatenate_features(*features):
    return np.concatenate(features,axis=1)

def keras_feature_fusion(*tensors):
    return layers.Concatenate(name="feature_fusion")(list(tensors))
