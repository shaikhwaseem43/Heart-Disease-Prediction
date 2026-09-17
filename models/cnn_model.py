from tensorflow.keras import layers, Model

def build_cnn_encoder(input_shape, embedding_dim=64):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv1D(32,3,padding="same",activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2,padding="same")(x)
    x = layers.Conv1D(64,3,padding="same",activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(embedding_dim,activation="relu")(x)
    return Model(inp,x,name="CNN_Encoder")
