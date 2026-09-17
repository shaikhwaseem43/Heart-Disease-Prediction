from tensorflow.keras import layers, Model

def build_lstm_encoder(input_shape, embedding_dim=64):
    inp = layers.Input(shape=input_shape)
    x = layers.LSTM(64,return_sequences=True)(inp)
    x = layers.Dropout(.2)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dense(embedding_dim,activation="relu")(x)
    return Model(inp,x,name="LSTM_Encoder")
