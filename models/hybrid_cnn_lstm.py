from tensorflow.keras import layers, Model

def build_hybrid_cnn_lstm(input_shape, n_classes=2):
    inp = layers.Input(shape=input_shape,name="hybrid_input")
    x = layers.Conv1D(64,3,padding="same",activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(.2)(x)
    x = layers.Conv1D(128,3,padding="same",activation="relu")(x)
    x = layers.MaxPooling1D(2,padding="same")(x)
    x = layers.LSTM(64,return_sequences=True)(x)
    x = layers.Dropout(.2)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dense(64,activation="relu",name="hybrid_embedding")(x)
    x = layers.Dropout(.3)(x)
    out = layers.Dense(n_classes,activation="softmax",name="prediction")(x)
    model = Model(inp,out,name="Hybrid_CNN_LSTM")
    model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    return model

def build_multimodal_fusion_model(tabular_shape, signal_shape, n_classes=2):
    tab = layers.Input(shape=tabular_shape,name="tabular_input")
    sig = layers.Input(shape=signal_shape,name="signal_input")
    t = layers.Conv1D(32,3,padding="same",activation="relu")(tab)
    t = layers.GlobalAveragePooling1D()(t)
    t = layers.Dense(64,activation="relu")(t)
    s = layers.Conv1D(32,7,padding="same",activation="relu")(sig)
    s = layers.BatchNormalization()(s)
    s = layers.MaxPooling1D(2,padding="same")(s)
    s = layers.LSTM(64)(s)
    s = layers.Dense(64,activation="relu")(s)
    fused = layers.Concatenate(name="feature_fusion")([t,s])
    fused = layers.Dense(128,activation="relu")(fused)
    fused = layers.Dropout(.3)(fused)
    fused = layers.Dense(64,activation="relu")(fused)
    out = layers.Dense(n_classes,activation="softmax")(fused)
    model = Model([tab,sig],out,name="Multimodal_CNN_LSTM_Fusion")
    model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    return model
