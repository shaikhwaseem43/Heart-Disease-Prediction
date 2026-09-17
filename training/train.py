from pathlib import Path
import json
import tensorflow as tf

def train_model(model,X_train,y_train,X_val=None,y_val=None,epochs=30,batch_size=32,output_dir="outputs"):
    Path(output_dir).mkdir(exist_ok=True)
    monitor="val_loss" if X_val is not None else "loss"
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor=monitor,patience=5,restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(Path(output_dir)/"best_model.keras"),monitor=monitor,save_best_only=True)
    ]
    history=model.fit(X_train,y_train,validation_data=((X_val,y_val) if X_val is not None else None),
                      epochs=epochs,batch_size=batch_size,callbacks=callbacks,verbose=1)
    with open(Path(output_dir)/"history.json","w") as f:
        json.dump({k:[float(v) for v in vals] for k,vals in history.history.items()},f,indent=2)
    return model,history
