import argparse
from pathlib import Path
import numpy as np
from preprocessing.preprocessing import load_csv,split_scale,tabular_to_sequence,make_synthetic_tabular
from models.hybrid_cnn_lstm import build_hybrid_cnn_lstm
from training.train import train_model
from evaluation.evaluate import evaluate_classifier,save_metrics,print_report

def run_demo(epochs=5):
    Xdf,y=make_synthetic_tabular()
    Xtr,Xte,ytr,yte,_=split_scale(Xdf,y)
    Xtr,Xte=tabular_to_sequence(Xtr),tabular_to_sequence(Xte)
    model=build_hybrid_cnn_lstm(Xtr.shape[1:],2)
    model,_=train_model(model,Xtr,ytr,Xte,yte,epochs=epochs)
    probs=model.predict(Xte,verbose=0)
    metrics,pred=evaluate_classifier(yte,probs)
    save_metrics(metrics,"outputs/demo_metrics.json")
    model.save("outputs/demo_hybrid_cnn_lstm.keras")
    print("DEMO METRICS (synthetic data; NOT research results):",metrics)
    print_report(yte,pred)

def run_csv(path,target,epochs=30):
    Xdf,y=load_csv(path,target)
    Xtr,Xte,ytr,yte,_=split_scale(Xdf,y)
    Xtr,Xte=tabular_to_sequence(Xtr),tabular_to_sequence(Xte)
    model=build_hybrid_cnn_lstm(Xtr.shape[1:],len(np.unique(y)))
    model,_=train_model(model,Xtr,ytr,Xte,yte,epochs=epochs)
    probs=model.predict(Xte,verbose=0)
    metrics,pred=evaluate_classifier(yte,probs)
    save_metrics(metrics,"outputs/test_metrics.json")
    model.save("outputs/hybrid_cnn_lstm.keras")
    print("TEST METRICS:",metrics)
    print_report(yte,pred)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--demo",action="store_true")
    p.add_argument("--csv")
    p.add_argument("--target",default="target")
    p.add_argument("--epochs",type=int,default=30)
    a=p.parse_args()
    Path("outputs").mkdir(exist_ok=True)
    if a.demo: run_demo(a.epochs)
    elif a.csv: run_csv(a.csv,a.target,a.epochs)
    else: p.error("Use --demo or --csv PATH --target COLUMN")
