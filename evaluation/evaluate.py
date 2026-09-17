import json
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix,classification_report

def evaluate_classifier(y_true, probabilities):
    probabilities=np.asarray(probabilities)
    pred=np.argmax(probabilities,axis=1)
    m={"accuracy":float(accuracy_score(y_true,pred)),
       "precision":float(precision_score(y_true,pred,zero_division=0)),
       "recall":float(recall_score(y_true,pred,zero_division=0)),
       "f1":float(f1_score(y_true,pred,zero_division=0))}
    if probabilities.shape[1]==2:
        m["roc_auc"]=float(roc_auc_score(y_true,probabilities[:,1]))
    return m,pred

def save_metrics(metrics,path):
    with open(path,"w") as f: json.dump(metrics,f,indent=2)

def print_report(y_true,y_pred):
    print("Confusion matrix:\n",confusion_matrix(y_true,y_pred))
    print(classification_report(y_true,y_pred,zero_division=0))
