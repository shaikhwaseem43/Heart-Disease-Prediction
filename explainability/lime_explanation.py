def create_lime_explainer(X_train, feature_names, class_names):
    from lime.lime_tabular import LimeTabularExplainer
    return LimeTabularExplainer(X_train,feature_names=list(feature_names),
        class_names=list(class_names),mode="classification",
        discretize_continuous=True,random_state=42)
