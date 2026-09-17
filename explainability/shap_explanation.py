def explain_tabular_model(predict_fn, background, samples):
    import shap
    explainer = shap.Explainer(predict_fn, background)
    return explainer, explainer(samples)
