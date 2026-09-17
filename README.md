# CardioRisk-XAI
Executable academic prototype for cardiovascular risk prediction.

Features:
- CSV clinical-data preprocessing
- CNN + LSTM hybrid model
- Feature-level fusion architecture
- Accuracy, precision, recall, F1 and ROC-AUC
- SHAP and LIME integration wrappers
- Early stopping/checkpointing
- Synthetic end-to-end demo

Run:
pip install -r requirements.txt
python main.py --demo

For real data:
python main.py --csv data/heart.csv --target target

IMPORTANT: The demo uses synthetic data only for software testing. It is not a research result or clinically validated system. Do not use identifiable patient data.
