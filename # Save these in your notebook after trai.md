# Save these in your notebook after training
import joblib

joblib.dump(LR, 'models/lr_model.pkl')
joblib.dump(DT, 'models/dt_model.pkl')
joblib.dump(RF, 'models/rf_model.pkl')
joblib.dump(vectorization, 'models/vectorizer.pkl')