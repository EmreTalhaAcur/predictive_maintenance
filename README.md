# Predictive Maintenance Classification Model

This project implements a **Machine Learning Classification Pipeline** utilizing sensor data (Temperature, Vibration, Operation Hours) to predict upcoming machine/engine failures (Healthy vs. Failure Impending).

## 🧠 Engineering Methodology & Best Practices
* **Data Leakage Prevention:** Rigorously implemented a **Train / Test Split (80% - 20%)** routine to evaluate model generalization on completely unseen data.
* **Validation (5-Fold Cross-Validation):** Evaluated model stability and measured variance by subjecting the training set to a 5-Fold cross-validation strategy, ensuring performance consistency.
* **Realistic Noise Injection:** Added **15% random noise** to the synthetic rule set to prevent the algorithm from directly memorizing patterns (Overfitting). This forced the Random Forest model to generalize effectively, resulting in a realistic and robust ~82-84% accuracy band rather than an overfitted 100% score.

## 📊 Evaluation Metrics (Classification Report)
Beyond raw accuracy, the project heavily evaluates industrial trade-offs by analyzing the balance between **Precision** and **Recall** to optimize predictive performance against maintenance costs.

## 🛠️ Tech Stack
* Python 3
* Scikit-Learn (RandomForestClassifier)
* Pandas
* NumPy
