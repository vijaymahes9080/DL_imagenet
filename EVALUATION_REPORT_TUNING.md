# 💎 ORIEN: Hyper-Parameter Optimization & Model Selection Report

## 1. Overview
Comprehensive optimization of the Neural Ecosystem's behavioral analysis subsystem.
- **Timestamp**: 2026-04-15 20:04:12
- **Dataset**: behavioral_features_full.csv
- **Models Evaluated**: RandomForest, GradientBoosting, SVM, MLP

---

## 2. Detailed Tuning Results (Every Trial Recorded)
Below are the metrics for each hyper-parameter combination tested during Grid Search.

| Model            | Parameters                                                                  |   Mean_Test_Accuracy |   Mean_Test_AUC_ROC |   Mean_Test_F1 |   Mean_Train_Accuracy |
|:-----------------|:----------------------------------------------------------------------------|---------------------:|--------------------:|---------------:|----------------------:|
| MLP              | {'activation': 'relu', 'alpha': 0.001, 'hidden_layer_sizes': (64,)}         |             0.674277 |            0.744064 |       0.68109  |              0.68417  |
| MLP              | {'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': (64,)}        |             0.674277 |            0.744037 |       0.68109  |              0.684932 |
| MLP              | {'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (32, 32, 32)} |             0.672755 |            0.749819 |       0.677664 |              0.674277 |
| MLP              | {'activation': 'tanh', 'alpha': 0.001, 'hidden_layer_sizes': (32, 32, 32)}  |             0.671233 |            0.749903 |       0.676623 |              0.674277 |
| MLP              | {'activation': 'tanh', 'alpha': 0.001, 'hidden_layer_sizes': (64,)}         |             0.668189 |            0.738004 |       0.690911 |              0.668189 |
| MLP              | {'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (64,)}        |             0.668189 |            0.738004 |       0.690911 |              0.668189 |
| GradientBoosting | {'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 100}                |             0.666667 |            0.739978 |       0.674516 |              0.858447 |
| RandomForest     | {'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 50}               |             0.665145 |            0.722394 |       0.668085 |              0.999239 |
| SVM              | {'C': 10.0, 'gamma': 'scale', 'kernel': 'rbf'}                              |             0.663623 |            0.739978 |       0.665162 |              0.694064 |
| SVM              | {'C': 1.0, 'gamma': 'auto', 'kernel': 'rbf'}                                |             0.663623 |            0.74526  |       0.678016 |              0.679604 |
| RandomForest     | {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 50}             |             0.663623 |            0.722741 |       0.667128 |              0.999239 |
| RandomForest     | {'max_depth': 20, 'min_samples_split': 5, 'n_estimators': 50}               |             0.663623 |            0.719905 |       0.665995 |              0.984779 |
| GradientBoosting | {'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 50}                 |             0.6621   |            0.738546 |       0.666332 |              0.826484 |
| RandomForest     | {'max_depth': None, 'min_samples_split': 5, 'n_estimators': 50}             |             0.6621   |            0.720656 |       0.665053 |              0.984018 |
| RandomForest     | {'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 200}              |             0.660578 |            0.724562 |       0.669423 |              1        |
| RandomForest     | {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 200}            |             0.660578 |            0.724354 |       0.668506 |              1        |
| MLP              | {'activation': 'tanh', 'alpha': 0.001, 'hidden_layer_sizes': (128, 64)}     |             0.660578 |            0.742674 |       0.66742  |              0.680365 |
| MLP              | {'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (128, 64)}    |             0.660578 |            0.742758 |       0.66742  |              0.680365 |
| GradientBoosting | {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50}                  |             0.660578 |            0.735863 |       0.666542 |              0.850076 |
| SVM              | {'C': 10.0, 'gamma': 'auto', 'kernel': 'rbf'}                               |             0.660578 |            0.736447 |       0.663153 |              0.697869 |
| RandomForest     | {'max_depth': 10, 'min_samples_split': 5, 'n_estimators': 100}              |             0.660578 |            0.729552 |       0.668753 |              0.968037 |
| RandomForest     | {'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 200}              |             0.660578 |            0.728941 |       0.667854 |              0.996956 |
| GradientBoosting | {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 100}                |             0.659056 |            0.754712 |       0.664731 |              0.738965 |
| SVM              | {'C': 1.0, 'gamma': 'scale', 'kernel': 'rbf'}                               |             0.659056 |            0.745538 |       0.676007 |              0.679604 |
| RandomForest     | {'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 100}              |             0.659056 |            0.725215 |       0.666455 |              0.995434 |
| RandomForest     | {'max_depth': 10, 'min_samples_split': 5, 'n_estimators': 50}               |             0.657534 |            0.729956 |       0.661449 |              0.965753 |
| RandomForest     | {'max_depth': 10, 'min_samples_split': 5, 'n_estimators': 200}              |             0.656012 |            0.733278 |       0.66217  |              0.97032  |
| RandomForest     | {'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 100}              |             0.656012 |            0.722157 |       0.669256 |              1        |
| RandomForest     | {'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 50}               |             0.65449  |            0.725285 |       0.658521 |              0.990107 |
| RandomForest     | {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 100}            |             0.65449  |            0.722255 |       0.667442 |              1        |
| RandomForest     | {'max_depth': None, 'min_samples_split': 5, 'n_estimators': 200}            |             0.65449  |            0.729052 |       0.660471 |              0.990868 |
| MLP              | {'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': (32, 32, 32)} |             0.65449  |            0.728107 |       0.663248 |              0.70624  |
| GradientBoosting | {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 50}                  |             0.65449  |            0.725549 |       0.664871 |              0.986301 |
| RandomForest     | {'max_depth': 20, 'min_samples_split': 5, 'n_estimators': 200}              |             0.65449  |            0.729052 |       0.660471 |              0.991629 |
| RandomForest     | {'max_depth': None, 'min_samples_split': 5, 'n_estimators': 100}            |             0.652968 |            0.722825 |       0.660674 |              0.987062 |
| RandomForest     | {'max_depth': 20, 'min_samples_split': 5, 'n_estimators': 100}              |             0.652968 |            0.722602 |       0.660674 |              0.987062 |
| MLP              | {'activation': 'relu', 'alpha': 0.001, 'hidden_layer_sizes': (128, 64)}     |             0.652968 |            0.7268   |       0.642127 |              0.703957 |
| MLP              | {'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': (128, 64)}    |             0.651446 |            0.726661 |       0.642186 |              0.703196 |
| GradientBoosting | {'learning_rate': 0.2, 'max_depth': 3, 'n_estimators': 50}                  |             0.651446 |            0.722408 |       0.650234 |              0.923135 |
| MLP              | {'activation': 'relu', 'alpha': 0.001, 'hidden_layer_sizes': (32, 32, 32)}  |             0.651446 |            0.728301 |       0.643769 |              0.705479 |
| GradientBoosting | {'learning_rate': 0.2, 'max_depth': 3, 'n_estimators': 100}                 |             0.648402 |            0.711635 |       0.652001 |              0.99239  |
| GradientBoosting | {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100}                 |             0.645358 |            0.714595 |       0.652092 |              1        |
| GradientBoosting | {'learning_rate': 0.2, 'max_depth': 5, 'n_estimators': 50}                  |             0.643836 |            0.712955 |       0.646337 |              0.999239 |
| GradientBoosting | {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 50}                 |             0.637747 |            0.731526 |       0.653268 |              0.715373 |
| SVM              | {'C': 10.0, 'gamma': 'scale', 'kernel': 'poly'}                             |             0.634703 |            0.712232 |       0.685285 |              0.653729 |
| GradientBoosting | {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100}                 |             0.633181 |            0.722157 |       0.641719 |              0.945205 |
| GradientBoosting | {'learning_rate': 0.2, 'max_depth': 5, 'n_estimators': 100}                 |             0.633181 |            0.705477 |       0.630164 |              1        |
| SVM              | {'C': 10.0, 'gamma': 'auto', 'kernel': 'poly'}                              |             0.630137 |            0.713428 |       0.67095  |              0.670472 |
| SVM              | {'C': 0.1, 'gamma': 'auto', 'kernel': 'rbf'}                                |             0.628615 |            0.736447 |       0.715636 |              0.622527 |
| SVM              | {'C': 0.1, 'gamma': 'scale', 'kernel': 'rbf'}                               |             0.622527 |            0.73831  |       0.713812 |              0.617199 |
| SVM              | {'C': 1.0, 'gamma': 'auto', 'kernel': 'poly'}                               |             0.570776 |            0.731109 |       0.696807 |              0.579148 |
| SVM              | {'C': 1.0, 'gamma': 'scale', 'kernel': 'poly'}                              |             0.570776 |            0.73047  |       0.696807 |              0.579148 |
| SVM              | {'C': 0.1, 'gamma': 'auto', 'kernel': 'poly'}                               |             0.564688 |            0.730136 |       0.693816 |              0.570015 |
| SVM              | {'C': 0.1, 'gamma': 'scale', 'kernel': 'poly'}                              |             0.560122 |            0.725688 |       0.691621 |              0.563927 |

---

## 3. Optimal Parameter Configuration
Selected based on the highest validation accuracy during k-fold cross-validation.

| Model | Optimal Parameters | Best CV Accuracy |
| :--- | :--- | :--- |
| **RandomForest** | `{'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 'max_depth': 20, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': 50, 'n_jobs': None, 'oob_score': False, 'random_state': 42, 'verbose': 0, 'warm_start': False}` | **0.6651** |
| **GradientBoosting** | `{'ccp_alpha': 0.0, 'criterion': 'friedman_mse', 'init': None, 'learning_rate': 0.01, 'loss': 'log_loss', 'max_depth': 5, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 100, 'n_iter_no_change': None, 'random_state': 42, 'subsample': 1.0, 'tol': 0.0001, 'validation_fraction': 0.1, 'verbose': 0, 'warm_start': False}` | **0.6667** |
| **SVM** | `{'C': 1.0, 'break_ties': False, 'cache_size': 200, 'class_weight': None, 'coef0': 0.0, 'decision_function_shape': 'ovr', 'degree': 3, 'gamma': 'auto', 'kernel': 'rbf', 'max_iter': -1, 'probability': True, 'random_state': 42, 'shrinking': True, 'tol': 0.001, 'verbose': False}` | **0.6636** |
| **MLP** | `{'activation': 'relu', 'alpha': 0.0001, 'batch_size': 'auto', 'beta_1': 0.9, 'beta_2': 0.999, 'early_stopping': False, 'epsilon': 1e-08, 'hidden_layer_sizes': (64,), 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'max_fun': 15000, 'max_iter': 1000, 'momentum': 0.9, 'n_iter_no_change': 10, 'nesterovs_momentum': True, 'power_t': 0.5, 'random_state': 42, 'shuffle': True, 'solver': 'adam', 'tol': 0.0001, 'validation_fraction': 0.1, 'verbose': False, 'warm_start': False}` | **0.6743** |

---

## 4. Final Performance & Fit Analysis
Evaluation on the unseen hold-out test set.

| Model            |   Train_Accuracy |   Test_Accuracy |   AUC_ROC |   F1_Score |   Fit_Gap | Status         |
|:-----------------|-----------------:|----------------:|----------:|-----------:|----------:|:---------------|
| RandomForest     |         1        |        0.648485 |  0.730238 |   0.658824 | 0.351515  | Overfitting ⚠️ |
| GradientBoosting |         0.821918 |        0.654545 |  0.72921  |   0.645963 | 0.167372  | Overfitting ⚠️ |
| SVM              |         0.680365 |        0.660606 |  0.712313 |   0.678161 | 0.0197592 | Optimized ✅   |
| MLP              |         0.680365 |        0.648485 |  0.70908  |   0.658824 | 0.0318804 | Optimized ✅   |

### 🔍 Fit Diagnosis
- **Winner**: SVM
- **Fit Status**: Optimized ✅
- **Accuracy Gap**: 0.0198

---

## 5. Advanced Finetuning (Step 7)
Applied **CalibratedClassifierCV** to the SVM model.
- **Original Accuracy**: 0.6606
- **Calibrated Accuracy**: 0.6545
- **Implementation**: Applied CalibratedClassifierCV to enhance reliability of behavioral probability scores.

---

## 6. XAI & Ablation Study (Step 9)

### 🧠 Explainable AI (SHAP)
SHAP analysis completed. Top features identified.
**Top contributing features to behavioral classification:**
1. `density`
2. `dist_total`
3. `vel_avg`

### 🔬 Ablation Analysis
Verification of feature necessity by systematic removal.
- **Baseline Accuracy**: 0.6606
- **Post-Ablation Accuracy**: 0.4970
- **Accuracy Drop**: 0.1636
- **Features Removed**: density, dist_total

---

## 7. Final Recommendation
The **SVM** model is recommended for production deployment due to its superior generalization and stability.

