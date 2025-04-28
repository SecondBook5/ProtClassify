# Protein Classification Challenge

A two-layered, data-driven strategy for protein function classification that balances interpretability, non-linear representation, and ensemble robustness.

---

## Layer 1: Feature Compression & Meta-Learning

**Goal:** Reduce 4,922 original features into compact, complementary representations, then learn how to weight them per sample.

| Pathway                 | Method                          | Output                         | Strength                                              |
|-------------------------|---------------------------------|--------------------------------|-------------------------------------------------------|
| Linear Compression      | PCA (retain 99% variance)       | Principal components           | Noise reduction, preserves global variance            |
| Sparse Selection        | Lasso logistic (data-driven C)  | Subset of original features    | Hard feature pruning, biological interpretability     |
| Bayesian Selection      | MCMC feature selection          | Probabilistic sparse subset    | Uncertainty quantification, preserves original axes   |
| Nonlinear Compression   | VAE (Variational Autoencoder)   | Learned latent factors         | Captures complex manifolds                            |

All four compressed representations are concatenated into a single meta-feature matrix.  
A TabNet meta-learner is trained on these meta-features to soft-select and attend to the best feature space for each protein.

---

## Layer 2: Single-Model Training & Final Ensembles

**Base Models:** trained on TabNet-selected features from Layer 1  

| Model                       | Pipeline                                                                                             |
|-----------------------------|------------------------------------------------------------------------------------------------------|
| Random Forest               | Baseline → RandomizedSearchCV → GridSearchCV → Optuna → Save → Predict                                |
| XGBoost                     | Baseline → RandomizedSearchCV → GridSearchCV → Optuna → Save → Predict                                |
| Logistic Regression (Lasso) | L1-penalized → Feature selection → Retrain → Predict                                                  |
| MLP Neural Network (GPU)    | Baseline → Early stopping → Optuna → Save → Predict                                                  |
| TabNet (GPU)                | Baseline → Early stopping → Optuna → Save → Predict                                                  |

**Final Ensembles:**  
- **Soft Voting:** average predicted probabilities of all base models  
- **Stacking (LightGBM meta-learner):** learn optimal combination of base predictions  

Why both?  
Soft voting provides a stable baseline; stacking can squeeze extra accuracy by learning when to trust each model.

---

## End-to-End Workflow

1. **Data Loading & Preprocessing**  
2. **Layer 1 – Dimensionality Reduction & Meta-Learning**  
   - Fit PCA, Lasso, MCMC selector, VAE  
   - Concatenate outputs → TabNet meta-learner → TabNet-selected features  
3. **Layer 2 – Model Training**  
   - Train RF, XGB, LR, MLP, TabNet on TabNet-selected features  
   - Hyperparameter tuning via RandomizedSearchCV → GridSearchCV → Optuna  
4. **Ensembling**  
   - Generate soft-voting and stacking predictions  
5. **Validation & Explainability**  
   - 5-fold CV monitoring, fold-variance analysis, McNemar’s test, bootstrap confidence intervals  
   - SHAP and feature importance plots  
6. **Submission**  
   - Save single-model and ensemble CSVs with `Entry, ProteinClass`  

---

## Deliverables

- **Models:** `*.joblib` for each tuned model and Optuna study  
- **Compressed datasets:** PCA, Lasso, MCMC, VAE outputs, TabNet-selected features  
- **Predictions:** `y_pred_*.npy` and formatted CSVs  
- **Reports:** Confusion matrices, classification reports, SHAP plots  

This README captures our updated, layered strategy—leveraging both linear/sparse and nonlinear/manifold views, dynamically fused by TabNet, then ensembled across multiple model paradigms to maximize accuracy and robustness.
