# Protein Classification Challenge

# Protein Function Classification – Full Project Plan

## Single Model Training

| Model | Flow |
|:---|:---|
| Random Forest | Baseline Fit → RandomizedSearchCV (5-Fold) → GridSearchCV (5-Fold) → Optuna Bayesian Optimization → Save Best Model → Predict Evaluation Set |
| XGBoost | Baseline Fit → RandomizedSearchCV (5-Fold) → GridSearchCV (5-Fold) → Optuna Bayesian Optimization → Save Best Model → Predict Evaluation Set |
| Logistic Regression (Lasso) | Train with L1 Regularization → Feature Selection → Retrain if necessary → Predict Evaluation Set |
| MLP Neural Network (GPU) | Baseline Fit → Early Stopping → RandomizedSearchCV or Optuna Bayesian Optimization → Save Best Model → Predict Evaluation Set |
| TabNet (GPU) | Baseline Fit → Early Stopping → Optuna Bayesian Optimization → Save Best Model → Predict Evaluation Set |

## Ensemble Building

| Ensemble Type | Flow |
|:---|:---|
| Soft Voting Ensemble | Aggregate predictions from RF, XGB, LR, MLP, TabNet → Predict Evaluation Set |
| Stacking Ridge Meta-Learner | Out-of-Fold Predictions from RF, XGB, LR, MLP, TabNet → Ridge Regression → Predict Evaluation Set |
| Stacking LightGBM Meta-Learner | Out-of-Fold Predictions from RF, XGB, LR, MLP, TabNet → LightGBM → Predict Evaluation Set |
| Stacking Bayesian Ridge Meta-Learner | Out-of-Fold Predictions → Bayesian Ridge → Predict Evaluation Set |

## Feature Optimization

| Task | Method |
|:---|:---|
| Ablation Studies | Systematically remove feature groups (Base, Pfam, ProtLearn, Peptides, ProtBERT, ESM2) and measure performance drop |
| Lasso Feature Pruning | Use Logistic Regression L1 regularization to select important features and prune the feature space |

## Explainability

| Task | Model |
|:---|:---|
| SHAP Values | Random Forest, XGBoost, Stacking Ensemble (optional) |
| Feature Importance Plots | Top 20 Features for each model |

## Validation and Statistical Proof

| Task | Methodology |
|:---|:---|
| Cross-Validation Score Monitoring | 5-Fold CV used in every tuning step, confirm stability of folds |
| Fold Variance Analysis | Check if standard deviation across folds is low |
| McNemar's Test | Statistically test if two models are significantly different |
| Bootstrap Resampling | Calculate 95% confidence intervals for Accuracy and F1 |
| Confusion Matrices | Generate and plot for each major model and ensemble |

## Engineering Practices

| Feature | Implementation |
|:---|:---|
| Checkpointing | Save models after each tuning step as `.joblib` files |
| Automatic Loading | Check if model exists before retraining |
| Time Tracking | Decorator `@track_time` for each major operation |
| Progress Bars | `tqdm` to show model training and tuning progress |
| Parallelization | `n_jobs=-1` used in RandomizedSearchCV, GridSearchCV, cross_val_score, and model fitting |
| Print Statements | Clearly indicate which stage of training, tuning, or evaluation is running |

## Submission Plan

| Submission Type | Deliverables |
|:---|:---|
| Single Best Models | Random Forest, XGBoost, Logistic Regression, MLP, TabNet |
| Ensemble Models | Soft Voting, Stacking Ridge, Stacking LightGBM, Stacking Bayesian Ridge |
| Total CSVs | 6–8 Submission Files |
| Format | Columns: 'Entry', 'ProteinClass' |

## Final Execution Order

1. Single Models: RF → XGB → LR → MLP → TabNet  
2. Ensemble Models: Soft Voting → Stacking Ridge → Stacking LightGBM → Stacking Bayesian Ridge  
3. Feature Optimization: Ablation Studies → Lasso Feature Pruning  
4. Explainability: SHAP values and feature importance plots  
5. Validation: Cross-validation monitoring, Fold variance analysis, McNemar’s Test, Bootstrap Resampling, Confusion Matrices  
6. Submission: Save 6–8 formatted CSVs for competition
