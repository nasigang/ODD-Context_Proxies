# 05. Model Training and Frozen Model Checkpoints

Contains training pipelines and the sealed, frozen model artifacts for all nested candidate architectures.

## Model Family (`HistGradientBoostingClassifier`, scikit-learn 1.7.1)
- Hyperparameters: `max_iter=100`, `max_depth=6`, `learning_rate=0.1`, scenario-weight balancing ($w_{s,t} = 1 / n_s$).

## Frozen Checkpoints (`models_frozen/`)
- `M_P.pkl`: Physical baseline model ($P_{\text{clean}}$, 12 features).
- `M_E.pkl`: Context-only model ($E_{\text{all}}$, 17 features).
- `M_P_Estatic.pkl`: Static geometry model ($P_{\text{clean}} + E_{\text{static}}$, 17 features).
- `M_P_Ecomp.pkl`: Actor composition model ($P_{\text{clean}} + E_{\text{comp}}$, 18 features).
- `M_P_Einteract.pkl`: Secondary core model ($P_{\text{clean}} + E_{\text{interact}}$, 18 features).
- `M_P_Eall.pkl`: Primary full model ($P_{\text{clean}} + E_{\text{all}}$, 29 features).
- `M_P_Eall_Ehist.pkl`: Extended history variant.
- `nuisance_models.pkl`: Residualization and nuisance feature estimators.
- `model_and_preproc_manifest.json`: Checksum and feature metadata manifest.
