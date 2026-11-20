import os
import json
import time

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# Always work from the script's folder so mlruns/, the .pkl and the .json files
# are created next to mlops.py and app.py (Streamlit) can find them.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ----------------------------------------------------------------------------
# 1. Load & clean
# ----------------------------------------------------------------------------
df = pd.read_csv(r"C:\Users\tsrav\Downloads\phone_addiction_dataset (1).csv")
df.info()

# Name and Location are identifiers (2933 and 2726 unique values) - no predictive meaning
df = df.drop(columns=['Name', 'Location'])

# fix the typo in the column name
df = df.rename(columns={'Interllectual_Performance': 'Intellectual_Performance'})

print(df.isnull().sum())
print("Duplicates:", df.duplicated().sum())

df = df.drop_duplicates()
print("Shape:", df.shape)

# ----------------------------------------------------------------------------
# 2. Feature engineering
# ----------------------------------------------------------------------------
ENGINEERED_COLS = ['Social_Media_Ratio', 'Gaming_Ratio', 'Education_Ratio']

df['Social_Media_Ratio'] = df['Time_on_Social_Media'] / (df['Daily_Usage_Hours'] + 1e-5)
df['Gaming_Ratio']       = df['Time_on_Gaming']       / (df['Daily_Usage_Hours'] + 1e-5)
df['Education_Ratio']    = df['Time_on_Education']    / (df['Daily_Usage_Hours'] + 1e-5)

# ----------------------------------------------------------------------------
# 3. Train / test split
# ----------------------------------------------------------------------------
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.2
RANDOM_STATE = 42

X = df.drop(columns='Addiction_Level')
y = df['Addiction_Level']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=TEST_SIZE,
                                                    random_state=RANDOM_STATE)

# ----------------------------------------------------------------------------
# 3. Preprocessing pipelines
# ----------------------------------------------------------------------------
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.to_list()

onehot_cols = ['Gender', 'Phone_Usage_Purpose']

num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

onehot_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    [
        ("num", num_pipeline, num_cols),
        ("ohe", onehot_pipeline, onehot_cols)
    ],
    remainder='drop'
)

# ----------------------------------------------------------------------------
# 4. Model building (Optuna hyperparameter tuning)
# ----------------------------------------------------------------------------
N_TRIALS = 20
cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
optuna.logging.set_verbosity(optuna.logging.WARNING)

fit_times = {}   # Optuna search time per model (seconds)
studies = {}     # name -> optuna Study
models = {}      # name -> fitted best pipeline


# One "builder" per algorithm: takes an Optuna trial, returns an estimator
def build_ridge(trial):
    return Ridge(
        alpha=trial.suggest_float('alpha', 1e-3, 1e3, log=True),
        solver=trial.suggest_categorical('solver', ['auto', 'svd', 'cholesky']),
        random_state=42,
    )


def build_dt(trial):
    return DecisionTreeRegressor(
        max_depth=trial.suggest_int('max_depth', 2, 30),
        min_samples_split=trial.suggest_int('min_samples_split', 2, 50),
        min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 20),
        criterion=trial.suggest_categorical('criterion', ['squared_error', 'absolute_error']),
        random_state=42,
    )


def build_rf(trial):
    return RandomForestRegressor(
        n_estimators=trial.suggest_int('n_estimators', 100, 300, step=50),
        max_depth=trial.suggest_int('max_depth', 5, 30),
        min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
        min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 4),
        random_state=42,
    )


def build_gb(trial):
    return GradientBoostingRegressor(
        n_estimators=trial.suggest_int('n_estimators', 100, 300, step=50),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        max_depth=trial.suggest_int('max_depth', 2, 5),
        subsample=trial.suggest_float('subsample', 0.7, 1.0),
        random_state=42,
    )


def build_svr(trial):
    return SVR(
        kernel='rbf',
        C=trial.suggest_float('C', 0.5, 10, log=True),
        gamma=trial.suggest_float('gamma', 1e-3, 1.0, log=True),
        epsilon=trial.suggest_float('epsilon', 0.01, 0.3),
    )


def make_pipeline(estimator):
    return Pipeline([('preprocessor', preprocessor), ('regressor', estimator)])


def tune(name, build_fn, n_trials=N_TRIALS):
    """Run an Optuna study for one algorithm, then fit the best pipeline."""
    def objective(trial):
        pipe = make_pipeline(build_fn(trial))
        return cross_val_score(pipe, X_train, y_train, scoring='r2', cv=cv, n_jobs=-1).mean()

    print(f"\n--- Optimizing {name} ({n_trials} trials) ---")
    study = optuna.create_study(direction='maximize',
                                sampler=TPESampler(seed=RANDOM_STATE),
                                study_name=name)
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials)
    fit_times[name] = time.time() - t0

    # rebuild the best estimator from the best trial's params and fit on all training data
    best_pipe = make_pipeline(build_fn(optuna.trial.FixedTrial(study.best_params)))
    best_pipe.fit(X_train, y_train)

    studies[name] = study
    models[name] = best_pipe
    print(f"Best CV R2 for {name}: {study.best_value:.4f}")


def report(name):
    model, study = models[name], studies[name]
    pred_tr, pred_te = model.predict(X_train), model.predict(X_test)
    print(f"\n===== {name} =====")
    print("Best Parameters:", study.best_params)
    print("Best CV R2       :", study.best_value)
    print("Training R2 score:", r2_score(y_train, pred_tr))
    print("Testing R2 score :", r2_score(y_test, pred_te))
    print("MAE              :", mean_absolute_error(y_test, pred_te))
    print("RMSE             :", np.sqrt(mean_squared_error(y_test, pred_te)))
    return pred_tr, pred_te


tune("Linear Regression (Ridge)", build_ridge)
tune("Decision Tree", build_dt)
tune("Random Forest", build_rf)
tune("Gradient Boosting", build_gb)
tune("SVR", build_svr)

train_preds, test_preds = {}, {}
for _name in models:
    train_preds[_name], test_preds[_name] = report(_name)

# ----------------------------------------------------------------------------
# 5. Comparison table
# ----------------------------------------------------------------------------
names  = list(models.keys())
tr_pr  = [train_preds[n] for n in names]
te_pr  = [test_preds[n] for n in names]

comparison = pd.DataFrame({
    "Model": names,
    "Training R2": [r2_score(y_train, p) for p in tr_pr],
    "Testing R2":  [r2_score(y_test, p) for p in te_pr],
    "MAE":  [mean_absolute_error(y_test, p) for p in te_pr],
    "RMSE": [np.sqrt(mean_squared_error(y_test, p)) for p in te_pr],
})
comparison["Overfit gap"] = comparison["Training R2"] - comparison["Testing R2"]
print("\n--- Comparison (sorted by Testing R2) ---")
print(comparison.sort_values("Testing R2", ascending=False).reset_index(drop=True).round(4).to_string(index=False))

# ----------------------------------------------------------------------------
# 6. Model serialisation
# ----------------------------------------------------------------------------
# FIX: choose the best model by TESTING R2 (the old code used the first numeric
# column, which was "Training R2" and picked the most overfit model).
best_idx = comparison["Testing R2"].idxmax()
best_name = comparison.loc[best_idx, "Model"]
best_model = models[best_name]
print(f"\nBest model: {best_name} (Testing R2 = {comparison.loc[best_idx, 'Testing R2']:.4f})")

joblib.dump(best_model, 'best_addiction_model.pkl')
print(f"Complete model pipeline ({best_name}) saved successfully!")

# the saved pipeline takes RAW rows (preprocessing is inside it)
loaded = joblib.load('best_addiction_model.pkl')
print(loaded.predict(X_test.head()))

comparison.to_csv('model_comparison.csv', index=False)

# Model sizes (KB)
model_sizes = {}
for _name, _model in models.items():
    _tmp = "_size_check.pkl"
    joblib.dump(_model, _tmp)
    model_sizes[_name] = os.path.getsize(_tmp) / 1024
    os.remove(_tmp)


# Schema file used by the Streamlit app to build the input form
def _py(v):
    return v.item() if hasattr(v, "item") else v


input_numeric = [c for c in num_cols if c not in ENGINEERED_COLS]
schema = {
    "target": "Addiction_Level",
    "target_min": float(y.min()),
    "target_max": float(y.max()),
    "best_model": best_name,
    "all_columns": X.columns.tolist(),
    "engineered": ENGINEERED_COLS,
    "numeric": {
        c: {
            "min": float(X[c].min()),
            "max": float(X[c].max()),
            "median": float(X[c].median()),
            "is_int": bool(pd.api.types.is_integer_dtype(X[c])),
        } for c in input_numeric
    },
    "categorical": {c: sorted(X[c].dropna().astype(str).unique().tolist()) for c in onehot_cols},
    # values for columns the model ignores but still expects in the input frame
    "defaults": {
        c: (_py(X[c].mode().iloc[0]) if not X[c].mode().empty else None)
        for c in X.columns
    },
}
with open('feature_schema.json', 'w') as f:
    json.dump(schema, f, indent=2, default=str)
print("Saved: best_addiction_model.pkl, model_comparison.csv, feature_schema.json")

# ----------------------------------------------------------------------------
# 7. MLOps (MLflow)
# ----------------------------------------------------------------------------
mlflow.set_experiment("Mobile_Addiction_Prediction")


def evaluate_model(model, X_train, y_train, X_test, y_test):
    pred_tr = model.predict(X_train)
    t0 = time.time()
    pred_te = model.predict(X_test)
    test_time = time.time() - t0
    train_r2 = r2_score(y_train, pred_tr)
    test_r2 = r2_score(y_test, pred_te)
    return {
        "train_r2": train_r2,
        "test_r2": test_r2,
        "overfit_gap": train_r2 - test_r2,
        "mae": mean_absolute_error(y_test, pred_te),
        "rmse": np.sqrt(mean_squared_error(y_test, pred_te)),
        "test_time_s": test_time,
    }


LOG_EACH_TRIAL_AS_RUN = False   # True = one nested MLflow run per Optuna trial

model_ids = {name: i for i, name in enumerate(models.keys())}
print("Model IDs:", model_ids)

session = time.strftime("%Y%m%d_%H%M%S")   # groups the runs of one execution
run_ids = {}

# One TOP-LEVEL run per model (no parent run), so every chart in the MLflow UI
# can plot all models directly.
for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name) as run:
        run_ids[model_name] = run.info.run_id
        study = studies[model_name]

        mlflow.set_tags({
            "model_type": model_name,
            "task": "regression",
            "tuner": "optuna",
            "run_type": "final_model",
            "session": session,
            "is_best": str(model_name == best_name),
        })

        # best hyperparameters found by Optuna
        mlflow.log_param("model_name", model_name)
        mlflow.log_params(study.best_params)
        mlflow.log_param("n_trials", len(study.trials))
        mlflow.log_params({"test_size": TEST_SIZE, "random_state": RANDOM_STATE})

        metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
        metrics["cv_r2"] = study.best_value
        metrics["search_time_s"] = fit_times[model_name]
        metrics["model_size_kb"] = model_sizes[model_name]
        metrics["model_id"] = model_ids[model_name]
        mlflow.log_metrics(metrics)

        # Optuna optimisation history (chart "trial_cv_r2" in the MLflow UI)
        best_so_far = -np.inf
        for t in study.trials:
            if t.value is None:
                continue
            best_so_far = max(best_so_far, t.value)
            mlflow.log_metric("trial_cv_r2", t.value, step=t.number)
            mlflow.log_metric("best_cv_r2_so_far", best_so_far, step=t.number)
        _trials_csv = "_optuna_trials.csv"
        study.trials_dataframe().to_csv(_trials_csv, index=False)
        mlflow.log_artifact(_trials_csv, artifact_path="optuna")
        os.remove(_trials_csv)

        # Optional: also log every trial as its own nested run (like MLflowCallback)
        if LOG_EACH_TRIAL_AS_RUN:
            for t in study.trials:
                with mlflow.start_run(run_name=f"trial_{t.number}", nested=True):
                    mlflow.set_tag("run_type", "optuna_trial")
                    mlflow.log_params(t.params)
                    if t.value is not None:
                        mlflow.log_metric("cv_r2", t.value)

        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=X_train.head(3),
            serialization_format="cloudpickle",
            # registered_model_name=f"Addiction_{model_name.replace(' ', '_')}",
        )
        print(model_name, {k: round(float(v), 4) for k, v in metrics.items()})

# Separate summary run: holds the comparison table, schema and chart
with mlflow.start_run(run_name=f"Summary_{session}"):
    mlflow.set_tags({"run_type": "summary", "session": session,
                     "best_model": best_name, "best_run_id": run_ids[best_name]})
    mlflow.log_artifact('model_comparison.csv')
    mlflow.log_artifact('feature_schema.json')
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        plot_df = comparison.set_index("Model")
        plot_df[["Training R2", "Testing R2"]].plot(kind="bar", ax=axes[0], title="R2 (train vs test)")
        plot_df[["MAE", "RMSE"]].plot(kind="bar", ax=axes[1], title="Error (lower is better)")
        plot_df["Overfit gap"].plot(kind="bar", ax=axes[2], color="tab:red", title="Overfit gap")
        for a in axes:
            a.tick_params(axis="x", rotation=30)
            for lbl in a.get_xticklabels():
                lbl.set_ha("right")
        plt.tight_layout()
        mlflow.log_figure(fig, "model_comparison_chart.png")
        plt.close(fig)
    except Exception as e:
        print("Chart skipped:", e)

with open('best_run_id.txt', 'w') as f:
    f.write(run_ids[best_name])

print("\nBest model :", best_name)
print("Best run id:", run_ids[best_name], "(saved to best_run_id.txt)")
print("Session    :", session)
print("\nStart the UI in a separate terminal:  python -m mlflow ui --port 5001")
print("Then open http://127.0.0.1:5001 > Mobile_Addiction_Prediction")
print(f'In the runs filter type:  tags.session = "{session}" and tags.run_type = "final_model"')
