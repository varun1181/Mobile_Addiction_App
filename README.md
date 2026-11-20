📱 Mobile Addiction Prediction

Predicts a person's phone addiction level from daily habits, sleep, mood and screen-time data. Five regression models are tuned with Optuna, tracked with MLflow, and the best one (Gradient Boosting) is served through a Streamlit web app.

Live app: https://mobileaddictionapp-bj92ms7gn6wmq5sdjm9q6b.streamlit.app

Results

Models were tuned with 20 Optuna trials each (3-fold cross-validation) and scored on a held-out 20% test set.

Model	Test R²	MAE	RMSE	Overfit gap
Gradient Boosting	0.982	0.135	0.219	0.016
SVR	0.964	0.199	0.310	0.012
Random Forest	0.952	0.218	0.359	0.042
Decision Tree	0.863	0.332	0.605	0.101
Linear Regression (Ridge)	0.754	0.647	0.809	-0.012

Gradient Boosting was chosen: it has the best test R² and the lowest error, and its cross-validated R² (0.971) is close to its test R², so the score is reliable. The weak Ridge result shows the relationship between the features and the addiction level is not linear.

Dataset
6,000 rows and 22 columns, 5,968 rows after removing duplicates.
Target: Addiction_Level (numeric).
Features: age, gender, daily and weekend phone hours, phone checks per day, apps used, screen time before bed, time on social media / gaming / education, sleep, exercise, anxiety, depression, self-esteem, social interactions, family communication and the main purpose of phone use.
Name and Location are dropped because they are identifiers.
Three features are engineered: social media, gaming and education time as a share of daily usage.

The dataset itself is not included in this repository.

Project structure
├── app.py                      # Streamlit web app
├── mlops.py                    # training, tuning and MLflow tracking
├── best_addiction_model.pkl    # saved Gradient Boosting pipeline
├── feature_schema.json         # input ranges and defaults used by the app
├── requirements.txt            # package versions
└── .streamlit/
    └── config.toml             # app theme

mlops.py also writes model_comparison.csv and best_run_id.txt. The mlruns/ folder (or mlflow.db) holds the MLflow history and is not needed to run the app.

How it works
Preprocessing: median imputation and scaling for numbers, most-frequent imputation and one-hot encoding for categories, all inside one scikit-learn pipeline. The saved model accepts raw rows.
Tuning: one Optuna study per algorithm (TPE sampler, 20 trials, R² as the objective).
Tracking: every model is logged to MLflow with its best parameters, train/test R², CV R², MAE, RMSE, overfit gap, timings, model size and the Optuna trial history.
Selection: the model with the highest test R² is saved as best_addiction_model.pkl.
App: Streamlit loads the saved pipeline, builds the form from feature_schema.json, and shows the predicted level on a gauge with short tips.
Run it locally

1. Install the packages

bash
pip install pandas numpy scikit-learn optuna mlflow joblib streamlit matplotlib

2. Train and track the models (edit the CSV path near the top of mlops.py first)

bash
python mlops.py

3. Explore the runs in MLflow (run this in a separate terminal from the project folder)

bash
python -m mlflow ui --port 5001

Open http://127.0.0.1:5001, choose the Mobile_Addiction_Prediction experiment, filter with tags.run_type = "final_model", and sort by test_r2.

4. Start the app

bash
python -m streamlit run app.py

Open http://localhost:8501.

Deploy on Streamlit Community Cloud
Put app.py, best_addiction_model.pkl, feature_schema.json, requirements.txt and .streamlit/config.toml in a GitHub repository.
On share.streamlit.io, create an app from the repo with app.py as the main file.
In Advanced settings, choose Python 3.12 or newer. The pinned package versions need it.
Keep the scikit-learn version in requirements.txt equal to the one used for training, otherwise the model may fail to load.
Notes
The app gives a model estimate, not a medical diagnosis.
The model file was trained on a specific scikit-learn version. Retrain with mlops.py after changing versions.
Tech stack

Python · pandas · NumPy · scikit-learn · Optuna · MLflow · Streamlit
