import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sys 
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    C_param = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    solver_param = sys.argv[2] if len(sys.argv) > 2 else "liblinear"
    dataset_path = sys.argv[3] if len(sys.argv) > 3 else "heart_disease_preprocessing.csv"
    
    print(f"Running with parameters: C={C_param}, solver='{solver_param}', dataset='{dataset_path}'")

    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"Error: File dataset '{dataset_path}' tidak ditemukan.")
        exit()
    
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():        
        model = LogisticRegression(C=C_param, solver=solver_param, max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)                
        print(f"Accuracy: {accuracy}")
        
        mlflow.log_param("C", C_param)
        mlflow.log_param("solver", solver_param)
        mlflow.log_metric("accuracy", accuracy)        
        mlflow.sklearn.log_model(model, "model")
        
        print("\nModel dan metrik berhasil di-log ke MLflow.")