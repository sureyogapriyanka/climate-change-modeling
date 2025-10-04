"""
models.py — trains, evaluates, and compares models
"""

import pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path

def train_and_evaluate(X, y, test_size=0.2, random_state=42):
    """Train multiple regression models and evaluate performance"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=random_state, n_estimators=100),
        "Gradient Boosting": GradientBoostingRegressor(random_state=random_state)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results[name] = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": mean_squared_error(y_test, y_pred, squared=False),
            "R2": r2_score(y_test, y_pred)
        }
        print(f"\n✅ {name} Results:")
        for metric, val in results[name].items():
            print(f"{metric}: {val:.4f}")

    return results


def save_results(results: dict, out_csv="reports/tables/model_results.csv"):
    """Save model results to CSV"""
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results).T
    results_df.to_csv(out_csv)
    print(f"💾 Model results saved to: {out_csv}")
    return results_df


def plot_results(results_df: pd.DataFrame, out_path="reports/figures/model_r2.png"):
    """Create bar chart of R² scores"""
    plt.figure(figsize=(8,5))
    sns.barplot(data=results_df.reset_index(), x="index", y="R2")
    plt.title("Model Comparison (R²)")
    plt.ylabel("R² Score")
    plt.xlabel("Model")
    plt.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    print(f"📊 Plot saved to: {out_path}")
    plt.show()
