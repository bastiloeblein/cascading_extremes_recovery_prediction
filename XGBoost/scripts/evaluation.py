import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns


def get_metrics(df, target_var):
    y_true = df[target_var]
    y_pred = df["preds"]
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    bias = np.mean(y_pred - y_true)
    return pd.Series({"RMSE": rmse, "MAE": mae, "Bias": bias, "Count": len(df)})


def evaluate_with_gap_analysis(df_results, target_var, ndvi_input):
    df_clean = df_results.dropna(subset=[target_var]).copy()

    # Split in Clear Sky (Input vorhanden) und Cloudy (Input NaN)
    df_clear = df_clean[df_clean[ndvi_input].notna()]
    df_cloudy = df_clean[df_clean[ndvi_input].isna()]

    m_clear = (
        get_metrics(df_clear, target_var)
        if len(df_clear) > 0
        else pd.Series({"RMSE": np.nan})
    )
    m_cloudy = (
        get_metrics(df_cloudy, target_var)
        if len(df_cloudy) > 0
        else pd.Series({"RMSE": np.nan})
    )

    return m_clear, m_cloudy


def plot_feature_importance(model_package, top_n=20):
    """
    Plottet die Feature Importance in absteigender Reihenfolge und
    gibt die exakten Werte als Tabelle aus.
    """
    model = model_package["model"]
    features = model_package["features"]
    model_name = model_package.get("model_name", "Model")

    # Importance aus dem XGBRegressor extrahieren
    importances = model.feature_importances_

    # DataFrame erstellen und sortieren
    fi_df = (
        pd.DataFrame({"Feature": features, "Importance_Gain": importances})
        .sort_values(by="Importance_Gain", ascending=False)
        .reset_index(drop=True)
    )

    # 1. Plotten
    plt.figure(figsize=(10, 0.4 * min(len(fi_df), top_n) + 2))
    ax = sns.barplot(
        x="Importance_Gain",
        y="Feature",
        data=fi_df.head(top_n),
        palette="magma",
        hue="Feature",
        legend=False,
    )

    # Werte direkt an die Balken schreiben
    for p in ax.patches:
        width = p.get_width()
        ax.text(
            width + 0.002,
            p.get_y() + p.get_height() / 2,
            f"{width:.4f}",
            va="center",
            fontsize=10,
        )

    plt.title(f"Feature Importance (Sorted by Gain) - {model_name}", fontsize=14)
    plt.xlabel("Normalized Importance Score")
    plt.ylabel("Features")
    plt.xlim(0, fi_df["Importance_Gain"].max() * 1.15)  # Platz für Text rechts
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    # 2. Tabellarische Ausgabe
    print(f"\n--- Feature Importance Liste: {model_name} ---")
    print(
        fi_df.head(top_n).to_string(
            index=False, formatters={"Importance_Gain": "{:,.4f}".format}
        )
    )

    return fi_df
