import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import shap

st.set_page_config(page_title="Inflation Forecasting Dashboard", layout="wide")

# =========================
# STYLE
# =========================
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
      .kpi-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 14px 16px;
      }
      .muted { color: rgba(255,255,255,0.65); font-size: 0.9rem; }
      .small { font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 Inflation Forecasting (CPI) + Financial News Sentiment")
st.markdown(
    "<div class='muted'>Walk-forward backtesting • XGBoost regression • Optional SHAP explainability • Downloadable forecasts</div>",
    unsafe_allow_html=True
)
st.write("")


# =========================
# HELPERS
# =========================
@st.cache_data(show_spinner=False)
def load_cpi(start_date: str = "2015-01-01") -> pd.DataFrame:
    # FRED direct CSV endpoint
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"
    cpi = pd.read_csv(url)
    cpi.columns = ["date", "cpi"]
    cpi["date"] = pd.to_datetime(cpi["date"])
    cpi = cpi[cpi["date"] >= pd.to_datetime(start_date)].copy()

    # YoY inflation
    cpi["inflation_yoy"] = cpi["cpi"].pct_change(12, fill_method=None)
    cpi = cpi.dropna().sort_values("date").reset_index(drop=True)
    return cpi


@st.cache_data(show_spinner=False)
def load_monthly_sentiment(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    if "date" not in df.columns or "sentiment" not in df.columns:
        raise ValueError("CSV must have columns: date, sentiment")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def make_features(df: pd.DataFrame, horizon: int, n_lags: int):
    df = df.copy().sort_values("date").reset_index(drop=True)

    # Predict future inflation
    df["target"] = df["inflation_yoy"].shift(-horizon)

    # Lags
    for l in range(1, n_lags + 1):
        df[f"infl_lag{l}"] = df["inflation_yoy"].shift(l)
        df[f"sentiment_lag{l}"] = df["sentiment"].shift(l)

    df = df.dropna().reset_index(drop=True)

    feature_cols = (
        ["sentiment"]
        + [f"sentiment_lag{l}" for l in range(1, n_lags + 1)]
        + [f"infl_lag{l}" for l in range(1, n_lags + 1)]
    )

    X = df[feature_cols]
    y = df["target"]
    return df, X, y, feature_cols


def walk_forward_oof(X, y, params, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof = np.zeros(len(y))

    for train_idx, test_idx in tscv.split(X):
        model = xgb.XGBRegressor(**params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[test_idx] = model.predict(X.iloc[test_idx])

    return oof


@st.cache_data(show_spinner=False)
def run_backtest_cached(X_df: pd.DataFrame, y_ser: pd.Series, params: dict, n_splits: int):
    """
    Cache backtest results so Streamlit Cloud doesn't retrain on every rerun.
    """
    oof = walk_forward_oof(X_df, y_ser, params=params, n_splits=n_splits)
    mae = mean_absolute_error(y_ser, oof)
    return oof, mae


@st.cache_resource(show_spinner=False)
def train_final_model_cached(X_df: pd.DataFrame, y_ser: pd.Series, params: dict):
    """
    Cache trained model object (resource cache).
    """
    m = xgb.XGBRegressor(**params)
    m.fit(X_df, y_ser)
    return m


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("Data")
    sentiment_file = st.file_uploader(
        "Upload monthly sentiment CSV (columns: date, sentiment)",
        type=["csv"]
    )
    start_date = st.text_input("FRED start date", value="2015-01-01")

    st.header("Forecast Setup")
    horizon = st.slider("Forecast horizon (months ahead)", 1, 12, 1)
    n_lags = st.slider("Number of lags (inflation + sentiment)", 0, 12, 1)
    n_splits = st.slider("Walk-forward folds", 3, 10, 5)

    st.header("XGBoost Params")
    n_estimators = st.slider("n_estimators", 100, 2000, 500, step=50)
    max_depth = st.slider("max_depth", 2, 10, 3)
    learning_rate = st.number_input("learning_rate", 0.01, 0.3, 0.05, step=0.01)
    subsample = st.slider("subsample", 0.5, 1.0, 0.9)
    colsample_bytree = st.slider("colsample_bytree", 0.5, 1.0, 0.9)
    random_state = st.number_input("random_state", value=42, step=1)

    st.header("Explainability")
    run_shap = st.checkbox("Compute SHAP (slower on cloud)", value=False)

    run_btn = st.button("🚀 Run backtest")


# =========================
# MAIN
# =========================
if not sentiment_file:
    st.info("Upload your **monthly sentiment CSV** to begin.")
    st.stop()

if not run_btn:
    st.warning("Adjust settings in the sidebar and click **Run backtest**.")
    st.stop()


# -------------------------
# Load + merge data
# -------------------------
with st.spinner("Loading CPI (FRED) + sentiment and building features..."):
    cpi = load_cpi(start_date=start_date)
    sent = load_monthly_sentiment(sentiment_file)

    df = pd.merge(
        cpi[["date", "inflation_yoy"]],
        sent[["date", "sentiment"]],
        on="date",
        how="inner"
    ).dropna().sort_values("date").reset_index(drop=True)

    df_feat, X, y, feature_cols = make_features(df, horizon=horizon, n_lags=n_lags)

st.success(f"Dataset ready ✅ Rows: {len(df_feat)} | Features: {len(feature_cols)}")
st.write("**Features:**", feature_cols)

params = dict(
    n_estimators=n_estimators,
    max_depth=max_depth,
    learning_rate=learning_rate,
    subsample=subsample,
    colsample_bytree=colsample_bytree,
    objective="reg:squarederror",
    random_state=int(random_state),
)

# -------------------------
# Walk-forward backtest (CACHED)
# -------------------------
with st.spinner("Running walk-forward validation..."):
    oof, mae = run_backtest_cached(X, y, params=params, n_splits=n_splits)

# -------------------------
# Train final model (CACHED)
# -------------------------
final_model = train_final_model_cached(X, y, params)

# -------------------------
# KPI cards
# -------------------------
k1, k2, k3 = st.columns(3)

with k1:
    st.markdown(
        f"<div class='kpi-card'><div class='muted'>Walk-forward OOF MAE</div>"
        f"<div style='font-size:1.6rem; font-weight:700'>{mae:.5f}</div></div>",
        unsafe_allow_html=True
    )

with k2:
    last_date = df_feat["date"].iloc[-1].date()
    last_pred = float(oof[-1])
    st.markdown(
        f"<div class='kpi-card'><div class='muted'>Latest OOF prediction</div>"
        f"<div style='font-size:1.6rem; font-weight:700'>{last_pred:.4f}</div>"
        f"<div class='muted small'>Date: {last_date}</div></div>",
        unsafe_allow_html=True
    )

with k3:
    start = df_feat["date"].iloc[0].date()
    end = df_feat["date"].iloc[-1].date()
    st.markdown(
        f"<div class='kpi-card'><div class='muted'>Backtest window</div>"
        f"<div style='font-size:1.2rem; font-weight:700'>{start} → {end}</div>"
        f"<div class='muted small'>Rows: {len(df_feat)} | Features: {len(feature_cols)}</div></div>",
        unsafe_allow_html=True
    )

st.write("")

# -------------------------
# Table + backtest plot
# -------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Recent rows")
    st.dataframe(df_feat[["date", "inflation_yoy", "sentiment", "target"]].tail(12))

with col2:
    st.subheader("Backtest: Actual vs Predicted")
    fig = plt.figure(figsize=(10, 4))
    plt.plot(df_feat["date"], y.values, label="Actual (target)")
    plt.plot(df_feat["date"], oof, label="Predicted (OOF)")
    plt.title("Actual vs Predicted (Walk-forward)")
    plt.xlabel("Date")
    plt.ylabel("YoY Inflation")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

# -------------------------
# Future forecast
# -------------------------
st.subheader("🔮 Next Inflation Forecast")
latest_X = X.iloc[[-1]]
future_pred = float(final_model.predict(latest_X)[0])
st.metric(label=f"Predicted YoY Inflation (t+{horizon})", value=f"{future_pred:.4f}")
st.caption("Uses the latest available feature row in your merged dataset.")

# -------------------------
# Feature importance chart
# -------------------------
st.subheader("📊 Feature Importance (XGBoost)")
imp = final_model.feature_importances_
imp_df = pd.DataFrame({"feature": feature_cols, "importance": imp}).sort_values("importance", ascending=False)

fig_imp = plt.figure(figsize=(8, 4))
plt.barh(imp_df["feature"], imp_df["importance"])
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.title("XGBoost Feature Importance")
plt.tight_layout()
st.pyplot(fig_imp, use_container_width=True)

# -------------------------
# Error-over-time chart
# -------------------------
st.subheader("📉 Prediction Error Over Time")
errors = (y.values - oof)
fig_err = plt.figure(figsize=(10, 4))
plt.plot(df_feat["date"], errors)
plt.axhline(0, linestyle="--")
plt.title("Prediction Error (Actual − Predicted)")
plt.xlabel("Date")
plt.ylabel("Error")
plt.grid(True, alpha=0.25)
plt.tight_layout()
st.pyplot(fig_err, use_container_width=True)

# -------------------------
# SHAP (OPTIONAL)
# -------------------------
st.subheader("🔍 Explainability (SHAP)")

if run_shap:
    with st.spinner("Computing SHAP values (can take 30–90s on cloud)..."):
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X)

    shap_col1, shap_col2 = st.columns([1, 1])

    with shap_col1:
        st.caption("Global feature importance (mean |SHAP|)")
        fig_bar = plt.figure()
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        st.pyplot(fig_bar, use_container_width=True)

    with shap_col2:
        st.caption("SHAP beeswarm (direction + magnitude)")
        fig_swarm = plt.figure()
        shap.summary_plot(shap_values, X, show=False)
        st.pyplot(fig_swarm, use_container_width=True)

    # Plain-English interpretation
    st.subheader("🧠 Model Interpretation (Plain English)")
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    top_feature = importance_df.loc[0, "feature"]
    st.write(f"**Most influential feature:** `{top_feature}`")

    if "infl_lag" in top_feature:
        st.write("→ The model relies heavily on **inflation persistence** (past inflation predicting future inflation).")
    elif "sentiment" in top_feature:
        st.write("→ The model relies heavily on **financial news sentiment** signals.")

    # Directional takeaway for sentiment
    if "sentiment" in feature_cols:
        sent_idx = feature_cols.index("sentiment")
        corr = np.corrcoef(X["sentiment"].values, shap_values[:, sent_idx])[0, 1]
        if np.isfinite(corr):
            if corr > 0:
                st.write("**Direction:** Higher (more positive) sentiment generally pushes predictions **up**.")
            elif corr < 0:
                st.write("**Direction:** Higher (more positive) sentiment generally pushes predictions **down**.")
            else:
                st.write("**Direction:** Sentiment effect is mixed/neutral on average.")

    st.dataframe(importance_df.head(10))

else:
    st.info("SHAP is OFF to keep the cloud app fast. Turn on **Compute SHAP** in the sidebar if you want explanations.")

# -------------------------
# Download results
# -------------------------
st.subheader("📌 Download results")
out = df_feat[["date"]].copy()
out["actual_target"] = y.values
out["pred_oof"] = oof
out["error"] = errors

st.download_button(
    "Download predictions CSV",
    data=out.to_csv(index=False).encode("utf-8"),
    file_name="oof_predictions.csv",
    mime="text/csv"
)










