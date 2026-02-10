import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

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
    "<div class='muted'>Walk-forward backtesting • XGBoost regression • Optional SHAP • Downloadable forecasts</div>",
    unsafe_allow_html=True
)
st.write("")

# Detect if running on Streamlit Cloud
IS_CLOUD = bool(st.secrets.get("STREAMLIT_CLOUD", "")) if hasattr(st, "secrets") else False


# =========================
# DATA LOADERS
# =========================
@st.cache_data(show_spinner=False)
def load_cpi(start_date: str = "2015-01-01") -> pd.DataFrame:
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"
    cpi = pd.read_csv(url)
    cpi.columns = ["date", "cpi"]
    cpi["date"] = pd.to_datetime(cpi["date"])
    cpi = cpi[cpi["date"] >= pd.to_datetime(start_date)].copy()

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

    df["target"] = df["inflation_yoy"].shift(-horizon)

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


# =========================
# FAST WALK-FORWARD (WITH PROGRESS)
# =========================
def walk_forward_oof_fast(X, y, params, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof = np.full(len(y), np.nan)

    prog = st.progress(0)
    status = st.empty()
    t0 = time.time()

    splits = list(tscv.split(X))
    total = len(splits)

    for k, (train_idx, test_idx) in enumerate(splits, start=1):
        status.write(f"Training fold {k}/{total} ...")
        model = xgb.XGBRegressor(**params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[test_idx] = model.predict(X.iloc[test_idx])

        prog.progress(int(100 * k / total))

    status.write(f"✅ Done in {time.time() - t0:.1f}s")
    return oof


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
    n_splits = st.slider("Walk-forward folds", 2, 8, 3)

    st.header("XGBoost Params")
    n_estimators = st.slider("n_estimators", 50, 2000, 300, step=50)
    max_depth = st.slider("max_depth", 2, 10, 3)
    learning_rate = st.number_input("learning_rate", 0.01, 0.3, 0.05, step=0.01)
    subsample = st.slider("subsample", 0.5, 1.0, 0.9)
    colsample_bytree = st.slider("colsample_bytree", 0.5, 1.0, 0.9)
    random_state = st.number_input("random_state", value=42, step=1)

    st.header("Explainability")
    run_shap = st.checkbox("Compute SHAP (slow)", value=False)

    run_btn = st.button("🚀 Run backtest")


# =========================
# MAIN
# =========================
if not sentiment_file:
    st.info("Upload your **monthly sentiment CSV** to begin.")
    st.stop()

if run_btn:
    # Clear previous results
    st.session_state.pop("results", None)

    with st.spinner("Loading CPI + sentiment and building features..."):
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

    # -------- FAST BACKTEST PARAMS (caps trees to avoid cloud hanging) --------
    fast_params = dict(
        n_estimators=min(int(n_estimators), 200),  # cap trees for backtest speed
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective="reg:squarederror",
        random_state=int(random_state),
    )

    # -------- FULL FINAL MODEL PARAMS (uses chosen n_estimators) -------------
    full_params = dict(
        n_estimators=int(n_estimators),
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective="reg:squarederror",
        random_state=int(random_state),
    )

    st.subheader("Walk-forward backtest (fast mode)")
    oof = walk_forward_oof_fast(X, y, params=fast_params, n_splits=n_splits)
    mae = mean_absolute_error(y, oof)

    st.session_state["results"] = {
        "df_feat": df_feat,
        "X": X,
        "y": y,
        "feature_cols": feature_cols,
        "oof": oof,
        "mae": mae,
        "full_params": full_params
    }

# Show results if available
if "results" not in st.session_state:
    st.warning("Adjust settings in the sidebar and click **Run backtest**.")
    st.stop()

res = st.session_state["results"]
df_feat = res["df_feat"]
X = res["X"]
y = res["y"]
feature_cols = res["feature_cols"]
oof = res["oof"]
mae = res["mae"]
full_params = res["full_params"]

# =========================
# FINAL MODEL (FULL)
# =========================
with st.spinner("Training final model (full params)..."):
    final_model = xgb.XGBRegressor(**full_params)
    final_model.fit(X, y)

# =========================
# KPI CARDS
# =========================
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

# =========================
# TABLE + PLOT
# =========================
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

# =========================
# FUTURE FORECAST
# =========================
st.subheader("🔮 Next Inflation Forecast")
future_pred = float(final_model.predict(X.iloc[[-1]])[0])
st.metric(label=f"Predicted YoY Inflation (t+{horizon})", value=f"{future_pred:.4f}")

# =========================
# FEATURE IMPORTANCE
# =========================
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

# =========================
# OPTIONAL SHAP
# =========================
st.subheader("🔍 Explainability (SHAP)")
if run_shap:
    with st.spinner("Computing SHAP values..."):
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X)

    fig_bar = plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    st.pyplot(fig_bar, use_container_width=True)

    fig_swarm = plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    st.pyplot(fig_swarm, use_container_width=True)
else:
    st.info("Turn on SHAP in the sidebar if you want explainability (slower).")

# =========================
# DOWNLOAD
# =========================
st.subheader("📌 Download results")
out = df_feat[["date"]].copy()
out["actual_target"] = y.values
out["pred_oof"] = oof
out["error"] = y.values - oof

st.download_button(
    "Download predictions CSV",
    data=out.to_csv(index=False).encode("utf-8"),
    file_name="oof_predictions.csv",
    mime="text/csv"
)










