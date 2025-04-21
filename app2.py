import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Helper: min–max scale a column to [0, 10]
def min_max_scale(series: pd.Series) -> pd.Series:
    return (series - series.min()) / (series.max() - series.min()) * 10

st.title("TVM Restaurant/Hotel/Café Ranking System")

st.markdown("""
Upload your enriched CSV and tweak the weights to see how
your custom ranking compares against the baseline (default) ranking.
Then select an establishment to view its detailed metrics in a radar chart.
""")

# --- SIDEBAR: Custom Weights ---
st.sidebar.title("Adjust Metric Weights")
pop_wt    = st.sidebar.slider("Popularity Score",     0.0, 1.0, 0.2, 0.05)
rating_wt = st.sidebar.slider("Average Rating",       0.0, 1.0, 0.2, 0.05)
amb_wt    = st.sidebar.slider("Ambiance Score",       0.0, 1.0, 0.2, 0.05)
srv_wt    = st.sidebar.slider("Service Score",        0.0, 1.0, 0.2, 0.05)
uniq_wt   = st.sidebar.slider("Uniqueness Score",     0.0, 1.0, 0.1, 0.05)
nri_wt    = st.sidebar.slider("NRI-Friendliness",     0.0, 1.0, 0.1, 0.05)

# Prevent division by zero
total_wt = pop_wt + rating_wt + amb_wt + srv_wt + uniq_wt + nri_wt
if total_wt == 0:
    st.sidebar.warning("At least one weight must be non-zero.")
    total_wt = 1.0

# Baseline (default) weights
orig_weights = {
    "pop":    0.2,
    "rating": 0.2,
    "amb":    0.2,
    "srv":    0.2,
    "uniq":   0.1,
    "nri":    0.1
}

uploaded = st.file_uploader("Upload your CSV (e.g. enriched_TVM_50.csv)", type="csv")
if not uploaded:
    st.info("Please upload a CSV to get started.")
    st.stop()

df = pd.read_csv(uploaded)

# Required columns
required = [
    "Restaurant_Name", "Popularity_Score", "Avg_Rating",
    "Ambiance_Score",   "Service_Score",    "Uniqueness_Score",
    "NRI_Friendly_Score"
]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing columns in CSV: {missing}")
    st.stop()

# --- 1. Scale everything to [0,10] ---
df["Popularity_S"]    = min_max_scale(df["Popularity_Score"])
df["Avg_Rating_S"]    = min_max_scale(df["Avg_Rating"] * 2)  # convert 5→10 scale, then rescale
df["Ambiance_S"]      = min_max_scale(df["Ambiance_Score"])
df["Service_S"]       = min_max_scale(df["Service_Score"])
df["Uniqueness_S"]    = min_max_scale(df["Uniqueness_Score"])
df["NRI_Friendly_S"]  = min_max_scale(df["NRI_Friendly_Score"])

# --- 2. Baseline composite (fixed weights) ---
df["Baseline_Score"] = (
    orig_weights["pop"]    * df["Popularity_S"]    +
    orig_weights["rating"] * df["Avg_Rating_S"]    +
    orig_weights["amb"]    * df["Ambiance_S"]      +
    orig_weights["srv"]    * df["Service_S"]       +
    orig_weights["uniq"]   * df["Uniqueness_S"]    +
    orig_weights["nri"]    * df["NRI_Friendly_S"]
)

# --- 3. Custom composite (user-defined weights) ---
df["Custom_Score"] = (
    pop_wt    * df["Popularity_S"]    +
    rating_wt * df["Avg_Rating_S"]    +
    amb_wt    * df["Ambiance_S"]      +
    srv_wt    * df["Service_S"]       +
    uniq_wt   * df["Uniqueness_S"]    +
    nri_wt    * df["NRI_Friendly_S"]
) / total_wt

# --- 4. Compute ranks (1 is best) ---
df["Baseline_Rank"] = df["Baseline_Score"].rank(ascending=False, method="min").astype(int)
df["Custom_Rank"]   = df["Custom_Score"].rank(ascending=False,   method="min").astype(int)

# --- 5. Show side-by-side ranking table ---
df_sorted = df.sort_values("Custom_Score", ascending=False).reset_index(drop=True)
display_cols = [
    "Restaurant_Name",
    "Baseline_Score", "Baseline_Rank",
    "Custom_Score",   "Custom_Rank"
]
df_display = (
    df_sorted[display_cols]
      .rename(columns={
         "Baseline_Score": "Baseline Score",
         "Baseline_Rank":  "Baseline Rank",
         "Custom_Score":   "Custom Score",
         "Custom_Rank":    "Custom Rank"
      })
      .round(2)
)

st.subheader("Ranked List: Baseline vs. Custom")
st.dataframe(df_display)

# --- 6. Select an establishment for details ---
restaurants = df_sorted["Restaurant_Name"].tolist()
if "selected" not in st.session_state:
    st.session_state.selected = restaurants[0]

choice = st.selectbox(
    "Select an establishment to inspect:", 
    restaurants, 
    index=restaurants.index(st.session_state.selected)
)
st.session_state.selected = choice

rest = df[df["Restaurant_Name"] == choice].iloc[0]

st.markdown(f"### Details for **{choice}**")
st.write(f"**Baseline Score:** {rest['Baseline_Score']:.2f}  —  **Baseline Rank:** {rest['Baseline_Rank']}")
st.write(f"**Custom Score:**   {rest['Custom_Score']:.2f}  —  **Custom Rank:**   {rest['Custom_Rank']}")

# --- 7. Radar chart of scaled metrics ---
metrics = ["Popularity_S","Avg_Rating_S","Ambiance_S","Service_S","Uniqueness_S","NRI_Friendly_S"]
labels  = ["Popularity","Avg Rating","Ambiance","Service","Uniqueness","NRI Friendly"]
values  = [rest[m] for m in metrics]
# close the loop
values += values[:1]
labels += labels[:1]

fig = go.Figure(
    data=[go.Scatterpolar(r=values, theta=labels, fill="toself", name=choice)],
    layout=go.Layout(
        title=f"Metric Breakdown for {choice}",
        polar=dict(radialaxis=dict(visible=True, range=[0,10])),
        showlegend=False
    )
)
st.plotly_chart(fig, use_container_width=True)
