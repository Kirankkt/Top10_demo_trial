import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Helper function: min–max scale a column to [0, 10]
def min_max_scale(series):
    return (series - series.min()) / (series.max() - series.min()) * 10

st.title("TVM Restaurant/Hotel/Café Ranking System")

st.markdown("""
Upload your file and adjust the weights to see how 
the overall ranking changes. Then select an establishment to see its metrics on a 
radar chart. 
""")

# -- SIDEBAR: Metric Weights --
st.sidebar.title("Adjust Metric Weights")
pop_wt    = st.sidebar.slider("Popularity Score Weight",     0.0, 1.0, 0.2, 0.05)
rating_wt = st.sidebar.slider("Average Rating Weight",       0.0, 1.0, 0.2, 0.05)
amb_wt    = st.sidebar.slider("Ambiance Weight",             0.0, 1.0, 0.2, 0.05)
srv_wt    = st.sidebar.slider("Service Weight",              0.0, 1.0, 0.2, 0.05)
uniq_wt   = st.sidebar.slider("Uniqueness Weight",           0.0, 1.0, 0.1, 0.05)
nri_wt    = st.sidebar.slider("NRI-Friendliness Weight",     0.0, 1.0, 0.1, 0.05)

# Prevent all-zero
total_weight = pop_wt + rating_wt + amb_wt + srv_wt + uniq_wt + nri_wt
if total_weight == 0:
    st.sidebar.warning("Please assign at least one non-zero weight!")
    total_weight = 1.0

# Keep original (default) weights for comparison
orig_weights = {
    "pop": 0.2,
    "rating": 0.2,
    "amb": 0.2,
    "srv": 0.2,
    "uniq": 0.1,
    "nri": 0.1
}

uploaded_file = st.file_uploader("Upload CSV file", type="csv")
if not uploaded_file:
    st.info("Please upload the 'enriched_TVM_50.csv' file to begin.")
    st.stop()

df = pd.read_csv(uploaded_file)
required_cols = [
    "Restaurant_Name", "Popularity_Score", "Avg_Rating", 
    "Ambiance_Score", "Service_Score", "Uniqueness_Score", 
    "NRI_Friendly_Score"
]
for col in required_cols:
    if col not in df.columns:
        st.error(f"Column '{col}' not found in CSV.")
        st.stop()

# 1. scale all
df["Popularity_Score_S"]   = min_max_scale(df["Popularity_Score"])
df["Avg_Rating_S"]         = min_max_scale(df["Avg_Rating"] * 2)
df["Ambiance_Score_S"]     = min_max_scale(df["Ambiance_Score"])
df["Service_Score_S"]      = min_max_scale(df["Service_Score"])
df["Uniqueness_Score_S"]   = min_max_scale(df["Uniqueness_Score"])
df["NRI_Friendly_Score_S"] = min_max_scale(df["NRI_Friendly_Score"])

# 2. original composite (fixed/default weights)
df["Original_Composite"] = (
    orig_weights["pop"]    * df["Popularity_Score_S"]   +
    orig_weights["rating"] * df["Avg_Rating_S"]         +
    orig_weights["amb"]    * df["Ambiance_Score_S"]     +
    orig_weights["srv"]    * df["Service_Score_S"]      +
    orig_weights["uniq"]   * df["Uniqueness_Score_S"]   +
    orig_weights["nri"]    * df["NRI_Friendly_Score_S"]
)  # sum of orig_weights = 1.0

# 3. custom composite (user‑defined)
df["Custom_Composite"] = (
    pop_wt  * df["Popularity_Score_S"]   +
    rating_wt * df["Avg_Rating_S"]       +
    amb_wt  * df["Ambiance_Score_S"]     +
    srv_wt  * df["Service_Score_S"]      +
    uniq_wt * df["Uniqueness_Score_S"]   +
    nri_wt  * df["NRI_Friendly_Score_S"]
) / total_weight

# 4. compute ranks (1 = best)
df["Original_Rank"] = df["Original_Composite"].rank(method="min", ascending=False).astype(int)
df["Custom_Rank"]   = df["Custom_Composite"].rank(method="min",   ascending=False).astype(int)

# 5. present sorted by custom
df_sorted = df.sort_values("Custom_Composite", ascending=False).reset_index(drop=True)
st.subheader("Ranked List (Based on Your Weights)")
st.dataframe(
    df_sorted[
        ["Restaurant_Name", 
         "Original_Composite", "Original_Rank", 
         "Custom_Composite",   "Custom_Rank"]
    ].round(2)
)

# 6. select box
restaurant_list = df_sorted["Restaurant_Name"].tolist()
if "sel" not in st.session_state:
    st.session_state.sel = restaurant_list[0]
if st.session_state.sel not in restaurant_list:
    st.session_state.sel = restaurant_list[0]

sel = st.selectbox("Select an establishment:", restaurant_list,
                   index=restaurant_list.index(st.session_state.sel))
st.session_state.sel = sel

# 7. details
rest = df[df["Restaurant_Name"] == sel].iloc[0]
st.markdown(f"### Details for **{sel}**")
st.write("• Original Composite Score:", round(rest["Original_Composite"], 2))
st.write("• Custom Composite Score:  ", round(rest["Custom_Composite"],   2))

# Radar chart (unchanged)
metrics = [
    "Popularity_Score_S", "Avg_Rating_S", "Ambiance_Score_S",
    "Service_Score_S",   "Uniqueness_Score_S", "NRI_Friendly_Score_S"
]
labels = ["Popularity","Avg Rating","Ambiance","Service","Uniqueness","NRI Friendly"]
vals   = [rest[m] for m in metrics]
vals  += vals[:1]; labels += labels[:1]

fig = go.Figure(
    data=[go.Scatterpolar(r=vals, theta=labels, fill='toself', name=sel)],
    layout=go.Layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,10])),
        showlegend=False,
        title=f"Metric Breakdown for {sel}"
    )
)
st.plotly_chart(fig, use_container_width=True)
