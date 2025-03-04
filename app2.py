import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Helper function: min–max scale a column to [0, 10]
def min_max_scale(series):
    return (series - series.min()) / (series.max() - series.min()) * 10

st.title("TVM Restaurant/Hotel/Café Ranking System")

st.markdown("""
Upload your **enriched_TVM_50.csv** file and adjust the **weights** to see how 
the overall ranking changes. Select an establishment to see its metrics on a 
**radar chart**. **Your selected restaurant** will remain the same even if you 
move the sliders.
""")

# Sidebar for adjusting metric weights
st.sidebar.title("Adjust Metric Weights")
pop_wt = st.sidebar.slider("Popularity Score Weight", 0.0, 1.0, 0.2, 0.05)
rating_wt = st.sidebar.slider("Average Rating Weight", 0.0, 1.0, 0.2, 0.05)
amb_wt = st.sidebar.slider("Ambiance Weight", 0.0, 1.0, 0.2, 0.05)
srv_wt = st.sidebar.slider("Service Weight", 0.0, 1.0, 0.2, 0.05)
uniq_wt = st.sidebar.slider("Uniqueness Weight", 0.0, 1.0, 0.1, 0.05)
nri_wt = st.sidebar.slider("NRI-Friendliness Weight", 0.0, 1.0, 0.1, 0.05)

# Avoid division by zero if all weights are 0
total_weight = pop_wt + rating_wt + amb_wt + srv_wt + uniq_wt + nri_wt
if total_weight == 0:
    st.sidebar.warning("Please assign at least one non-zero weight!")
    total_weight = 1.0

uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is not None:
    # Read CSV
    df = pd.read_csv(uploaded_file)

    # Check that the columns exist
    required_cols = [
        "Restaurant_Name", "Popularity_Score", "Avg_Rating", "Ambiance_Score",
        "Service_Score", "Uniqueness_Score", "NRI_Friendly_Score"
    ]
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Column '{col}' not found in CSV. Please upload the correct file.")
            st.stop()

    # 1. Scale the numeric columns to [0, 10]
    #    - For Avg_Rating (originally out of 5), first multiply by 2 => out of 10
    df["Avg_Rating_Scaled"] = df["Avg_Rating"] * 2

    # Apply min–max scaling for each metric
    df["Popularity_Score_Scaled"]    = min_max_scale(df["Popularity_Score"])
    df["Avg_Rating_Scaled"]          = min_max_scale(df["Avg_Rating_Scaled"])
    df["Ambiance_Score_Scaled"]      = min_max_scale(df["Ambiance_Score"])
    df["Service_Score_Scaled"]       = min_max_scale(df["Service_Score"])
    df["Uniqueness_Score_Scaled"]    = min_max_scale(df["Uniqueness_Score"])
    df["NRI_Friendly_Score_Scaled"]  = min_max_scale(df["NRI_Friendly_Score"])

    # 2. Compute a custom composite score using user-defined weights
    df["Custom_Composite"] = (
        pop_wt  * df["Popularity_Score_Scaled"] +
        rating_wt  * df["Avg_Rating_Scaled"] +
        amb_wt  * df["Ambiance_Score_Scaled"] +
        srv_wt  * df["Service_Score_Scaled"] +
        uniq_wt * df["Uniqueness_Score_Scaled"] +
        nri_wt  * df["NRI_Friendly_Score_Scaled"]
    ) / total_weight

    # Sort by custom composite descending
    df_sorted = df.sort_values("Custom_Composite", ascending=False)

    st.subheader("Ranked List (Based on Your Weights)")
    st.dataframe(
        df_sorted[["Restaurant_Name", "Custom_Composite"]].reset_index(drop=True)
    )

    # Create a list of all restaurants (in sorted order)
    restaurant_list = df_sorted["Restaurant_Name"].tolist()

    # Use a key for the selectbox so Streamlit remembers the user's choice
    selected_restaurant = st.selectbox(
        "Select an establishment:",
        restaurant_list,
        key="selected_restaurant"
    )

    # Retrieve the row for the selected restaurant
    rest_data = df[df["Restaurant_Name"] == selected_restaurant].iloc[0]

    st.markdown(f"### Details for **{selected_restaurant}**")
    st.write("**Custom Composite Score:**", round(rest_data["Custom_Composite"], 2))

    # Prepare data for the radar chart (use scaled metrics, each in [0, 10])
    metrics = [
        "Popularity_Score_Scaled",
        "Avg_Rating_Scaled",
        "Ambiance_Score_Scaled",
        "Service_Score_Scaled",
        "Uniqueness_Score_Scaled",
        "NRI_Friendly_Score_Scaled"
    ]
    labels = [
        "Popularity",
        "Avg Rating",
        "Ambiance",
        "Service",
        "Uniqueness",
        "NRI Friendly"
    ]
    values = [rest_data[m] for m in metrics]

    # Close the loop for the radar chart
    values += values[:1]
    labels += labels[:1]

    # Create the radar chart using Plotly
    fig = go.Figure(
        data=[
            go.Scatterpolar(r=values, theta=labels, fill='toself', name=selected_restaurant)
        ],
        layout=go.Layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]  # each metric scaled to 0-10
                )
            ),
            showlegend=False,
            title=f"Metric Breakdown for {selected_restaurant}"
        )
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Please upload the 'enriched_TVM_50.csv' file to begin.")

