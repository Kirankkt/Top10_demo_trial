import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.title("TVM Restaurant/Hotel/Café Ranking System")

st.markdown("Upload your enriched CSV file to see the rankings and a detailed radar chart of key metrics.")

# File uploader to accept CSV input
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is not None:
    # Read CSV into a DataFrame
    df = pd.read_csv(uploaded_file)

    # If Avg_Rating_Scaled column does not exist, create it (scale Avg_Rating to 10)
    if 'Avg_Rating_Scaled' not in df.columns:
        df['Avg_Rating_Scaled'] = df['Avg_Rating'] * 2

    # Convert key metric columns to numeric
    numeric_cols = ['Popularity_Score', 'Avg_Rating_Scaled', 'Ambiance_Score',
                    'Service_Score', 'Uniqueness_Score', 'NRI_Friendly_Score', 'Composite_Score']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    st.header("Top 50 Rankings")
    df_sorted = df.sort_values("Composite_Score", ascending=False)
    st.dataframe(df_sorted[["Restaurant_Name", "Composite_Score"]].reset_index(drop=True))

    # Let user select an establishment
    restaurant_list = df_sorted["Restaurant_Name"].tolist()
    selected_restaurant = st.selectbox("Select an establishment to view details:", restaurant_list)

    # Retrieve data for the selected restaurant
    restaurant_data = df[df["Restaurant_Name"] == selected_restaurant].iloc[0]
    st.subheader(f"Details for {selected_restaurant}")
    st.write(f"**Composite Score:** {restaurant_data['Composite_Score']}")

    # Define the metrics for the radar chart
    metrics = ['Popularity_Score', 'Avg_Rating_Scaled', 'Ambiance_Score', 
               'Service_Score', 'Uniqueness_Score', 'NRI_Friendly_Score']
    values = [restaurant_data[m] for m in metrics]

    # Close the loop for the radar chart
    values += values[:1]
    metrics += metrics[:1]

    # Create the radar (spider) chart using Plotly
    fig = go.Figure(
        data=[
            go.Scatterpolar(r=values, theta=metrics, fill='toself', name=selected_restaurant)
        ],
        layout=go.Layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )
            ),
            showlegend=False,
            title=f"Metric Breakdown for {selected_restaurant}"
        )
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Please upload the CSV file (e.g., enriched_TVM_50.csv) to view the rankings and chart.")
