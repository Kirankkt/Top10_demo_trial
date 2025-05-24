import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
#A trial

# ---------- Helper Functions ----------

@st.cache_data
def load_data():
    # Replace 'dummy_dataset.csv' with your dataset file path.
    df = pd.read_csv('Updated_Trivandrum_Data__No_Duplicates_.csv')
    return df

def convert_price_range(x):
    """Convert a price range string (e.g., "$$$") to a numeric value (the count of '$')."""
    if isinstance(x, str):
        return len(x.strip())
    return np.nan

def preprocess_data(df):
    # Convert Price Range to a numeric value.
    df['Price Range Numeric'] = df['Price Range'].apply(convert_price_range)
    # Set a default 'Uniqueness' metric for categories that need it.
    df['Uniqueness'] = 1
    # Create an Accessibility metric based on Public Transport Proximity (m); lower is better.
    scaler_access = MinMaxScaler()
    df['Public Transport Proximity Norm'] = scaler_access.fit_transform(df[['Public Transport Proximity (m)']])
    df['Accessibility'] = 1 - df['Public Transport Proximity Norm']
    
    # List of metric columns used in ranking.
    metric_cols = [
        "Rating", "Reviews", "Footfall", "Repeat Visitors (%)", "Avg Stay Duration (mins)",
        "Price Range Numeric", "Popularity Index", "Instagram Mentions",
        "Historical Mentions", "Heritage Status", "Years Since Establishment",
        "TripAdvisor Ranking", "Google Check-ins"
    ]
    scaler_metrics = MinMaxScaler()
    df_norm = df.copy()
    df_norm[metric_cols] = scaler_metrics.fit_transform(df[metric_cols])
    # Carry over precomputed Accessibility and Uniqueness.
    df_norm['Accessibility'] = df['Accessibility']
    df_norm['Uniqueness'] = df['Uniqueness']
    return df_norm

def get_default_category_weights(cat):
    """
    Returns the default weight dictionary for a given category.
    For 'temple' or 'church', they share the same default.
    """
    cat_lower = cat.strip().lower()
    if cat_lower in ['temple', 'church']:
        return {
            'Heritage Status': 0.35,
            'Years Since Establishment': 0.25,
            'Footfall': 0.25,
            'Accessibility': 0.15
        }
    weights_dict = {
        'restaurant': {
            'Rating': 0.25,
            'Reviews': 0.20,
            'Footfall': 0.15,
            'Repeat Visitors (%)': 0.15,
            'Avg Stay Duration (mins)': 0.15,
            'Price Range Numeric': 0.10
        },
        'nightclub': {
            'Rating': 0.20,
            'Reviews': 0.15,
            'Footfall': 0.20,
            'Popularity Index': 0.20,
            'Instagram Mentions': 0.15,
            'Avg Stay Duration (mins)': 0.10
        },
        'museum': {
            'Historical Mentions': 0.40,
            'Heritage Status': 0.20,
            'Years Since Establishment': 0.20,
            'Footfall': 0.20
        },
        'theatre': {
            'Rating': 0.30,
            'Reviews': 0.20,
            'Avg Stay Duration (mins)': 0.20,
            'Instagram Mentions': 0.15,
            'TripAdvisor Ranking': 0.15
        },
        'attraction': {
            'Instagram Mentions': 0.25,
            'Google Check-ins': 0.25,
            'Footfall': 0.25,
            'TripAdvisor Ranking': 0.25
        },
        'boutique': {
            'Uniqueness': 0.30,
            'Reviews': 0.25,
            'Instagram Mentions': 0.25,
            'Price Range Numeric': 0.20
        },
        'palace': {
            'Heritage Status': 0.30,
            'Historical Mentions': 0.30,
            'Instagram Mentions': 0.20,
            'Footfall': 0.20
        },
        'antique store': {
            'Uniqueness': 0.30,
            'Instagram Mentions': 0.30,
            'Reviews': 0.20,
            'Price Range Numeric': 0.20
        }
    }
    return weights_dict.get(cat_lower, None)

def compute_category_score(row, weights):
    score = 0
    for metric, weight in weights.items():
        score += weight * row.get(metric, 0)
    return score

def compute_custom_scores(df, custom_weights):
    # Compute composite scores using the provided (normalized) custom weights.
    df = df.copy()
    df['Custom_Composite_Score'] = df.apply(lambda row: compute_category_score(row, custom_weights), axis=1)
    return df

# ---------- Streamlit App ----------

def main():
    st.title("Trivandrum Top 10 Curated Venues")
    st.write(
        "This decision-support tool displays the top 10 venues for a selected category based on a composite score. "
        "CEOs can adjust raw weights for key metrics—the app will automatically normalize these weights (so they sum to 1) "
        "and update the rankings in real time."
    )
    
    # Load and preprocess data.
    df = load_data()
    df_norm = preprocess_data(df)
    
    # Sidebar: Category selection.
    available_categories = [
        'Restaurant', 'Nightclub', 'Museum', 'Theatre',
        'Attraction', 'Boutique', 'Temple', 'Church',
        'Palace', 'Antique Store'
    ]
    st.sidebar.header("Configuration")
    selected_category = st.sidebar.selectbox("Select Category", available_categories)
    cat_lower = selected_category.strip().lower()
    
    # Filter dataset based on category.
    if cat_lower in ['temple', 'church']:
        df_filtered = df_norm[df_norm['Category'].str.strip().str.lower().isin(['temple', 'church'])]
        default_weights = get_default_category_weights("temple")
    else:
        df_filtered = df_norm[df_norm['Category'].str.strip().str.lower() == cat_lower]
        default_weights = get_default_category_weights(selected_category)
    
    if df_filtered.empty:
        st.warning(f"No entries found for the category: {selected_category}")
        return
    
    # Display default weights.
    st.sidebar.subheader(f"Default Weights for {selected_category.title()}")
    for metric, weight in default_weights.items():
        st.sidebar.write(f"**{metric}:** {weight}")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Adjust Weights")
    st.sidebar.write(
        "Enter raw weight values below. These values will be automatically normalized so that their total sum equals 1."
    )
    
    # Form for adjusting raw weights.
    with st.sidebar.form("weight_adjustments"):
        raw_weights = {}
        for metric, default_value in default_weights.items():
            raw_value = st.number_input(
                label=f"Raw weight for {metric}",
                min_value=0.0,
                max_value=10.0,
                value=default_value,
                step=0.05,
                format="%.2f"
            )
            raw_weights[metric] = raw_value
        submitted = st.form_submit_button("Update Rankings")
    
    # Normalize raw weights.
    total_raw = sum(raw_weights.values())
    if total_raw > 0:
        normalized_weights = {metric: val / total_raw for metric, val in raw_weights.items()}
    else:
        normalized_weights = default_weights  # Fallback if total is zero.
    
    st.sidebar.subheader("Normalized Weights")
    for metric, norm_val in normalized_weights.items():
        st.sidebar.write(f"**{metric}:** {norm_val:.2f}")
    
    # Draw a pie chart for the normalized weights.
    fig, ax = plt.subplots(figsize=(4, 4))
    labels = list(normalized_weights.keys())
    sizes = list(normalized_weights.values())
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.sidebar.pyplot(fig)
    
    # Compute composite scores using the custom normalized weights.
    if submitted:
        df_custom = compute_custom_scores(df_filtered, normalized_weights)
    else:
        # If form not submitted, use default weights.
        df_custom = compute_custom_scores(df_filtered, get_default_category_weights(selected_category))
    
    # Sort the venues by composite score.
    df_custom = df_custom.sort_values(by='Custom_Composite_Score', ascending=False)
    
    # Main area: Display rankings.
    st.subheader(f"Top 10 {selected_category.title()} by Composite Score")
    st.dataframe(df_custom[['Name', 'Custom_Composite_Score']].reset_index(drop=True))
    
    # Bar Chart: Breakdown of composite score for the top venue.
    if not df_custom.empty:
        top_venue = df_custom.iloc[0]
        st.markdown("### Breakdown of Top Venue Score")
        breakdown = {}
        for metric, weight in normalized_weights.items():
            metric_value = top_venue.get(metric, 0)
            breakdown[metric] = weight * metric_value
        breakdown_df = pd.DataFrame({
            'Metric': list(breakdown.keys()),
            'Contribution': list(breakdown.values())
        })
        # Create a bar chart.
        st.bar_chart(breakdown_df.set_index('Metric'))
    
    # Download Button: Allow exporting the ranking as a CSV file.
    csv = df_custom[['Name', 'Custom_Composite_Score']].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Rankings as CSV",
        data=csv,
        file_name='top_venues.csv',
        mime='text/csv'
    )

if __name__ == "__main__":
    main()
