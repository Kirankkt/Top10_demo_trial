import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ---------- Helper Functions ----------

@st.cache_data
def load_data():
    # Replace 'dummy_dataset.csv' with the path to your dataset file.
    df = pd.read_csv('Updated_Trivandrum_Data__No_Duplicates_.csv')
    return df

def convert_price_range(x):
    """Convert a price range string (e.g., "$$$") to a numeric value by counting '$'."""
    if isinstance(x, str):
        return len(x.strip())
    return np.nan

def preprocess_data(df):
    # Convert Price Range to a numeric value.
    df['Price Range Numeric'] = df['Price Range'].apply(convert_price_range)
    # Set a default 'Uniqueness' metric for boutiques and antique stores.
    df['Uniqueness'] = 1
    # Create an Accessibility metric based on Public Transport Proximity (m): lower is better.
    scaler_access = MinMaxScaler()
    df['Public Transport Proximity Norm'] = scaler_access.fit_transform(df[['Public Transport Proximity (m)']])
    df['Accessibility'] = 1 - df['Public Transport Proximity Norm']
    
    # List of metric columns to be normalized.
    metric_cols = [
        "Rating", "Reviews", "Footfall", "Repeat Visitors (%)", "Avg Stay Duration (mins)",
        "Price Range Numeric", "Popularity Index", "Instagram Mentions",
        "Historical Mentions", "Heritage Status", "Years Since Establishment",
        "TripAdvisor Ranking", "Google Check-ins"
    ]
    scaler_metrics = MinMaxScaler()
    df_norm = df.copy()
    df_norm[metric_cols] = scaler_metrics.fit_transform(df[metric_cols])
    # Carry over the precomputed Accessibility and Uniqueness metrics.
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
    st.write("This app displays the top 10 venues in a category based on a composite score. CEOs can adjust the raw weights; these values will automatically be normalized to preserve total weight (i.e. sum to 1).")
    
    # Load and preprocess data.
    df = load_data()
    df_norm = preprocess_data(df)
    
    # List of available categories.
    available_categories = [
        'Restaurant', 'Nightclub', 'Museum', 'Theatre',
        'Attraction', 'Boutique', 'Temple', 'Church',
        'Palace', 'Antique Store'
    ]
    
    # Create a drop-down widget for category selection.
    selected_category = st.selectbox("Select Category", available_categories)
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

    st.subheader(f"Default Weights for {selected_category.title()}")
    # Show the default weights.
    for metric, weight in default_weights.items():
        st.write(f"**{metric}:** {weight}")
    
    st.markdown("---")
    st.subheader("Adjust Weights")
    st.write("Modify the raw weights below. They will be automatically normalized (so the sum equals 1) before recalculating the composite scores.")
    
    # Create a form to adjust weights.
    with st.form("weight_adjustments"):
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
    
    # Normalize the raw weights to sum to 1.
    total_raw = sum(raw_weights.values())
    if total_raw > 0:
        normalized_weights = {metric: val / total_raw for metric, val in raw_weights.items()}
    else:
        normalized_weights = default_weights  # Fallback if total is zero (should not occur)
    
    st.subheader("Normalized Weights")
    for metric, norm_val in normalized_weights.items():
        st.write(f"**{metric}:** {norm_val:.2f}")
    
    # Compute and display the ranking using the normalized weights.
    if submitted:
        df_custom = compute_custom_scores(df_filtered, normalized_weights)
        df_custom = df_custom.sort_values(by='Custom_Composite_Score', ascending=False)
        st.subheader(f"Top 10 {selected_category.title()} by Custom Composite Score")
        st.dataframe(df_custom[['Name', 'Custom_Composite_Score']].reset_index(drop=True))
    else:
        df_default = compute_custom_scores(df_filtered, get_default_category_weights(selected_category))
        df_default = df_default.sort_values(by='Custom_Composite_Score', ascending=False)
        st.subheader(f"Top 10 {selected_category.title()} by Composite Score")
        st.dataframe(df_default[['Name', 'Custom_Composite_Score']].reset_index(drop=True))

if __name__ == "__main__":
    main()
