import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ---------- Helper Functions ----------

@st.cache(allow_output_mutation=True)
def load_data():
    # Replace 'dummy_dataset.csv' with the path to your actual dataset.
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
    # Set a default 'Uniqueness' metric (for boutiques and antique stores)
    df['Uniqueness'] = 1  # You can later refine this metric if needed.
    
    # Create an Accessibility metric:
    # We assume "Public Transport Proximity (m)" is numeric; lower values are better.
    scaler_access = MinMaxScaler()
    df['Public Transport Proximity Norm'] = scaler_access.fit_transform(df[['Public Transport Proximity (m)']])
    df['Accessibility'] = 1 - df['Public Transport Proximity Norm']
    
    # List of metrics that will be used across various categories.
    metric_cols = [
        "Rating", "Reviews", "Footfall", "Repeat Visitors (%)", "Avg Stay Duration (mins)",
        "Price Range Numeric", "Popularity Index", "Instagram Mentions",
        "Historical Mentions", "Heritage Status", "Years Since Establishment",
        "TripAdvisor Ranking", "Google Check-ins"
    ]
    
    # Normalize these metrics using MinMaxScaler.
    scaler_metrics = MinMaxScaler()
    df_norm = df.copy()
    df_norm[metric_cols] = scaler_metrics.fit_transform(df[metric_cols])
    # Carry over the precomputed Accessibility and Uniqueness metrics.
    df_norm['Accessibility'] = df['Accessibility']
    df_norm['Uniqueness'] = df['Uniqueness']
    
    return df_norm

def get_category_weights(cat):
    """
    Returns the weight dictionary for a given category.
    For 'temple' or 'church', both use the same weights.
    """
    cat_lower = cat.strip().lower()
    if cat_lower in ['temple', 'church']:
        return {
            'Heritage Status': 0.35,
            'Years Since Establishment': 0.25,
            'Footfall': 0.25,
            'Accessibility': 0.15
        }
    # Weight dictionaries for other categories:
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
            'Instagram Mentions': 0.15,  # Proxy for social media mentions.
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

def assign_composite_score(df_norm):
    df_norm['Composite_Score'] = df_norm.apply(
        lambda row: compute_category_score(row, get_category_weights(row['Category'])), axis=1
    )
    return df_norm

# ---------- Main App ----------
def main():
    st.title("Trivandrum Top 10 Curated Venues")
    st.write("Select a category from the drop-down below to view the top 10 curated venues based on our composite score.")
    
    # Load and preprocess data.
    df = load_data()
    df_norm = preprocess_data(df)
    df_norm = assign_composite_score(df_norm)
    
    # List of available categories (adjust based on your dataset)
    available_categories = [
        'Restaurant', 'Nightclub', 'Museum', 'Theatre',
        'Attraction', 'Boutique', 'Temple', 'Church',
        'Palace', 'Antique Store'
    ]
    
    # Create a drop-down widget.
    category = st.selectbox("Select Category", available_categories)
    
    # Filter data based on the selected category.
    cat_lower = category.strip().lower()
    if cat_lower in ['temple', 'church']:
        # If the user selects either "Temple" or "Church", we filter for both.
        df_filtered = df_norm[df_norm['Category'].str.strip().str.lower().isin(['temple', 'church'])]
    else:
        df_filtered = df_norm[df_norm['Category'].str.strip().str.lower() == cat_lower]
    
    if df_filtered.empty:
        st.warning(f"No entries found for the category: {category}")
    else:
        top_venues = df_filtered.sort_values(by='Composite_Score', ascending=False).head(10)
        st.subheader(f"Top 10 {category.title()} by Composite Score")
        st.dataframe(top_venues[['Name', 'Composite_Score']].reset_index(drop=True))

if __name__ == "__main__":
    main()
