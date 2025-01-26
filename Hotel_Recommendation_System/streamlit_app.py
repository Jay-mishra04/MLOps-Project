import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('hotel_features.csv') 

hotel_features = load_data()

# Load users hotel history for insights
@st.cache_data
def load_users_hotel_data():
    return pd.read_csv("users_hotel_history.csv")  

users_hotel = load_users_hotel_data()

# Define the recommendation system function
def recommendation_system(place, price_range, days_range):
    """
    Recommend hotels based on place, price range, and days range.
    """
    # Filter by place
    filtered_hotels_by_place = hotel_features[hotel_features['place'] == place]
    
    # Filter by price range
    filtered_hotels_by_price = filtered_hotels_by_place[
        (filtered_hotels_by_place['price'] >= price_range[0]) & 
        (filtered_hotels_by_place['price'] <= price_range[1])
    ]
    
    # Filter by days range
    filtered_hotels_by_day = filtered_hotels_by_price[
        (filtered_hotels_by_price['days'] >= days_range[0]) & 
        (filtered_hotels_by_price['days'] <= days_range[1])
    ]
    
    return filtered_hotels_by_day

# Sidebar Navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Hotel Recommendations", "Insights and Visualizations", "Interactive Data Exploration"])

# Hotel Recommendations Section
if section == "Hotel Recommendations":
    st.title("Hotel Recommendation System")
    st.markdown("This section allows you to get hotel recommendations based on your preferences.")

    # Dropdown for place selection
    place = st.selectbox("Select a Place:", hotel_features['place'].unique())

    # Slider for price range selection
    price_range = st.slider("Select Price Range:", 
                            min_value=int(hotel_features['price'].min()), 
                            max_value=int(hotel_features['price'].max()), 
                            value=(100, 300))

    # Slider for days range selection
    days_range = st.slider("Select Stay Duration (Days):", 
                           min_value=int(hotel_features['days'].min()), 
                           max_value=int(hotel_features['days'].max()), 
                           value=(1, 4))

    # Button to trigger recommendations
    if st.button("Get Recommendations"):
        recommendations = recommendation_system(place, price_range, days_range)
        if not recommendations.empty:
            st.write("Recommended Hotels:")
            st.dataframe(recommendations)
        else:
            st.write("No hotels found matching your criteria.")

# Insights and Visualizations Section
elif section == "Insights and Visualizations":
    st.title("Insights and Visualizations")
    st.markdown("This section provides insights and visualizations to understand booking trends.")

    # Top Destinations
    st.subheader("Top Destinations")
    top_destinations = users_hotel['place'].value_counts().head(5).sort_values(ascending=False)
    st.bar_chart(top_destinations)

    # Top Hotels
    st.subheader("Top Hotels")
    top_hotels = users_hotel['hotelName'].value_counts().head(5).sort_values(ascending=False)
    st.bar_chart(top_hotels)

    # Spending Trends by Place
    st.subheader("Spending Trends by Place")
    spending_by_place = users_hotel.groupby('place')['price'].sum().sort_values(ascending=False)
    st.bar_chart(spending_by_place)

    # **New Chart: Number of Hotels in Different Places**
    st.subheader("Number of Hotels in Different Places")
    hotels_by_place = hotel_features.groupby('place')['hotelName'].nunique().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    hotels_by_place.plot(kind='bar', color='teal', ax=ax)
    ax.set_title("Number of Hotels in Different Places")
    ax.set_xlabel("Place")
    ax.set_ylabel("Number of Hotels")
    st.pyplot(fig)

    # Spending vs. Stay Duration
    st.subheader("Spending vs. Stay Duration")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=users_hotel, x='days', y='price', palette='coolwarm', ax=ax)
    ax.set_title("Spending vs. Stay Duration")
    ax.set_xlabel("Days of Stay")
    ax.set_ylabel("Total Spending")
    st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    correlation_matrix = users_hotel[['age', 'month', 'year', 'price', 'days']].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Correlation Between Features")
    st.pyplot(fig)

    # Monthly Booking Trends
    st.subheader("Monthly Booking Trends")
    monthly_trends = users_hotel['month'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    monthly_trends.plot(kind='line', marker='o', color='purple', ax=ax)
    ax.set_title("Monthly Booking Trends")
    ax.set_xlabel("Month")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # Yearly Booking Trends
    st.subheader("Yearly Booking Trends")
    yearly_trends = users_hotel['year'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    yearly_trends.plot(kind='line', marker='o', color='blue', ax=ax)
    ax.set_title("Yearly Booking Trends")
    ax.set_xlabel("Year")
    ax.set_ylabel("Count")
    st.pyplot(fig)


# Interactive Data Exploration Section
elif section == "Interactive Data Exploration":
    st.title("Interactive Data Exploration")
    st.markdown("This section allows you to explore the dataset interactively.")

    # Filters for exploration
    selected_place = st.selectbox("Filter by Place:", users_hotel['place'].unique())
    selected_year = st.selectbox("Filter by Year:", users_hotel['year'].unique())
    selected_month = st.selectbox("Filter by month:", users_hotel['month'].unique())

    selected_price_range = st.slider("Filter by Price Range:", 
                                     min_value=int(users_hotel['price'].min()), 
                                     max_value=int(users_hotel['price'].max()), 
                                     value=(100, 300))
    selected_days_range = st.slider("Filter by Stay Duration (Days):", 
                                    min_value=int(users_hotel['days'].min()), 
                                    max_value=int(users_hotel['days'].max()), 
                                    value=(1, 4))

    # Apply filters
    filtered_data = users_hotel[
        (users_hotel['place'] == selected_place) &
        (users_hotel['year'] == selected_year) &
        (users_hotel['month'] == selected_month) &
        (users_hotel['price'] >= selected_price_range[0]) & 
        (users_hotel['price'] <= selected_price_range[1]) &
        (users_hotel['days'] >= selected_days_range[0]) & 
        (users_hotel['days'] <= selected_days_range[1])
    ]

    # Display filtered data
    st.write("Filtered Data:")
    st.dataframe(filtered_data)
