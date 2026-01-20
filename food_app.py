"""
Simple Food Classifier & Recommender
1. Food Classifier: Input nutrition → Predict BMI category
2. Food Recommender: Input age/weight/height → Calculate BMI → Recommend foods
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Page config
st.set_page_config(page_title="Food Classifier & Recommender", page_icon="🍽️", layout="wide")

@st.cache_data
def load_data():
    """Load simplified food dataset"""
    return pd.read_csv('data/foods.csv')

def calculate_bmi(weight, height):
    """Calculate BMI"""
    height_m = height / 100
    bmi = weight / (height_m ** 2)
    return round(bmi, 1)

def get_bmi_category(bmi):
    """Get BMI category"""
    if bmi < 18.5:
        return "Underweight", "🔵"
    elif 18.5 <= bmi < 25:
        return "Normal", "🟢"
    elif 25 <= bmi < 30:
        return "Overweight", "🟡"
    else:
        return "Obese", "🔴"

def train_classifier(df):
    """Train KNN classifier"""
    # Features and target (excluding MealTime from features)
    X = df[['Calories', 'Protein', 'Carbs', 'Fat', 'Fiber']].values
    y = df['BMICategory'].values
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train KNN
    knn = KNeighborsClassifier(n_neighbors=3)  
    knn.fit(X_train_scaled, y_train)
    
    # Calculate accuracy
    y_pred = knn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    return knn, scaler, accuracy, len(X_train), len(X_test)

def classify_food(nutrition_values, knn, scaler):
    """Classify food and get similar foods"""
    # Scale input
    nutrition_scaled = scaler.transform([nutrition_values])
    
    # Get neighbors first
    distances, indices = knn.kneighbors(nutrition_scaled, n_neighbors=3)  # Changed from 5 to 3
    
    # Get the actual training data to count votes manually
    X_train = df[['Calories', 'Protein', 'Carbs', 'Fat', 'Fiber']].values
    y_train = df['BMICategory'].values
    X_train_split, _, y_train_split, _ = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
    
    # Count votes manually from the 5 nearest neighbors
    neighbor_votes = {}
    for train_idx in indices[0]:
        neighbor_category = y_train_split[train_idx]
        neighbor_votes[neighbor_category] = neighbor_votes.get(neighbor_category, 0) + 1
    
    # Find the category with most votes (proper tie-breaking)
    max_votes = max(neighbor_votes.values())
    tied_categories = [cat for cat, votes in neighbor_votes.items() if votes == max_votes]
    
    # If tie, pick the one with smallest average distance (closest neighbors)
    if len(tied_categories) > 1:
        category_distances = {}
        for i, train_idx in enumerate(indices[0]):
            neighbor_category = y_train_split[train_idx]
            if neighbor_category in tied_categories:
                if neighbor_category not in category_distances:
                    category_distances[neighbor_category] = []
                category_distances[neighbor_category].append(distances[0][i])
        
        # Pick category with smallest average distance
        avg_distances = {cat: np.mean(dists) for cat, dists in category_distances.items()}
        predicted_category = min(avg_distances.items(), key=lambda x: x[1])[0]
    else:
        predicted_category = tied_categories[0]
    
    # Get probabilities and classes for display
    probabilities = knn.predict_proba(nutrition_scaled)[0]
    classes = knn.classes_
    
    return predicted_category, probabilities, classes, distances[0], indices[0]

def recommend_foods_by_bmi(df, bmi_category, n_recs=6):
    """Recommend foods based on BMI category organized by meal times"""
    # Get foods for this BMI category
    category_foods = df[df['BMICategory'] == bmi_category].copy()
    
    if len(category_foods) == 0:
        return {}
    
    # Define ideal nutrition targets for each BMI category
    targets = {
        "Underweight": [400, 28, 45, 20, 4],  # High calories
        "Normal": [280, 22, 35, 12, 6],       # Balanced
        "Overweight": [180, 25, 20, 6, 8],    # Low calories, high protein
        "Obese": [120, 15, 15, 4, 6]          # Very low calories
    }
    
    target_nutrition = targets.get(bmi_category, [280, 22, 35, 12, 6])
    
    # Organize recommendations by meal time
    meal_recommendations = {}
    meal_times = ['Breakfast', 'Lunch', 'Dinner']
    
    for meal_time in meal_times:
        # Get foods for this meal time (including 'All' foods)
        meal_foods = category_foods[
            (category_foods['MealTime'] == meal_time) | 
            (category_foods['MealTime'] == 'All')
        ].copy()
        
        if len(meal_foods) == 0:
            meal_recommendations[meal_time] = pd.DataFrame()
            continue
        
        # Use KNN to find foods closest to target nutrition
        X = meal_foods[['Calories', 'Protein', 'Carbs', 'Fat', 'Fiber']].values
        
        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Scale target
        target_scaled = scaler.transform([target_nutrition])
        
        # Find nearest neighbors (limit to available foods)
        n_neighbors = min(2, len(meal_foods))  # Max 2 foods per meal
        if n_neighbors > 0:
            knn_recommender = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
            knn_recommender.fit(X_scaled)
            
            distances, indices = knn_recommender.kneighbors(target_scaled)
            meal_recommendations[meal_time] = meal_foods.iloc[indices[0]]
        else:
            meal_recommendations[meal_time] = pd.DataFrame()
    
    return meal_recommendations

# Load data and train model
df = load_data()
knn, scaler, accuracy, n_train, n_test = train_classifier(df)

# Header
st.title("🍽️ Food Classifier & Recommender")
st.markdown("*Two ML systems: Classify foods by nutrition OR Get recommendations by BMI*")

# Create two main sections
tab1, tab2 = st.tabs(["🔍 Food Classifier", "🎯 Food Recommender"])

# ============ TAB 1: FOOD CLASSIFIER ============
with tab1:
    st.subheader("🔍 Food Classifier")
    st.write("Enter nutrition values → AI predicts BMI category")
    
    # Input section - full width
    st.markdown("#### 📊 Enter Food Nutrition")
    
    # Create columns for input fields
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        calories = st.number_input("Calories", min_value=0, max_value=1000, value=200, step=10, key="cal1")
    with col2:
        protein = st.number_input("Protein (g)", min_value=0, max_value=100, value=12, step=1, key="prot1")
    with col3:
        carbs = st.number_input("Carbs (g)", min_value=0, max_value=200, value=25, step=1, key="carb1")
    with col4:
        fat = st.number_input("Fat (g)", min_value=0, max_value=100, value=5, step=1, key="fat1")
    with col5:
        fiber = st.number_input("Fiber (g)", min_value=0, max_value=50, value=4, step=1, key="fib1")
    
    # Center the classify button
    col_center = st.columns([1, 1, 1])
    with col_center[1]:
        classify_clicked = st.button("🔍 Classify Food", type="primary", key="classify1", use_container_width=True)
    
    # Results section - appears below when button is clicked
    if classify_clicked:
        # Classify
        nutrition_values = [calories, protein, carbs, fat, fiber]
        predicted_category, probabilities, classes, distances, indices = classify_food(
            nutrition_values, knn, scaler
        )
        
        # Show prediction
        category_colors = {"Underweight": "#3498db", "Normal": "#2ecc71", "Overweight": "#f39c12", "Obese": "#e74c3c"}
        category_emojis = {"Underweight": "🔵", "Normal": "🟢", "Overweight": "🟡", "Obese": "🔴"}
        
        color = category_colors.get(predicted_category, "#000000")
        emoji = category_emojis.get(predicted_category, "")
        
        # Create clean structured results display
        st.markdown("---")
            
        # Large prediction card with gradient
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {color}, {color}dd);
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            color: white;
            font-size: 24px;
            font-weight: bold;
            margin: 20px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        ">
            {emoji} PREDICTION: {predicted_category.upper()}
        </div>
        """, unsafe_allow_html=True)
        
        # Create three columns for organized display
        col_a, col_b, col_c = st.columns(3)
        
        # Count actual votes from all 3 neighbors
        X_train = df[['Calories', 'Protein', 'Carbs', 'Fat', 'Fiber']].values
        y_train = df['BMICategory'].values
        X_train_split, _, y_train_split, _ = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
        
        neighbor_votes = {}
        for train_idx in indices:
            neighbor_category = y_train_split[train_idx]
            neighbor_votes[neighbor_category] = neighbor_votes.get(neighbor_category, 0) + 1
        
        # COLUMN 1: KNN Neighbors
        with col_a:
            st.markdown("""
            <div style="
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                border-left: 4px solid #007bff;
                margin-bottom: 10px;
            ">
                <h4 style="color: #007bff; margin: 0 0 15px 0;">🔍 Similar Foods Used</h4>
            </div>
            """, unsafe_allow_html=True)
            
            for i, train_idx in enumerate(indices):
                distance = distances[i]
                similarity = max(0, (2 - distance) / 2 * 100)
                
                # Find matching food in original dataset
                train_nutrition = X_train_split[train_idx]
                train_category = y_train_split[train_idx]
                
                # Find the actual food
                matching_food = None
                for _, row in df.iterrows():
                    row_nutrition = [row['Calories'], row['Protein'], row['Carbs'], row['Fat'], row['Fiber']]
                    if np.allclose(train_nutrition, row_nutrition, rtol=1e-5):
                        matching_food = row
                        break
                
                if matching_food is not None:
                    food_emoji = category_emojis.get(train_category, '')
                    food_color = category_colors.get(train_category, "#666666")
                    
                    st.markdown(f"""
                    <div style="
                        background-color: white;
                        padding: 12px;
                        border-radius: 8px;
                        margin: 8px 0;
                        border-left: 3px solid {food_color};
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    ">
                        <strong>#{i+1} {matching_food['Name']}</strong><br>
                        <span style="color: {food_color};">{food_emoji} {train_category}</span><br>
                        <span style="color: #666; font-size: 14px;">📊 {similarity:.0f}% similar</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    food_emoji = category_emojis.get(train_category, '')
                    food_color = category_colors.get(train_category, "#666666")
                    
                    st.markdown(f"""
                    <div style="
                        background-color: white;
                        padding: 12px;
                        border-radius: 8px;
                        margin: 8px 0;
                        border-left: 3px solid {food_color};
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    ">
                        <strong>#{i+1} Training Food</strong><br>
                        <span style="color: {food_color};">{food_emoji} {train_category}</span><br>
                        <span style="color: #666; font-size: 14px;">📊 {similarity:.0f}% similar</span>
                    </div>
                    """, unsafe_allow_html=True)
        
        # COLUMN 2: Vote Results
        with col_b:
            st.markdown("""
            <div style="
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                border-left: 4px solid #28a745;
                margin-bottom: 10px;
            ">
                <h4 style="color: #28a745; margin: 0 0 15px 0;">🗳️ Voting Results</h4>
            </div>
            """, unsafe_allow_html=True)
            
            for category, votes in sorted(neighbor_votes.items(), key=lambda x: x[1], reverse=True):
                percentage = votes / 3 * 100
                vote_emoji = category_emojis.get(category, '')
                vote_color = category_colors.get(category, "#666666")
                bar_width = percentage
                
                st.markdown(f"""
                <div style="
                    background-color: white;
                    padding: 15px;
                    border-radius: 8px;
                    margin: 8px 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span><strong>{vote_emoji} {category}</strong></span>
                        <span style="color: {vote_color}; font-weight: bold;">{votes}/3 votes</span>
                    </div>
                    <div style="
                        background-color: #e9ecef;
                        height: 8px;
                        border-radius: 4px;
                        margin: 8px 0;
                        overflow: hidden;
                    ">
                        <div style="
                            background-color: {vote_color};
                            height: 100%;
                            width: {bar_width}%;
                            border-radius: 4px;
                        "></div>
                    </div>
                    <div style="text-align: center; color: #666; font-size: 14px;">
                        {percentage:.0f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # COLUMN 3: Confidence Scores
        with col_c:
            st.markdown("""
            <div style="
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                border-left: 4px solid #ffc107;
                margin-bottom: 10px;
            ">
                <h4 style="color: #ffc107; margin: 0 0 15px 0;">📊 Confidence Scores</h4>
            </div>
            """, unsafe_allow_html=True)
            
            for i, class_name in enumerate(classes):
                confidence = probabilities[i] * 100
                conf_emoji = category_emojis.get(class_name, '')
                conf_color = category_colors.get(class_name, "#666666")
                conf_width = confidence
                
                st.markdown(f"""
                <div style="
                    background-color: white;
                    padding: 15px;
                    border-radius: 8px;
                    margin: 8px 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span><strong>{conf_emoji} {class_name}</strong></span>
                        <span style="color: {conf_color}; font-weight: bold;">{confidence:.1f}%</span>
                    </div>
                    <div style="
                        background-color: #e9ecef;
                        height: 6px;
                        border-radius: 3px;
                        margin: 8px 0;
                        overflow: hidden;
                    ">
                        <div style="
                            background-color: {conf_color};
                            height: 100%;
                            width: {conf_width}%;
                            border-radius: 3px;
                        "></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Additional info section
        st.markdown("---")
        
        # Check for ties and validation
        max_votes = max(neighbor_votes.values())
        tied_categories = [cat for cat, votes in neighbor_votes.items() if votes == max_votes]
        
        if len(tied_categories) > 1:
            st.markdown(f"""
            <div style="background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 8px; margin: 10px 0;">
                <strong>⚖️ Tie Resolution:</strong> Categories {tied_categories} each got {max_votes} votes. 
                Resolved by choosing the category with closest average distance.
            </div>
            """, unsafe_allow_html=True)
        
        majority_winner = max(neighbor_votes.items(), key=lambda x: x[1])
        if len(tied_categories) == 1:
            if predicted_category == majority_winner[0]:
                st.markdown(f"""
                <div style="background-color: #d4edda; border: 1px solid #c3e6cb; padding: 15px; border-radius: 8px; margin: 10px 0;">
                    <strong>✅ Validation:</strong> Prediction matches majority vote: {majority_winner[0]} ({majority_winner[1]}/3 votes)
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color: #d1ecf1; border: 1px solid #bee5eb; padding: 15px; border-radius: 8px; margin: 10px 0;">
                <strong>✅ Tie Resolved:</strong> Chose {predicted_category} (closest average distance among tied categories)
            </div>
            """, unsafe_allow_html=True)

# ============ TAB 2: FOOD RECOMMENDER ============
with tab2:
    st.subheader("🎯 Food Recommender")
    st.write("Enter your details → Calculate BMI → Get personalized food recommendations")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 👤 Enter Your Details")
        
        age = st.number_input("Age (years)", min_value=10, max_value=100, value=25, key="age2")
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70, key="weight2")
        height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170, key="height2")
        gender = st.selectbox("Gender", ["Male", "Female"], key="gender2")
        
        if st.button("🎯 Get Food Recommendations", type="primary", key="recommend2"):
            # Calculate BMI
            bmi = calculate_bmi(weight, height)
            category, emoji = get_bmi_category(bmi)
            
            # Store results in session state
            st.session_state.bmi_result = {
                'bmi': bmi,
                'category': category,
                'emoji': emoji
            }
    
    with col2:
        st.markdown("#### 📊 Your Results")
        
        if 'bmi_result' in st.session_state:
            result = st.session_state.bmi_result
            bmi = result['bmi']
            category = result['category']
            emoji = result['emoji']
            
            # Show BMI result
            category_colors = {"Underweight": "#3498db", "Normal": "#2ecc71", "Overweight": "#f39c12", "Obese": "#e74c3c"}
            color = category_colors.get(category, "#000000")
            
            st.markdown(f"""
            <div style="background-color: {color}; padding: 15px; border-radius: 10px; text-align: center; color: white; font-size: 18px;">
                <strong>BMI: {bmi} → {emoji} {category}</strong>
            </div>
            """, unsafe_allow_html=True)
            
            # Get recommendations organized by meal times
            recommendations = recommend_foods_by_bmi(df, category)
            
            if any(len(meal_foods) > 0 for meal_foods in recommendations.values()):
                st.markdown("#### 🍽️ Recommended Foods by Meal Time")
                
                # Create columns for each meal
                meal_cols = st.columns(3)
                meal_times = ['Breakfast', 'Lunch', 'Dinner']
                meal_emojis = {'Breakfast': '🌅', 'Lunch': '☀️', 'Dinner': '🌙'}
                
                for i, meal_time in enumerate(meal_times):
                    with meal_cols[i]:
                        st.markdown(f"### {meal_emojis[meal_time]} {meal_time}")
                        
                        meal_foods = recommendations.get(meal_time, pd.DataFrame())
                        
                        if len(meal_foods) > 0:
                            for _, food in meal_foods.iterrows():
                                with st.expander(f"{emoji} {food['Name']} - {food['Calories']} cal"):
                                    col_a, col_b = st.columns(2)
                                    with col_a:
                                        st.write(f"**Nutrition:**")
                                        st.write(f"• Protein: {food['Protein']}g")
                                        st.write(f"• Carbs: {food['Carbs']}g")
                                        st.write(f"• Fat: {food['Fat']}g")
                                        st.write(f"• Fiber: {food['Fiber']}g")
                                    with col_b:
                                        st.success(f"**Why good for {category}:**")
                                        st.write(food['Reason'])
                        else:
                            st.info(f"No {category} foods available for {meal_time}")
            else:
                st.warning("No recommendations found for this BMI category.")

# Model info
st.markdown("---")
st.subheader("📈 Model Information")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Model Accuracy", f"{accuracy:.1%}")
with col2:
    st.metric("Training Foods", n_train)
with col3:
    st.metric("Test Foods", n_test)

st.info("**Algorithm**: K-Nearest Neighbors (K=3) with StandardScaler normalization")