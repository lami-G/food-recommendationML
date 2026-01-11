"""
Ethiopian Food Recommendation System - Advanced ML Version
A school project with weighted KNN model for BMI-based diet recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import os

# Page configuration
st.set_page_config(
    page_title="Ethiopian Food Recommender",
    page_icon="🇪🇹",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_data():
    """Load the food dataset"""
    data_path = os.path.join(BASE_DIR, 'data', 'ethiopian_foods.csv')
    df = pd.read_csv(data_path)
    return df

def calculate_bmi(weight, height):
    """Calculate BMI"""
    height_m = height / 100
    bmi = weight / (height_m ** 2)
    return round(bmi, 1)

def get_bmi_category(bmi):
    """Get BMI category with details"""
    if bmi < 18.5:
        return "Underweight", "🔵", "#3498db", "You need to gain weight healthily"
    elif 18.5 <= bmi < 25:
        return "Normal", "🟢", "#2ecc71", "Maintain your healthy weight"
    elif 25 <= bmi < 30:
        return "Overweight", "🟡", "#f39c12", "Focus on gradual weight loss"
    else:
        return "Obese", "🔴", "#e74c3c", "Prioritize weight loss for better health"

def calculate_daily_calories(weight, height, age, gender, bmi_category):
    """Calculate recommended daily calories"""
    if gender == "Male":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    
    maintenance_calories = bmr * 1.55
    
    if bmi_category == "Underweight":
        daily_calories = maintenance_calories + 500
        goal = "gain weight"
    elif bmi_category == "Normal":
        daily_calories = maintenance_calories
        goal = "maintain weight"
    elif bmi_category == "Overweight":
        daily_calories = maintenance_calories - 400
        goal = "lose weight gradually"
    else:
        daily_calories = maintenance_calories - 600
        goal = "lose weight safely"
    
    return round(daily_calories), goal

def get_bmi_weights(bmi_category):
    """
    Get feature weights based on BMI category
    This is the KEY to making recommendations BMI-specific
    Weights: [Calories, Protein, Carbs, Fat, Fiber]
    """
    if bmi_category == "Underweight":
        # Prioritize HIGH calories and fat for weight gain
        return np.array([3.0, 1.5, 2.0, 2.5, 0.5])
    elif bmi_category == "Normal":
        # Balanced weights
        return np.array([1.0, 1.5, 1.0, 1.0, 1.0])
    elif bmi_category == "Overweight":
        # Prioritize LOW calories, HIGH fiber and protein
        return np.array([2.0, 2.0, 1.0, 1.5, 2.5])
    else:  # Obese
        # Strongly prioritize LOWEST calories, HIGHEST fiber
        return np.array([3.5, 1.5, 1.0, 2.0, 3.0])

def get_target_nutrition(bmi_category):
    """Get ideal target nutrition based on BMI"""
    # [Calories, Protein, Carbs, Fat, Fiber]
    if bmi_category == "Underweight":
        return [400, 28, 50, 20, 4]  # High calories
    elif bmi_category == "Normal":
        return [280, 22, 35, 12, 6]  # Balanced
    elif bmi_category == "Overweight":
        return [180, 20, 25, 6, 8]   # Lower calories
    else:  # Obese
        return [120, 15, 15, 4, 6]   # Lowest calories

def weighted_knn_recommend(df, bmi_category, meal_time, n_recommendations=3):
    """
    Advanced KNN with weighted features based on BMI category
    This makes the model 'smarter' by prioritizing different nutrients
    """
    # Filter by meal time
    meal_foods = df[(df['MealTime'] == meal_time) | (df['MealTime'] == 'All')].copy()
    
    # Also filter by BMI category preference (but include some variety)
    bmi_foods = meal_foods[meal_foods['BMICategory'] == bmi_category]
    
    # If not enough BMI-specific foods, add Normal category
    if len(bmi_foods) < n_recommendations:
        normal_foods = meal_foods[meal_foods['BMICategory'] == 'Normal']
        bmi_foods = pd.concat([bmi_foods, normal_foods]).drop_duplicates()
    
    if len(bmi_foods) == 0:
        return None
    
    # Feature columns
    nutrition_cols = ['Calories', 'Protein', 'Carbs', 'Fat', 'Fiber']
    X = bmi_foods[nutrition_cols].values.astype(float)
    
    # Get BMI-specific weights
    weights = get_bmi_weights(bmi_category)
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply weights to features (this is the advanced part!)
    X_weighted = X_scaled * weights
    
    # Create KNN model
    n_neighbors = min(n_recommendations, len(bmi_foods))
    model = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    model.fit(X_weighted)
    
    # Get target nutrition and apply same transformation
    target = np.array([get_target_nutrition(bmi_category)])
    target_scaled = scaler.transform(target)
    target_weighted = target_scaled * weights
    
    # Find nearest neighbors
    distances, indices = model.kneighbors(target_weighted)
    
    return bmi_foods.iloc[indices[0]]

def calculate_food_score(food, bmi_category):
    """Calculate a suitability score for each food based on BMI"""
    weights = get_bmi_weights(bmi_category)
    target = get_target_nutrition(bmi_category)
    
    # Calculate how close the food is to target
    features = [food['Calories'], food['Protein'], food['Carbs'], food['Fat'], food['Fiber']]
    
    score = 0
    for i, (f, t, w) in enumerate(zip(features, target, weights)):
        if bmi_category in ["Underweight"]:
            # For underweight, higher calories = better score
            if i == 0:  # Calories
                score += (f / max(t, 1)) * w
            else:
                score += (1 - abs(f - t) / max(t, 1)) * w
        else:
            # For overweight/obese, lower calories = better score
            if i == 0:  # Calories
                score += (t / max(f, 1)) * w
            else:
                score += (1 - abs(f - t) / max(t, 1)) * w
    
    return round(score, 2)

def get_diet_advice(bmi_category):
    """Get dietary advice based on BMI category"""
    advice = {
        "Underweight": {
            "title": "🎯 Goal: Healthy Weight Gain",
            "tips": [
                "Eat HIGH calorie foods: Doro Wot, Kitfo, Genfo, Chechebsa",
                "Include healthy fats from kibbe (spiced butter)",
                "Eat 5-6 meals per day instead of 3",
                "Add calorie-dense snacks: Kolo, Dabo Kolo, Peanut Butter",
                "Drink smoothies and Atmit for extra calories"
            ],
            "avoid": "Don't fill up on low-calorie vegetables first",
            "color": "#3498db"
        },
        "Normal": {
            "title": "🎯 Goal: Maintain Healthy Weight",
            "tips": [
                "Keep eating balanced Ethiopian meals",
                "Mix protein (Tibs, Shiro) with vegetables (Gomen)",
                "Enjoy variety with Beyaynetu platter",
                "Moderate portion sizes",
                "Stay active and hydrated"
            ],
            "avoid": "Avoid excessive fried foods",
            "color": "#2ecc71"
        },
        "Overweight": {
            "title": "🎯 Goal: Gradual Weight Loss",
            "tips": [
                "Choose HIGH fiber foods: Misir Wot, Yekik Alicha",
                "Eat more vegetables: Gomen, Atkilt Wot",
                "Reduce injera portions (1-2 pieces max)",
                "Choose lean proteins: Grilled Fish, Turkey",
                "Snack on Lebleb (roasted chickpeas) not Kolo"
            ],
            "avoid": "Avoid high-fat dishes like Kitfo and fried Sambusa",
            "color": "#f39c12"
        },
        "Obese": {
            "title": "🎯 Goal: Safe Weight Loss",
            "tips": [
                "Focus on LOWEST calorie options: Gomen, Tikil Gomen",
                "Eat large portions of vegetables to feel full",
                "Choose steamed/grilled over fried",
                "Limit injera to 1 small piece or skip",
                "Snack on cucumber, carrots, apple slices"
            ],
            "avoid": "Avoid all high-calorie foods: Kitfo, Doro Wot, Kolo, fried foods",
            "color": "#e74c3c"
        }
    }
    return advice.get(bmi_category, advice["Normal"])

# Load data
df = load_data()

# Header
st.title("🇪🇹 Ethiopian Food Recommendation System")
st.markdown("*Advanced ML Project - Weighted KNN for BMI-based Diet Recommendations*")
st.markdown("---")

# Sidebar
st.sidebar.title("📋 Menu")
page = st.sidebar.radio("Choose:", [
    "🏠 Home",
    "⚖️ BMI & Diet Plan",
    "📂 Browse Foods",
    "🧠 How the ML Works"
])

# ============ HOME PAGE ============
if page == "🏠 Home":
    st.header("Welcome! እንኳን ደህና መጡ 👋")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("What This Project Does")
        st.write("""
        1. **Enter your details** - Age, weight, height, gender
        2. **Calculate your BMI** - Body Mass Index
        3. **Get SMART recommendations** - Based on YOUR BMI category
        4. **Understand WHY** - Each food explains its benefit
        """)
        
        st.subheader("🧠 Advanced ML Features")
        st.write("""
        - **Weighted K-Nearest Neighbors** algorithm
        - Different weights for each BMI category
        - Underweight → Prioritizes HIGH calories
        - Obese → Prioritizes LOW calories, HIGH fiber
        """)
        
    with col2:
        st.subheader("📊 Dataset Statistics")
        st.write(f"**Total Foods:** {len(df)}")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.write("**By Cuisine:**")
            cuisine_counts = df['Cuisine'].value_counts()
            for cuisine, count in cuisine_counts.items():
                st.write(f"- {cuisine}: {count}")
        
        with col_b:
            st.write("**By BMI Category:**")
            bmi_counts = df['BMICategory'].value_counts()
            for bmi, count in bmi_counts.items():
                emoji = {"Underweight": "🔵", "Normal": "🟢", "Overweight": "🟡", "Obese": "🔴"}.get(bmi, "")
                st.write(f"- {emoji} {bmi}: {count}")

# ============ BMI & DIET PLAN ============
elif page == "⚖️ BMI & Diet Plan":
    st.header("⚖️ BMI Calculator & Smart Diet Recommendations")
    
    st.subheader("Step 1: Enter Your Information")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        age = st.number_input("Age (years)", min_value=10, max_value=100, value=25)
    with col2:
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
    with col3:
        height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
    with col4:
        gender = st.selectbox("Gender", ["Male", "Female"])
    
    if st.button("🔍 Calculate BMI & Get Smart Recommendations", type="primary"):
        
        bmi = calculate_bmi(weight, height)
        category, emoji, color, health_msg = get_bmi_category(bmi)
        daily_calories, goal = calculate_daily_calories(weight, height, age, gender, category)
        
        st.markdown("---")
        
        # BMI Results
        st.subheader("Step 2: Your BMI Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Your BMI", f"{bmi} kg/m²")
        with col2:
            st.metric("Category", f"{emoji} {category}")
        with col3:
            st.metric("Daily Calories", f"{daily_calories} kcal")
        
        st.info(f"**BMI: {bmi}** → **{category}** | Goal: **{goal}** | Target: **{daily_calories} cal/day**")
        
        # Diet Advice
        st.markdown("---")
        st.subheader("Step 3: Personalized Diet Advice")
        
        advice = get_diet_advice(category)
        st.markdown(f"### {advice['title']}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**✅ Recommended:**")
            for tip in advice['tips']:
                st.write(f"• {tip}")
        with col2:
            st.write("**❌ Avoid:**")
            st.write(f"• {advice['avoid']}")

        # Food Recommendations
        st.markdown("---")
        st.subheader("Step 4: Your Smart Diet Plan 🍽️")
        
        breakfast_cal = int(daily_calories * 0.30)
        lunch_cal = int(daily_calories * 0.40)
        dinner_cal = int(daily_calories * 0.30)
        
        col1, col2, col3 = st.columns(3)
        
        # BREAKFAST
        with col1:
            st.markdown(f"### 🌅 Breakfast (~{breakfast_cal} cal)")
            
            breakfast_foods = weighted_knn_recommend(df, category, "Breakfast", 3)
            if breakfast_foods is not None and len(breakfast_foods) > 0:
                for _, food in breakfast_foods.iterrows():
                    score = calculate_food_score(food, category)
                    cuisine_flag = "🇪🇹" if food['Cuisine'] == "Ethiopian" else "🌍"
                    
                    with st.expander(f"{cuisine_flag} {food['Name']} ({food['Calories']} cal)"):
                        st.write(f"**{food['Description']}**")
                        st.success(f"💡 **Why:** {food['Reason']}")
                        st.write(f"**Match Score:** {score}/10")
                        st.write(f"**Nutrition:** {food['Protein']}g protein | {food['Carbs']}g carbs | {food['Fat']}g fat | {food['Fiber']}g fiber")
                        st.write(f"**Ingredients:** {food['Ingredients'].replace(';', ', ')}")
        
        # LUNCH
        with col2:
            st.markdown(f"### ☀️ Lunch (~{lunch_cal} cal)")
            
            lunch_foods = weighted_knn_recommend(df, category, "Lunch", 3)
            if lunch_foods is not None and len(lunch_foods) > 0:
                for _, food in lunch_foods.iterrows():
                    score = calculate_food_score(food, category)
                    cuisine_flag = "🇪🇹" if food['Cuisine'] == "Ethiopian" else "🌍"
                    
                    with st.expander(f"{cuisine_flag} {food['Name']} ({food['Calories']} cal)"):
                        st.write(f"**{food['Description']}**")
                        st.success(f"💡 **Why:** {food['Reason']}")
                        st.write(f"**Match Score:** {score}/10")
                        st.write(f"**Nutrition:** {food['Protein']}g protein | {food['Carbs']}g carbs | {food['Fat']}g fat | {food['Fiber']}g fiber")
                        st.write(f"**Ingredients:** {food['Ingredients'].replace(';', ', ')}")
        
        # DINNER
        with col3:
            st.markdown(f"### 🌙 Dinner (~{dinner_cal} cal)")
            
            dinner_foods = weighted_knn_recommend(df, category, "Dinner", 3)
            if dinner_foods is not None and len(dinner_foods) > 0:
                for _, food in dinner_foods.iterrows():
                    score = calculate_food_score(food, category)
                    cuisine_flag = "🇪🇹" if food['Cuisine'] == "Ethiopian" else "🌍"
                    
                    with st.expander(f"{cuisine_flag} {food['Name']} ({food['Calories']} cal)"):
                        st.write(f"**{food['Description']}**")
                        st.success(f"💡 **Why:** {food['Reason']}")
                        st.write(f"**Match Score:** {score}/10")
                        st.write(f"**Nutrition:** {food['Protein']}g protein | {food['Carbs']}g carbs | {food['Fat']}g fat | {food['Fiber']}g fiber")
                        st.write(f"**Ingredients:** {food['Ingredients'].replace(';', ', ')}")
        
        # Summary
        st.markdown("---")
        st.subheader("📊 Daily Summary")
        
        summary_df = pd.DataFrame({
            "Meal": ["🌅 Breakfast", "☀️ Lunch", "🌙 Dinner", "📊 Total"],
            "Calories": [breakfast_cal, lunch_cal, dinner_cal, daily_calories],
            "Percentage": ["30%", "40%", "30%", "100%"]
        })
        st.table(summary_df)

# ============ BROWSE FOODS ============
elif page == "📂 Browse Foods":
    st.header("📂 Browse All Foods")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        meal_filter = st.selectbox("Meal Time:", ["All"] + list(df['MealTime'].unique()))
    with col2:
        bmi_filter = st.selectbox("BMI Category:", ["All"] + list(df['BMICategory'].unique()))
    with col3:
        cuisine_filter = st.selectbox("Cuisine:", ["All"] + list(df['Cuisine'].unique()))
    with col4:
        veg_filter = st.selectbox("Vegetarian:", ["All", "Yes", "No"])
    
    filtered_df = df.copy()
    if meal_filter != "All":
        filtered_df = filtered_df[filtered_df['MealTime'] == meal_filter]
    if bmi_filter != "All":
        filtered_df = filtered_df[filtered_df['BMICategory'] == bmi_filter]
    if cuisine_filter != "All":
        filtered_df = filtered_df[filtered_df['Cuisine'] == cuisine_filter]
    if veg_filter != "All":
        filtered_df = filtered_df[filtered_df['IsVegetarian'] == veg_filter]
    
    st.write(f"**Showing {len(filtered_df)} foods**")
    
    # Sort by calories
    sort_order = st.radio("Sort by Calories:", ["Low to High", "High to Low"], horizontal=True)
    filtered_df = filtered_df.sort_values('Calories', ascending=(sort_order == "Low to High"))
    
    for _, food in filtered_df.iterrows():
        bmi_emoji = {"Underweight": "🔵", "Normal": "🟢", "Overweight": "🟡", "Obese": "🔴"}.get(food['BMICategory'], "")
        cuisine_flag = "🇪🇹" if food['Cuisine'] == "Ethiopian" else "🌍"
        
        with st.expander(f"{cuisine_flag} {bmi_emoji} {food['Name']} - {food['Calories']} cal"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**{food['Description']}**")
                st.write(f"**Cuisine:** {food['Cuisine']}")
                st.write(f"**Meal:** {food['MealTime']} | **Best for:** {food['BMICategory']}")
                st.success(f"💡 {food['Reason']}")
            with col2:
                st.write("**Nutrition:**")
                st.write(f"- Calories: {food['Calories']} kcal")
                st.write(f"- Protein: {food['Protein']}g")
                st.write(f"- Carbs: {food['Carbs']}g")
                st.write(f"- Fat: {food['Fat']}g")
                st.write(f"- Fiber: {food['Fiber']}g")

# ============ HOW ML WORKS ============
elif page == "🧠 How the ML Works":
    st.header("🧠 How the Machine Learning Works")
    
    st.subheader("1. The Problem")
    st.write("""
    We want to recommend foods based on a person's BMI category:
    - **Underweight** → Need HIGH calorie foods
    - **Normal** → Need BALANCED foods  
    - **Overweight** → Need LOWER calorie, HIGH fiber foods
    - **Obese** → Need LOWEST calorie, SIMPLE foods
    """)
    
    st.subheader("2. The Solution: Weighted K-Nearest Neighbors")
    st.write("""
    **Standard KNN** finds similar items based on features (Calories, Protein, Carbs, Fat, Fiber).
    
    **Our Weighted KNN** gives different importance to each feature based on BMI:
    """)
    
    # Show weights table
    weights_df = pd.DataFrame({
        "BMI Category": ["Underweight", "Normal", "Overweight", "Obese"],
        "Calories Weight": [3.0, 1.0, 2.0, 3.5],
        "Protein Weight": [1.5, 1.5, 2.0, 1.5],
        "Carbs Weight": [2.0, 1.0, 1.0, 1.0],
        "Fat Weight": [2.5, 1.0, 1.5, 2.0],
        "Fiber Weight": [0.5, 1.0, 2.5, 3.0]
    })
    st.table(weights_df)
    
    st.write("""
    **What this means:**
    - For **Underweight**: Calories (3.0) and Fat (2.5) are most important → finds HIGH calorie foods
    - For **Obese**: Calories (3.5) and Fiber (3.0) are most important → finds LOW calorie, HIGH fiber foods
    """)
    
    st.subheader("3. The Algorithm Steps")
    st.code("""
    1. Filter foods by meal time and BMI category
    2. Extract nutrition features: [Calories, Protein, Carbs, Fat, Fiber]
    3. Scale features using StandardScaler (mean=0, std=1)
    4. Apply BMI-specific weights to scaled features
    5. Use KNN to find foods closest to target nutrition
    6. Return top N recommendations
    """, language="text")
    
    st.subheader("4. Target Nutrition by BMI")
    target_df = pd.DataFrame({
        "BMI Category": ["Underweight", "Normal", "Overweight", "Obese"],
        "Target Calories": [400, 280, 180, 120],
        "Target Protein": [28, 22, 20, 15],
        "Target Carbs": [50, 35, 25, 15],
        "Target Fat": [20, 12, 6, 4],
        "Target Fiber": [4, 6, 8, 6]
    })
    st.table(target_df)
    
    st.subheader("5. Technologies Used")
    st.write("""
    - **Python** - Programming language
    - **Pandas** - Data manipulation
    - **NumPy** - Numerical operations
    - **Scikit-learn** - KNN, StandardScaler
    - **Streamlit** - Web interface
    """)

# Footer
st.markdown("---")
st.markdown("*Made with ❤️ | Ethiopian Food Recommender - Advanced ML School Project 🇪🇹*")
