# 🍽️ Food App Code Explanation - Complete Beginner's Guide

*Think of this as your friend explaining every single line of code in simple words*

## 🎯 What This File Does
`food_app.py` is like a **digital restaurant menu with a smart waiter**. You tell it about food nutrition or your body details, and it gives you smart recommendations using artificial intelligence.

---

## 📚 Part 1: Importing Tools (Lines 1-12)

```python
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```

**What's happening here?**
- **Think of imports like borrowing tools from a toolbox**
- `streamlit` = Tool to make websites easily (like WordPress but for data)
- `pandas` = Tool to work with spreadsheets (like Excel but in Python)
- `numpy` = Tool for math calculations (like a super calculator)
- `sklearn` = Collection of AI tools (like having a smart robot assistant)

**Real-world analogy**: You're preparing to cook, so you gather all your kitchen tools first.

---

## 🏠 Part 2: Setting Up the Website (Line 13)

```python
st.set_page_config(page_title="Food Classifier & Recommender", page_icon="🍽️", layout="wide")
```

**What this does:**
- Sets the website title (what you see in browser tab)
- Adds a food emoji icon 🍽️
- Makes the layout wide (uses full screen)

**Like**: Putting up a sign on your restaurant that says "Joe's Food Place 🍽️"

---

## 📊 Part 3: Function to Load Food Data (Lines 15-21)

```python
@st.cache_data
def load_data():
    """Load simplified food dataset"""
    return pd.read_csv('data/foods.csv')
```

**Breaking it down:**
- `@st.cache_data` = "Remember this, don't reload it every time" (like bookmarking)
- `def load_data():` = Creating a function called "load_data" (like creating a recipe)
- `pd.read_csv('data/foods.csv')` = Open the Excel file with all our food information

**Real-world analogy**: Like having a cookbook that you only need to open once, then you remember all the recipes.

---

## 🧮 Part 4: BMI Calculator Functions (Lines 22-38)

```python
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
```

**What's happening:**

### `calculate_bmi` function:
- Takes your weight (kg) and height (cm)
- Converts height to meters: `height / 100`
- Uses BMI formula: `weight ÷ (height in meters)²`
- `round(bmi, 1)` = Round to 1 decimal place (like 23.4 instead of 23.456789)

### `get_bmi_category` function:
- Takes BMI number and decides which health category you're in
- Returns category name + colored emoji
- **Like a traffic light system**: 🔵 Too low, 🟢 Good, 🟡 Caution, 🔴 Stop

**Real-world analogy**: Like a doctor's scale that not only weighs you but also tells you if you're in a healthy range.

---

## 🤖 Part 5: Training the AI Brain (Lines 39-62)

```python
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
```

**Breaking this down step by step:**

### Step 1: Prepare the Data
```python
X = df[['Calories', 'Protein', 'Carbs', 'Fat', 'Fiber']].values
y = df['BMICategory'].values
```
- `X` = The nutrition information (like ingredients list)
- `y` = The health category (like the final dish name)
- **Like**: Collecting all recipe ingredients (X) and what dish they make (y)

### Step 2: Split Data for Training and Testing
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```
- Takes 70% of foods to teach the AI
- Keeps 30% hidden to test if AI learned correctly
- **Like**: Teaching a student with some examples, then testing with new problems

### Step 3: Scale the Numbers
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
```
- Makes all numbers similar size (calories might be 300, protein might be 12)
- **Like**: Converting all measurements to the same unit (all in grams instead of mixing grams and kilograms)

### Step 4: Train the AI
```python
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)
```
- Creates a "K-Nearest Neighbors" AI with 3 neighbors
- **Like**: Teaching someone to recognize dogs by showing them 3 similar dogs each time

### Step 5: Test How Good It Is
```python
y_pred = knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
```
- Tests the AI on foods it hasn't seen before
- Calculates how often it gets the right answer
- **Like**: Giving a final exam to see if the student learned well

---

## 🔍 Part 6: The Food Classification Function (Lines 63-107)

```python
def classify_food(nutrition_values, knn, scaler):
    """Classify food and get similar foods"""
    # Scale input
    nutrition_scaled = scaler.transform([nutrition_values])
    
    # Get neighbors first
    distances, indices = knn.kneighbors(nutrition_scaled, n_neighbors=3)
```

**What this does:**
- Takes your food nutrition input
- Scales it the same way as training data
- Finds the 3 most similar foods the AI learned from
- **Like**: You describe a dish, and the chef finds 3 similar recipes they know

### The Voting Process:
```python
neighbor_votes = {}
for train_idx in indices[0]:
    neighbor_category = y_train_split[train_idx]
    neighbor_votes[neighbor_category] = neighbor_votes.get(neighbor_category, 0) + 1
```

**What's happening:**
- Each of the 3 similar foods "votes" for a health category
- Counts how many votes each category gets
- **Like**: Asking 3 friends "Is this healthy?" and counting their answers

### Tie-Breaking:
```python
if len(tied_categories) > 1:
    # Pick the one with smallest average distance (closest neighbors)
```
- If there's a tie (like 2 say "healthy", 1 says "unhealthy"), pick the closest match
- **Like**: If friends disagree, trust the friend who knows you best

---

## 🎯 Part 7: Food Recommendation Function (Lines 108-165)

```python
def recommend_foods_by_bmi(df, bmi_category, n_recs=6):
    """Recommend foods based on BMI category organized by meal times"""
```

**What this does:**
- Takes your BMI category (like "Overweight")
- Finds foods that are good for people in that category
- Organizes them by meal time (Breakfast, Lunch, Dinner)

### Setting Ideal Nutrition Targets:
```python
targets = {
    "Underweight": [400, 28, 45, 20, 4],  # High calories
    "Normal": [280, 22, 35, 12, 6],       # Balanced
    "Overweight": [180, 25, 20, 6, 8],    # Low calories, high protein
    "Obese": [120, 15, 15, 4, 6]          # Very low calories
}
```
- **Like**: A nutritionist's ideal meal plan for each body type
- Numbers are: [Calories, Protein, Carbs, Fat, Fiber]

### Finding Similar Foods:
```python
knn_recommender = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
knn_recommender.fit(X_scaled)
distances, indices = knn_recommender.kneighbors(target_scaled)
```
- Finds foods closest to the ideal nutrition for your body type
- **Like**: Finding recipes that match your dietary goals

---

## 🖥️ Part 8: The Website Interface (Lines 166-400)

### Loading and Training:
```python
df = load_data()
knn, scaler, accuracy, n_train, n_test = train_classifier(df)
```
- Loads the food data
- Trains the AI when the website starts
- **Like**: Opening the restaurant and training the chef before customers arrive

### Creating the Website Layout:
```python
st.title("🍽️ Food Classifier & Recommender")
tab1, tab2 = st.tabs(["🔍 Food Classifier", "🎯 Food Recommender"])
```
- Creates the main title
- Makes two tabs (like having two different menus)

### Tab 1 - Food Classifier:
```python
calories = st.number_input("Calories", min_value=0, max_value=1000, value=200, step=10)
protein = st.number_input("Protein (g)", min_value=0, max_value=100, value=12, step=1)
```
- Creates input boxes where you type nutrition information
- **Like**: Order form where you describe what you want

### The Classification Button:
```python
if st.button("🔍 Classify Food", type="primary"):
    nutrition_values = [calories, protein, carbs, fat, fiber]
    predicted_category, probabilities, classes, distances, indices = classify_food(
        nutrition_values, knn, scaler
    )
```
- When you click the button, it runs the AI classification
- **Like**: Pressing "Submit Order" and waiting for the chef's recommendation

### Displaying Results:
```python
st.markdown(f"""
<div style="background-color: {color}; padding: 15px; border-radius: 10px; text-align: center; color: white; font-size: 18px;">
    <strong>{emoji} Predicted: {predicted_category}</strong>
</div>
""", unsafe_allow_html=True)
```
- Shows the result in a colored box
- **Like**: Getting your order receipt with a colored status (green = good, red = warning)

### Tab 2 - Food Recommender:
```python
age = st.number_input("Age (years)", min_value=10, max_value=100, value=25)
weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
```
- Creates input boxes for your personal information
- **Like**: Filling out a health questionnaire

### BMI Calculation and Display:
```python
bmi = calculate_bmi(weight, height)
category, emoji = get_bmi_category(bmi)
```
- Calculates your BMI and determines your health category
- **Like**: A digital scale that tells you your health status

### Food Recommendations:
```python
recommendations = recommend_foods_by_bmi(df, category)
```
- Gets personalized food suggestions based on your BMI
- **Like**: A nutritionist creating a custom meal plan just for you

---

## 🎯 Summary: How Everything Works Together

1. **You start the app** → Website loads, AI trains itself
2. **You enter food nutrition** → AI finds 3 similar foods, they vote on health category
3. **OR you enter personal details** → App calculates BMI, finds ideal foods for your body type
4. **You get results** → Colored displays, explanations, and personalized recommendations

**The whole thing is like having a smart nutritionist friend who:**
- Knows about 54 Ethiopian foods
- Can instantly tell you if a food is healthy for your body type
- Gives you meal suggestions based on your health goals
- Explains everything in simple terms with pretty colors and emojis

**No internet needed** - everything runs on your computer like a desktop app, but looks like a website!