# 🍽️ Food Classifier & Recommender

A single-file Streamlit web application with two ML-powered features:

## 🔍 **Food Classifier**
- **Input**: Enter nutrition values (calories, protein, carbs, fat, fiber)
- **AI Process**: K-Nearest Neighbors finds 3 similar foods from training data
- **Output**: Predicts BMI category with voting breakdown and confidence scores
- **Shows**: Actual neighbors used, vote counting, similarity percentages

## 🎯 **Food Recommender** 
- **Input**: Enter personal details (age, weight, height, gender)
- **Process**: Calculate BMI → Determine health category → Find ideal nutrition
- **Output**: Personalized Ethiopian food recommendations by meal time
- **Organization**: 🌅 Breakfast, ☀️ Lunch, 🌙 Dinner (2 foods each)

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run food_app.py
```

**Access**: Opens at http://localhost:8501

## 🏗️ Architecture

**Type**: Single-file Streamlit web application  
**Backend**: None - everything runs in one Python process  
**Data**: Local CSV file with caching optimization  
**ML Models**: Trained in-memory on app startup

## ⚙️ Features

### Tab 1: Food Classifier
- Nutrition input fields with validation
- **KNN Voting Display**: Shows 3 actual neighbors used for prediction
- **Vote Breakdown**: Category vote counts (e.g., "Overweight: 2/3 votes")
- **Similarity Scores**: Distance-based similarity percentages
- **Confidence Scores**: Probability distribution across all categories
- **Tie Resolution**: Automatic tie-breaking using closest average distance

### Tab 2: Food Recommender
- Personal details input (age, weight, height, gender)
- **BMI Calculator**: Automatic BMI calculation and categorization
- **Smart Recommendations**: KNN finds foods closest to ideal nutrition for your BMI
- **Meal Organization**: 
  - 🌅 **Breakfast**: 2 recommended foods
  - ☀️ **Lunch**: 2 recommended foods  
  - 🌙 **Dinner**: 2 recommended foods
- **Detailed Info**: Nutrition breakdown and health explanations for each food

## 🤖 Machine Learning Details

- **Algorithm**: K-Nearest Neighbors (K=3) for both classification and recommendation
- **Preprocessing**: StandardScaler for feature normalization
- **Training Split**: 70% training (~82 foods) / 30% testing (36~ foods)
- **Distance Metric**: Euclidean distance for similarity matching
- **Features**: [Calories, Protein, Carbs, Fat, Fiber]
- **Tie-Breaking**: Distance-based resolution when vote counts are equal

### How It Works:
1. **Training**: Model learns from 38 Ethiopian foods with known BMI categories
2. **Classification**: Finds 3 most similar foods → counts votes → predicts category
3. **Recommendation**: Defines ideal nutrition per BMI → finds closest matching foods

## 📊 Dataset

**54 Ethiopian Foods** across 4 BMI categories and 5 meal times:

### By BMI Category:
- � **Undewrweight**: 12 high-calorie foods (280-480 cal)
- � **ONormal**: 14 balanced foods (150-400 cal)  
- 🟡 **Overweight**: 12 moderate-calorie foods (150-230 cal)
- 🔴 **Obese**: 16 low-calorie foods (40-150 cal)

### By Meal Time:
- � **Breaekfast**:
- ☀️ **Lunch**: 
- � **Dinner*l*: 
- 🍞 **All Meals**: 

**File**: `data/foods.csv` with columns: Name, Calories, Protein, Carbs, Fat, Fiber, BMICategory, MealTime, Reason

## 📚 Educational Materials

- **`MyLab2_Food_Class.ipynb`**: Interactive Jupyter notebook with step-by-step KNN learning

## 🎯 Key Learning Concepts

1. **K-Nearest Neighbors**: How similarity-based classification works
2. **Feature Scaling**: Why StandardScaler is essential for distance calculations  
3. **Train/Test Split**: How models learn and get evaluated
4. **Voting Systems**: How multiple neighbors contribute to final prediction
5. **Distance Metrics**: Understanding similarity in multi-dimensional space

Perfect for understanding machine learning fundamentals with real-world Ethiopian food data!