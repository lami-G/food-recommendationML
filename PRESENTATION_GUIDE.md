# 🇪🇹 Ethiopian Food Recommendation System - Complete Explanation Guide

## 📌 WHAT IS THIS PROJECT?

This is a **Machine Learning project** that recommends Ethiopian foods based on a person's **BMI (Body Mass Index)**. 

**Simple explanation:** You enter your weight and height → The system calculates if you're underweight, normal, overweight, or obese → Then it recommends the best foods for YOUR body type.

---

## 🎯 THE PROBLEM WE'RE SOLVING

Different people need different foods:
- **Underweight person** → Needs HIGH calorie foods to gain weight
- **Normal person** → Needs BALANCED foods to maintain weight
- **Overweight person** → Needs LOW calorie, HIGH fiber foods to lose weight
- **Obese person** → Needs LOWEST calorie foods for safe weight loss

**Our ML model learns which foods are best for each category!**

---

## 📊 THE DATA (ethiopian_foods.csv)

We have a dataset with **78 foods** containing:

| Column | What it means | Example |
|--------|---------------|---------|
| Name | Food name | Doro Wot, Kitfo, Shiro |
| Calories | Energy in food | 450, 280, 120 |
| Protein | Muscle-building nutrient (grams) | 35g, 15g |
| Carbs | Energy nutrient (grams) | 50g, 30g |
| Fat | Fat content (grams) | 20g, 8g |
| Fiber | Digestive health (grams) | 6g, 2g |
| BMICategory | Who should eat this | Underweight, Normal, Overweight, Obese |
| MealTime | When to eat | Breakfast, Lunch, Dinner |
| Cuisine | Origin | Ethiopian, International |

---

## 🧠 THE MACHINE LEARNING ALGORITHM: K-Nearest Neighbors (KNN)

### What is KNN?
KNN finds the **most similar items** to what you're looking for.

**Real-life example:** 
- You want to buy a phone similar to iPhone
- KNN looks at all phones and finds the 3 most similar ones (maybe Samsung, Google Pixel, etc.)
- It compares features like price, screen size, camera quality

### How KNN works in our project:
1. We have a "target" (ideal food for your BMI)
2. KNN finds foods that are CLOSEST to this target
3. It recommends the top 3 closest foods


---

## 🔧 CODE EXPLANATION - STEP BY STEP

### STEP 1: Import Libraries
```python
import pandas as pd          # For handling data tables
import numpy as np           # For math calculations
import matplotlib.pyplot as plt  # For creating charts
import seaborn as sns        # For beautiful visualizations
from sklearn.preprocessing import StandardScaler, LabelEncoder  # For preparing data
from sklearn.neighbors import NearestNeighbors  # THE ML ALGORITHM!
```

**Why we need these:**
- `pandas` = Works with data like Excel spreadsheets
- `numpy` = Does math operations fast
- `matplotlib/seaborn` = Creates graphs and charts
- `sklearn` = The Machine Learning library (has KNN algorithm)

---

### STEP 2: Load Data
```python
data = pd.read_csv('data/ethiopian_foods.csv')
```

**What this does:** Reads the CSV file and puts it into a table called `data`

---

### STEP 3: Explore Data
```python
data.head()      # Shows first 5 rows
data.shape       # Shows (rows, columns) → (78, 15)
data.info()      # Shows column types
data.describe()  # Shows statistics (mean, min, max)
```

**Why we do this:** To understand our data before training

---

### STEP 4: Data Cleaning
```python
data.isnull().sum()      # Check for missing values
data.duplicated().sum()  # Check for duplicate rows
```

**Why:** ML models don't work well with missing or duplicate data

---

### STEP 5: Data Transformation (IMPORTANT!)

#### Label Encoding - Converting text to numbers
```python
label = LabelEncoder()
df['Cuisine_Encoded'] = label.fit_transform(df['Cuisine'])
# Ethiopian → 0, International → 1
```

**Why:** Computers only understand numbers, not text!

#### Feature Scaling - Making numbers comparable
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**Why:** Calories are 100-500, but Fiber is 0-10. Scaling makes them comparable.

**Before scaling:** Calories=450, Fiber=6
**After scaling:** Calories=0.8, Fiber=0.5 (both on same scale)


---

## ⭐ THE CORE ML CODE EXPLAINED

### BMI Calculation
```python
def calculate_bmi(weight, height):
    height_m = height / 100          # Convert cm to meters
    bmi = weight / (height_m ** 2)   # BMI formula
    return round(bmi, 1)
```

**Example:** Weight=70kg, Height=170cm
- height_m = 170/100 = 1.7 meters
- bmi = 70 / (1.7 × 1.7) = 70 / 2.89 = **24.2**

### BMI Categories
```python
def get_bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"    # Too thin
    elif 18.5 <= bmi < 25:
        return "Normal"         # Healthy
    elif 25 <= bmi < 30:
        return "Overweight"     # Slightly heavy
    else:
        return "Obese"          # Need to lose weight
```

---

## 🎯 THE SECRET SAUCE: WEIGHTED KNN

### What are "Weights"?
Weights tell the model **what's more important** for each BMI category.

```python
def get_bmi_weights(bmi_category):
    # Weights: [Calories, Protein, Carbs, Fat, Fiber]
    
    if bmi_category == "Underweight":
        return [3.0, 1.5, 2.0, 2.5, 0.5]  # HIGH calories important!
    
    elif bmi_category == "Normal":
        return [1.0, 1.5, 1.0, 1.0, 1.0]  # Everything balanced
    
    elif bmi_category == "Overweight":
        return [2.0, 2.0, 1.0, 1.5, 2.5]  # LOW calories, HIGH fiber
    
    else:  # Obese
        return [3.5, 1.5, 1.0, 2.0, 3.0]  # LOWEST calories, HIGHEST fiber
```

**Understanding the weights:**
- Higher number = More important
- For **Underweight**: Calories weight is 3.0 (very important to get high calories)
- For **Obese**: Fiber weight is 3.0 (very important to feel full with low calories)

---

### The KNN Recommendation Function (THE MAIN ML CODE!)

```python
def weighted_knn_recommend(df, bmi_category, meal_time, n_recommendations=3):
    
    # STEP 1: Filter foods by meal time
    meal_foods = df[df['MealTime'] == meal_time]
    
    # STEP 2: Get nutrition features
    nutrition_cols = ['Calories', 'Protein', 'Carbs', 'Fat', 'Fiber']
    X = meal_foods[nutrition_cols].values
    
    # STEP 3: Scale the data (make numbers comparable)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # STEP 4: Apply weights (THIS IS THE SMART PART!)
    weights = get_bmi_weights(bmi_category)
    X_weighted = X_scaled * weights
    
    # STEP 5: Create KNN model
    model = NearestNeighbors(n_neighbors=3, metric='euclidean')
    model.fit(X_weighted)
    
    # STEP 6: Find closest foods to target
    target = get_target_nutrition(bmi_category)
    distances, indices = model.kneighbors(target)
    
    # STEP 7: Return recommended foods
    return meal_foods.iloc[indices[0]]
```


---

## 📈 HOW WE MEASURE ACCURACY

### 1. Hit Rate
"How many recommended foods match the correct BMI category?"

```python
# If we recommend for Underweight person:
# - Doro Wot (Underweight) ✅ HIT
# - Kitfo (Underweight) ✅ HIT  
# - Salad (Obese) ❌ MISS

Hit Rate = (Hits / Total) × 100 = (2/3) × 100 = 66.7%
```

### 2. Mean Absolute Error (MAE)
"How far are recommended calories from target?"

```python
Target for Underweight = 400 calories
Recommended food = 380 calories
Error = |400 - 380| = 20 calories

Lower MAE = Better recommendations
```

### 3. Training vs Test Accuracy (Overfitting Check)
```python
# Split data: 80% training, 20% testing
X_train, X_test = train_test_split(X, test_size=0.2)

# Train on 80%
model.fit(X_train)

# Test on 20% (data model never saw!)
train_accuracy = model.score(X_train)  # e.g., 85%
test_accuracy = model.score(X_test)    # e.g., 80%

# If both are similar → GOOD FIT ✅
# If train=95%, test=60% → OVERFITTING ⚠️
# If both are low (40%) → UNDERFITTING ⚠️
```

### 4. Cross-Validation
"Test the model multiple times with different data splits"

```python
# 5-Fold Cross Validation:
# Fold 1: Train on 80%, Test on 20% → 82%
# Fold 2: Train on different 80%, Test on different 20% → 78%
# Fold 3: → 85%
# Fold 4: → 80%
# Fold 5: → 83%

Average = 81.6% (This is more reliable than single test!)
```

---

## 🔄 THE COMPLETE FLOW

```
USER INPUT                    PROCESSING                      OUTPUT
───────────                   ──────────                      ──────
Weight: 70kg          →    Calculate BMI: 24.2        →    Category: Normal
Height: 170cm         →    Get weights: [1,1.5,1,1,1] →    Daily calories: 2000
Age: 25               →    Run KNN algorithm          →    
Gender: Male          →    Find closest foods         →    Breakfast: Firfir, Kinche
                                                           Lunch: Tibs, Shiro
                                                           Dinner: Atkilt Wot
```

---

## 🎓 KEY TERMS TO KNOW FOR PRESENTATION

| Term | Simple Explanation |
|------|-------------------|
| **Machine Learning** | Computer learns patterns from data |
| **KNN** | Finds most similar items |
| **BMI** | Body Mass Index - weight/height ratio |
| **Feature** | A column in data (Calories, Protein, etc.) |
| **Training** | Teaching the model with data |
| **Testing** | Checking if model works on new data |
| **Overfitting** | Model memorizes training data, fails on new data |
| **Underfitting** | Model is too simple, doesn't learn patterns |
| **Scaling** | Making all numbers on same scale |
| **Encoding** | Converting text to numbers |
| **Cross-Validation** | Testing model multiple times for reliability |
| **Accuracy** | Percentage of correct predictions |
| **MAE** | Average error in predictions |


---

## 💡 COMMON QUESTIONS & ANSWERS

### Q: Why did you choose KNN algorithm?
**A:** KNN is perfect for recommendation systems because:
1. It finds similar items based on features
2. Easy to understand and explain
3. Works well with small datasets
4. No complex training needed

### Q: What makes your model "smart"?
**A:** The **weighted features**! Different BMI categories prioritize different nutrients:
- Underweight → High weight on Calories (needs more energy)
- Obese → High weight on Fiber (needs to feel full with less calories)

### Q: How do you know the model is accurate?
**A:** We use multiple metrics:
1. Hit Rate (BMI category match)
2. MAE (calorie accuracy)
3. Cross-validation (consistent performance)
4. Train/Test split (no overfitting)

### Q: What is overfitting?
**A:** When model memorizes training data but fails on new data.
- Like a student who memorizes answers but can't solve new problems
- We prevent this by testing on data the model never saw

### Q: Why do you scale the data?
**A:** Because features have different ranges:
- Calories: 50-500
- Fiber: 0-10

Without scaling, Calories would dominate because numbers are bigger. Scaling makes them equal.

### Q: What's the difference between training and testing?
**A:** 
- **Training (80%)**: Model learns patterns from this data
- **Testing (20%)**: We check if model works on NEW data it never saw

---

## 📱 THE WEB APP (web_app.py)

The Streamlit web app has 4 pages:

1. **Home** - Introduction and statistics
2. **BMI & Diet Plan** - Enter your info, get recommendations
3. **Browse Foods** - See all foods with filters
4. **How ML Works** - Explains the algorithm

### To run:
```bash
python -m streamlit run web_app.py
```

---

## 🏆 PROJECT SUMMARY

**What we built:** A smart food recommendation system for Ethiopian cuisine

**The ML approach:** Weighted K-Nearest Neighbors

**Key innovation:** Different weights for different BMI categories

**Technologies:**
- Python (programming language)
- Pandas (data handling)
- Scikit-learn (machine learning)
- Streamlit (web interface)
- Matplotlib/Seaborn (visualizations)

**Results:**
- Model recommends appropriate foods for each BMI category
- Good accuracy with no overfitting
- User-friendly web interface

---

## 🎤 PRESENTATION TIPS

1. **Start with the problem:** "Different people need different foods based on their body type"

2. **Show the solution:** "Our ML model recommends the right Ethiopian foods for YOUR BMI"

3. **Explain simply:** "KNN finds the most similar foods to what your body needs"

4. **Show results:** Run the web app and demonstrate live

5. **Mention accuracy:** "We tested with cross-validation and achieved X% accuracy"

6. **Conclude:** "This helps people make healthier food choices based on their body type"

---

## ✅ CHECKLIST BEFORE PRESENTATION

- [ ] Understand what BMI is and how it's calculated
- [ ] Know the 4 BMI categories
- [ ] Explain KNN in simple terms
- [ ] Understand why we use weights
- [ ] Know what overfitting/underfitting means
- [ ] Be able to explain accuracy metrics
- [ ] Run the web app successfully
- [ ] Practice explaining the code flow

**Good luck with your presentation! 🎉**
