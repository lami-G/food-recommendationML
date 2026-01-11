# 🇪🇹 Ethiopian Food Recommendation System

A school machine learning project that calculates BMI and recommends personalized diet plans using Ethiopian and international foods.

## 📖 What This Project Does

1. **Enter your details** - Age, weight, height, gender
2. **Calculate BMI** - Body Mass Index
3. **Get personalized diet** - Breakfast, Lunch, Dinner recommendations
4. **BMI-specific foods** - Different foods for Underweight, Normal, Overweight, Obese

## 🧠 Machine Learning Features

- **Weighted K-Nearest Neighbors (KNN)** algorithm
- Different feature weights for each BMI category
- Underweight → Prioritizes HIGH calorie foods
- Obese → Prioritizes LOW calorie, HIGH fiber foods

## 🚀 How to Run

### Step 1: Install Python
Download from https://www.python.org/downloads/

### Step 2: Install Libraries
```bash
cd EthiopianFoodRecommender
python -m pip install pandas scikit-learn streamlit numpy
```

### Step 3: Run the Web App
```bash
python -m streamlit run web_app.py
```

Opens at **http://localhost:8501**

## 📁 Project Structure

```
EthiopianFoodRecommender/
├── web_app.py              # Main web application
├── data/
│   └── ethiopian_foods.csv # Food dataset (70+ foods)
├── requirements.txt        # Python libraries
└── README.md              # This file
```

## 📊 Dataset

**70+ foods** including:
- 🇪🇹 Ethiopian: Doro Wot, Kitfo, Shiro, Injera, Gomen, Misir Wot
- 🌍 International: Oatmeal, Grilled Chicken, Salads, Steamed Vegetables

## 🎓 Technologies Used

- Python
- Pandas (data handling)
- Scikit-learn (KNN algorithm)
- Streamlit (web interface)
- NumPy (calculations)

## 👨‍🎓 School Project

Made for learning Machine Learning concepts!
