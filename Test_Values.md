# 🧪 Food Classification Test Values

## 🔵 **UNDERWEIGHT** (High Calorie)
```
Calories: 450
Protein: 35
Carbs: 15
Fat: 28
Fiber: 2
```

## 🟢 **NORMAL** (Balanced)
```
Calories: 280
Protein: 15
Carbs: 35
Fat: 8
Fiber: 6
```

## 🟡 **OVERWEIGHT** (Low Calorie, High Protein)
```
Calories: 200
Protein: 35
Carbs: 0
Fat: 6
Fiber: 0
```

## 🔴 **OBESE** (Very Low Calorie)
```
Calories: 80
Protein: 3
Carbs: 15
Fat: 1
Fiber: 5
```

---

## 📋 **How to Test:**
1. Copy nutrition values from any category above
2. Enter them in your Food Classifier app
3. Click "Classify Food" 
4. Check if prediction matches expected category

## 🎯 **Additional Test Cases:**

### More Underweight Examples:
- `350, 30, 2, 25, 0` (High fat protein)
- `480, 22, 55, 18, 3` (High carb meal)

### More Normal Examples:
- `250, 35, 2, 10, 0` (Lean protein)
- `320, 10, 45, 12, 4` (Balanced breakfast)

### More Overweight Examples:
- `180, 30, 0, 4, 0` (Very lean protein)
- `230, 13, 32, 6, 8` (High fiber legume)

### More Obese Examples:
- `100, 4, 15, 2, 4` (Light soup)
- `120, 4, 12, 7, 4` (Vegetable dish)