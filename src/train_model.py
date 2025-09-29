from preprocessing import load_and_preprocess
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

df = load_and_preprocess("/home/cat/projects/Moscow_housing/data/raw/data.csv")

# Признаки и целевая переменная 
X = df.drop(columns=['Price'])
y = df['Price']

cat_features = ['Apartment type', 'Metro station', 'Region', 'Renovation']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение CatBoost 
cat = CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6, verbose=100, random_state=42)
cat.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_test, y_test))

# Оценка модели 
y_pred = cat.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R²:", r2_score(y_test, y_pred))


cat.save_model("catboost_moscow_housing.cbm")

