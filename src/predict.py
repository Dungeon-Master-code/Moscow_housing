import pandas as pd
from catboost import CatBoostRegressor

def predict_new_apartment(new_apartment, model_path="catboost_moscow_housing.cbm"):
    # Загружаем модель
    cat = CatBoostRegressor()
    cat.load_model(model_path)
    
    # Преобразуем словарь в DataFrame
    new_df = pd.DataFrame([new_apartment])

    # Feature Engineering
    new_df['Price_per_m2'] = new_df['Price'] / new_df['Area']
    new_df['Relative_floor'] = new_df['Floor'] / new_df['Number of floors']
    new_df['Living_percent'] = new_df['Living area'] / new_df['Area']
    new_df['Kitchen_percent'] = new_df['Kitchen area'] / new_df['Area']
    new_df['Near_metro'] = (new_df['Minutes to metro'] <= 10).astype(int)

    # Предсказание
    predicted_price = cat.predict(new_df)
    return predicted_price[0]

# Пример использования
if name == "main":
    new_apartment = {
        'Apartment type': 'Secondary',
        'Metro station': 'Пушкинская',
        'Minutes to metro': 5,
        'Region': 'Moscow region',
        'Number of rooms': 2,
        'Area': 60,
        'Living area': 40,
        'Kitchen area': 10,
        'Floor': 3,
        'Number of floors': 5,
        'Renovation': 'Cosmetic'
    }

    price = predict_new_apartment(new_apartment)
    print(f"Прогнозируемая цена квартиры: {price:,.0f} руб.")
