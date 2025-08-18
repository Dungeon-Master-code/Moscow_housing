import pandas as pd
import numpy as np

def load_and_preprocess(path="data/raw/data.csv"):
    df = pd.read_csv(path)
    
    # === Заполнение пропусков ===
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # === Приведение типов ===
    num_cols = ['Price', 'Minutes to metro', 'Number of rooms', 'Area', 'Living area', 
                'Kitchen area', 'Floor', 'Number of floors']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    cat_cols = ['Apartment type', 'Metro station', 'Region', 'Renovation']
    for col in cat_cols:
        df[col] = df[col].astype('category')

    # === Удаление выбросов ===
    df = df[(df['Price'] > 0) & (df['Price'] < 1e8)]
    df = df[(df['Area'] > 0) & (df['Area'] < 500)]

    # === Feature Engineering ===
    df['Price_per_m2'] = df['Price'] / df['Area']
    df['Relative_floor'] = df['Floor'] / df['Number of floors']
    df['Living_percent'] = df['Living area'] / df['Area']
    df['Kitchen_percent'] = df['Kitchen area'] / df['Area']
    df['Near_metro'] = (df['Minutes to metro'] <= 10).astype(int)

    return df
