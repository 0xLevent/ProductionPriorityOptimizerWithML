import pandas as pd
import numpy as np
import pyodbc
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sqlalchemy.types import Integer, Float, String, DateTime, Numeric
import datetime

# SQL Server bağlantısı
engine = create_engine('DB_URL')

print("Veritabanı bağlantısını kontrol")
with engine.connect() as connection:
    result = connection.execute(text("SELECT 1"))
    print(result.fetchone())
print("Veritabanı bağlantısı başarılı.")

query = "SELECT * FROM Siparisler2"
df = pd.read_sql(query, engine)

df['PlanlananBitisTarihi'] = pd.to_datetime(df['PlanlananBitisTarihi'])
df['TahminiBitisTarihi'] = pd.to_datetime(df['TahminiBitisTarihi'])
df['UretimBaslamaTarihi'] = pd.to_datetime(df['UretimBaslamaTarihi'])
df['Kalan_Gun'] = (df['PlanlananBitisTarihi'] - datetime.datetime.now()).dt.days
df['Uretim_Yogunlugu'] = df['SiparisMiktari'] / df['CalisanMakineSayisi'].replace(0, 1)
df['Birim_Uretim_Suresi'] = df['TahminiUretimSuresi'] / df['SiparisMiktari'].replace(0, 1)

def remove_outliers(df, column, factor=3):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

columns_to_clean = ['Birim_Uretim_Suresi', 'Uretim_Yogunlugu', 'FireOrani']
for col in columns_to_clean:
    df_temp = remove_outliers(df, col)
    if df_temp.shape[0] > 0:
        df = df_temp
    else:
        print(f"Warning: Removing outliers from {col} would result in an empty dataset. Skipping this column.")

# Özellik ve hedef değişkenler
numeric_features = ['IgneSayisi', 'SiparisMiktari', 'UretimMiktari', 'FireMiktari', 
                    'FireOrani', 'CalisanMakineSayisi', 'Kalan_Gun', 'Uretim_Yogunlugu', 'Birim_Uretim_Suresi']
target = 'TahminiUretimSuresi'

X = df[numeric_features]
y = df[target]

print("\nShape of X:", X.shape)
print("Shape of y:", y.shape)

# Veri bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nShape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)

# Önişleme pipeline'ı
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features)
    ])

models = {
    'Linear Regression': Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ]),
    'Random Forest': Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ]),
    'Gradient Boosting': Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(random_state=42))
    ])
}

best_model = None
best_score = float('-inf')

for name, model in models.items():
    print(f"\nEvaluating {name}...")
    
    if name == 'Linear Regression':
        param_grid = {}
    elif name == 'Random Forest':
        param_grid = {
            'regressor__n_estimators': [100, 200],
            'regressor__max_depth': [None, 10, 20],
            'regressor__min_samples_split': [2, 5]
        }
    else:  # Gradient Boosting
        param_grid = {
            'regressor__n_estimators': [100, 200],
            'regressor__learning_rate': [0.01, 0.1],
            'regressor__max_depth': [3, 4]
        }
    
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Çapraz doğrulama skorunu hesapla
    cv_score = cross_val_score(grid_search.best_estimator_, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    mean_cv_score = -np.mean(cv_score)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Cross-validation MSE: {mean_cv_score}")
    
    if mean_cv_score > best_score:
        best_score = mean_cv_score
        best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nTest MSE: {mse}, R2: {r2}")

df['Predicted_Production_Time'] = best_model.predict(X) 

def custom_sort(row):
    kalan_gun = max(row['Kalan_Gun'], 0)  # Negatif değerleri önle
    urgency = 1 / (kalan_gun + 1)  # Kalan gün azaldıkça aciliyet artar
    complexity = row['IgneSayisi'] * max(row['Birim_Uretim_Suresi'], 0.001)  # Çok küçük veya negatif değerleri önle
    production_load = row['SiparisMiktari'] / max(row['CalisanMakineSayisi'], 1)  # Sıfıra bölmeyi önle
    return urgency * complexity * production_load

df['Sort_Score'] = df.apply(custom_sort, axis=1)
df_sorted = df.sort_values('Sort_Score', ascending=False)

df_sorted['Uretim_Sirasi'] = range(1, len(df_sorted) + 1)


try:
    with engine.connect() as connection:
        with connection.begin():
            # Önce mevcut verileri sil
            delete_stmt = text("DELETE FROM predictions2")
            connection.execute(delete_stmt)
            print("Existing data deleted successfully")

            # Yeni verileri ekle
            for index, row in df_sorted.iterrows():
                print(f"Trying to insert row {index}")
                insert_stmt = text(
                    "INSERT INTO predictions2 (SiparisNumarasi, Predicted_Production_Time, Sort_Score, Uretim_Sirasi) VALUES (:SiparisNumarasi, :Predicted_Production_Time, :Sort_Score, :Uretim_Sirasi)"
                )
                connection.execute(insert_stmt, 
                                   {'SiparisNumarasi': row['SiparisNumarasi'],
                                    'Predicted_Production_Time': row['Predicted_Production_Time'],
                                    'Sort_Score': row['Sort_Score'],
                                    'Uretim_Sirasi': row['Uretim_Sirasi']})
                print(f"Row {index} inserted successfully")
    print("All data inserted successfully")
except Exception as e:
    print(f"An error occurred: {e}")

engine.dispose()

print("\nİşlem tamamlandı.")
