import pandas as pd
import numpy as np
import pyodbc
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# SQL Server bağlantısı
conn_str = 'DB URL'
conn = pyodbc.connect(conn_str)

query = "SELECT * FROM tableML"
df = pd.read_sql(query, conn)

print("Initial dataframe shape:", df.shape)
print("\nFirst few rows of the dataframe:")
print(df.head())
print("\nColumn data types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())

# Veri Ön İşleme
df['Planlanan Bitiş Tarihi'] = pd.to_datetime(df['Planlanan Bitiş Tarihi'])
df['Tahmini Bitiş Tarihi'] = pd.to_datetime(df['Tahmini Bitiş Tarihi'])
df['Üretim Başlama Tarihi'] = pd.to_datetime(df['Üretim Başlama Tarihi'])

print("\nShape after datetime conversion:", df.shape)

df['Kalan_Gun'] = (df['Planlanan Bitiş Tarihi'] - pd.Timestamp.now()).dt.days
df['Uretim_Yogunlugu'] = df['Sipariş Miktarı'] / df['Çalışan Makine Sayısı'].replace(0, 1)
df['Birim_Uretim_Suresi'] = df['Tahmini Üretim Süresi (saniye)'] / df['Sipariş Miktarı'].replace(0, 1)

print("Shape after adding new features:", df.shape)

print("\nColumns with infinite values:")
print(df.isin([np.inf, -np.inf]).sum())

print("\nMax values of numeric columns:")
print(df.select_dtypes(include=[np.number]).max())

print("\nMin values of numeric columns:")
print(df.select_dtypes(include=[np.number]).min())

# Sonsuz değerleri NaN ile değiştir
df = df.replace([np.inf, -np.inf], np.nan)

df = df.fillna(df.median())

def remove_outliers(df, column, factor=3):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

columns_to_clean = ['Birim_Uretim_Suresi', 'Uretim_Yogunlugu', 'Fire Oranı (%)']
for col in columns_to_clean:
    df_temp = remove_outliers(df, col)
    print(f"Shape after removing outliers from {col}:", df_temp.shape)
    if df_temp.shape[0] > 0:
        df = df_temp
    else:
        print(f"Warning: Removing outliers from {col} would result in an empty dataset. Skipping this column.")

print("Final shape after outlier removal:", df.shape)

numeric_features = ['İğne Sayısı', 'Sipariş Miktarı', 'Üretim Miktarı', 'Fire Miktarı', 
                    'Fire Oranı (%)', 'Çalışan Makine Sayısı', 'Kalan_Gun', 'Uretim_Yogunlugu', 'Birim_Uretim_Suresi']
categorical_features = ['Varyant', 'Model']
target = 'Tahmini Üretim Süresi (saniye)'

X = df[numeric_features + categorical_features]
y = df[target]

print("\nShape of X:", X.shape)
print("Shape of y:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nShape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
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
    else:  
        param_grid = {
            'regressor__n_estimators': [100, 200],
            'regressor__learning_rate': [0.01, 0.1],
            'regressor__max_depth': [3, 4]
        }
    
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
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

df['Predicted_Production_Time'] = best_model.predict(X) / 60

def custom_sort(row):
    urgency = 1 / (row['Kalan_Gun'] + 1)  
    complexity = row['İğne Sayısı'] * row['Birim_Uretim_Suresi']
    production_load = row['Sipariş Miktarı'] / row['Çalışan Makine Sayısı']
    return urgency * complexity * production_load

df['Sort_Score'] = df.apply(custom_sort, axis=1)
df_sorted = df.sort_values('Sort_Score', ascending=False)

df_sorted['Üretim Sırası'] = range(1, len(df_sorted) + 1)

plt.figure(figsize=(12, 6))
sns.scatterplot(x='Predicted_Production_Time', y='Tahmini Üretim Süresi (dakika)', data=df)
plt.title('Tahmin vs Gerçek Üretim Süresi')
plt.xlabel('Tahmin Edilen Üretim Süresi')
plt.ylabel('Gerçek Üretim Süresi')
plt.show()

df_sorted[['Sipariş Numarası', 'Predicted_Production_Time', 'Sort_Score', 'Üretim Sırası']].to_sql('predictions', conn, if_exists='replace', index=False)

conn.close()

print("\nTahminler hesaplandı, sıralandı ve 'predictions' tablosuna yazıldı.")
