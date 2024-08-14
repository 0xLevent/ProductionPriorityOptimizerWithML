import pandas as pd
import pyodbc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

conn_str = '.env'
conn = pyodbc.connect(conn_str)

query = "SELECT * FROM dataFromEx"
df = pd.read_sql(query, conn)

features = ['İğne Sayısı', 'Sipariş Miktarı', 'Üretim Miktarı', 'Fire Miktarı', 'Fire Oranı (%)', 'Çalışan Makine Sayısı']
target = 'Tahmini Üretim Süresi (saniye)'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")

df['Predicted_Production_Time'] = model.predict(X)

# Yeni tablo oluşturma
cursor = conn.cursor()
create_predictions_table_query = '''
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='predictions' AND xtype='U')
CREATE TABLE predictions (
    [Sipariş Numarası] INT PRIMARY KEY,
    [Predicted_Production_Time] FLOAT
)
'''
cursor.execute(create_predictions_table_query)
conn.commit()

# Tahminleri yeni tabloya yazma
insert_prediction_query = '''
INSERT INTO predictions ([Sipariş Numarası], [Predicted_Production_Time])
VALUES (?, ?)
'''

for index, row in df.iterrows():
    try:
        cursor.execute(insert_prediction_query, 
                       int(row['Sipariş Numarası']), 
                       float(row['Predicted_Production_Time']))
    except Exception as e:
        print(f"Hata oluştu (Sipariş Numarası: {row['Sipariş Numarası']}): {e}")

conn.commit()
cursor.close()
conn.close()

print("Tahminler yeni 'predictions' tablosuna yazıldı.")
