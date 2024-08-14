import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data(file_path):
    """Veriyi yükler ve ön işleme yapar."""
    df = pd.read_excel(file_path)
    date_columns = ['Planlanan Bitiş Tarihi', 'Tahmini Bitiş Tarihi', 'Üretim Başlama Tarihi']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Eksik değerleri doldur
    df['Çalışan Makine Sayısı'].fillna(df['Çalışan Makine Sayısı'].mean(), inplace=True)
    df['Çalışan Makinenin Ortalama Üretim Saniyesi'].fillna(df['Çalışan Makinenin Ortalama Üretim Saniyesi'].mean(), inplace=True)
    
    return df

def prioritize_products(data):
    data['Tahmini Üretim Süresi'] = data['Üretim Miktarı'] * data['Çalışan Makinenin Ortalama Üretim Saniyesi'] / data['Çalışan Makine Sayısı']
    
    data['Tahmini Bitiş Tarihi'] = data['Üretim Başlama Tarihi'] + pd.to_timedelta(data['Tahmini Üretim Süresi'], unit='s')
    
    data['Gecikme'] = (data['Tahmini Bitiş Tarihi'] - data['Planlanan Bitiş Tarihi']).dt.total_seconds()
    
    now = pd.Timestamp.now()
    data['Öncelik Puanı'] = data['Gecikme'] + (data['Planlanan Bitiş Tarihi'] - now).dt.total_seconds()
    
    return data.sort_values('Öncelik Puanı')

def calculate_extra_machines(row):
    if pd.isna(row['Tahmini Bitiş Tarihi']) or pd.isna(row['Planlanan Bitiş Tarihi']) or pd.isna(row['Tahmini Üretim Süresi']) or pd.isna(row['Çalışan Makine Sayısı']):
        return 0
    if row['Tahmini Bitiş Tarihi'] > row['Planlanan Bitiş Tarihi']:
        delay = (row['Tahmini Bitiş Tarihi'] - row['Planlanan Bitiş Tarihi']).total_seconds()
        extra_machines = np.ceil(delay / (row['Tahmini Üretim Süresi'] / row['Çalışan Makine Sayısı']))
        return max(0, int(extra_machines))
    return 0

def build_and_evaluate_model(data):
    features = ['Üretim Miktarı', 'Çalışan Makine Sayısı', 'Çalışan Makinenin Ortalama Üretim Saniyesi']
    target = 'Gecikme'
    
    X = data[features]
    y = data[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    return model, mae

def visualize_data(data):
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='Planlanan Bitiş Tarihi', y='Gecikme', hue='Çalışan Makine Sayısı', data=data)
    plt.title('Planlanan Bitiş Tarihi vs Gecikme')
    plt.savefig('gecikme_analizi.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(data['Öncelik Puanı'], bins=30, kde=True)
    plt.title('Öncelik Puanı Dağılımı')
    plt.savefig('oncelik_puani_dagilimi.png')
    plt.close()

def main():
    file_path = 'D:\\data.xlsx'
    data = load_and_preprocess_data(file_path)
    
    prioritized_products = prioritize_products(data)
    
    print("Ürünler öncelik sırasına göre:")
    print(prioritized_products[['Sipariş Numarası', 'Planlanan Bitiş Tarihi', 'Tahmini Bitiş Tarihi', 'Gecikme', 'Öncelik Puanı']])
    
    data['Gereken Ekstra Makine Sayısı'] = data.apply(calculate_extra_machines, axis=1)
    
    print("\nEkstra makine gerektiren ürünler:")
    extra_machines_needed = data[data['Gereken Ekstra Makine Sayısı'] > 0].sort_values('Gereken Ekstra Makine Sayısı', ascending=False)
    print(extra_machines_needed[['Sipariş Numarası', 'Planlanan Bitiş Tarihi', 'Tahmini Bitiş Tarihi', 'Gereken Ekstra Makine Sayısı']])
    
    model, mae = build_and_evaluate_model(data)
    print(f"\nModel Mean Absolute Error: {mae:.2f} saniye")
    
    joblib.dump(model, 'gecikme_tahmin_modeli.joblib')
    print("Model kaydedildi: gecikme_tahmin_modeli.joblib")
    
    visualize_data(data)
    print("Veri görselleştirmeleri kaydedildi: gecikme_analizi.png, oncelik_puani_dagilimi.png")
    
    ornek_urun = data.iloc[0]
    ornek_ozellikler = [[ornek_urun['Üretim Miktarı'], ornek_urun['Çalışan Makine Sayısı'], ornek_urun['Çalışan Makinenin Ortalama Üretim Saniyesi']]]
    tahmin_edilen_gecikme = model.predict(ornek_ozellikler)[0]
    print(f"\nÖrnek ürün (Sipariş Numarası: {ornek_urun['Sipariş Numarası']}) için tahmini gecikme: {tahmin_edilen_gecikme:.2f} saniye")

if __name__ == "__main__":
    main()