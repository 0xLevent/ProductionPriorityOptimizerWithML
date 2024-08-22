import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib
import datetime
import time

engine = create_engine('DB_URL')

def process_new_products():
    with engine.connect() as connection:
        new_products = pd.read_sql("SELECT SiparisNumarasi FROM NewProducts WHERE Processed = 0", connection)

    if new_products.empty:
        print("No new products to process.")
        return False

    with engine.connect() as connection:
        existing_predictions = pd.read_sql("SELECT * FROM predictions2", connection)

    new_products_data = pd.read_sql(f"SELECT * FROM Siparisler2 WHERE SiparisNumarasi IN {tuple(new_products['SiparisNumarasi'])}", engine)

    new_products_data['PlanlananBitisTarihi'] = pd.to_datetime(new_products_data['PlanlananBitisTarihi'])
    new_products_data['TahminiBitisTarihi'] = pd.to_datetime(new_products_data['TahminiBitisTarihi'])
    new_products_data['UretimBaslamaTarihi'] = pd.to_datetime(new_products_data['UretimBaslamaTarihi'])
    new_products_data['Kalan_Gun'] = (new_products_data['PlanlananBitisTarihi'] - pd.Timestamp.now()).dt.days
    new_products_data['Uretim_Yogunlugu'] = new_products_data['SiparisMiktari'] / new_products_data['CalisanMakineSayisi'].replace(0, 1)
    new_products_data['Birim_Uretim_Suresi'] = new_products_data['TahminiUretimSuresi'] / new_products_data['SiparisMiktari'].replace(0, 1)

    numeric_features = ['IgneSayisi', 'SiparisMiktari', 'UretimMiktari', 'FireMiktari', 
                        'FireOrani', 'CalisanMakineSayisi', 'Kalan_Gun', 'Uretim_Yogunlugu', 'Birim_Uretim_Suresi']

    model = joblib.load('production_model.joblib')

    X_new = new_products_data[numeric_features]
    new_products_data['Predicted_Production_Time'] = model.predict(X_new)

    def custom_sort(row):
        kalan_gun = max(row['Kalan_Gun'], 0)
        urgency = 1 / (kalan_gun + 1)
        complexity = row['IgneSayisi'] * max(row['Birim_Uretim_Suresi'], 0.001)
        production_load = row['SiparisMiktari'] / max(row['CalisanMakineSayisi'], 1)
        return urgency * complexity * production_load

    new_products_data['Sort_Score'] = new_products_data.apply(custom_sort, axis=1)

    combined_predictions = pd.concat([existing_predictions, new_products_data[['SiparisNumarasi', 'Predicted_Production_Time', 'Sort_Score']]])
    combined_predictions_sorted = combined_predictions.sort_values('Sort_Score', ascending=False)
    combined_predictions_sorted['Uretim_Sirasi'] = range(1, len(combined_predictions_sorted) + 1)

    try:
        with engine.connect() as connection:
            with connection.begin():
                connection.execute(text("DELETE FROM predictions2"))

                combined_predictions_sorted.to_sql('predictions2', connection, if_exists='append', index=False)

                connection.execute(text("UPDATE NewProducts SET Processed = 1 WHERE Processed = 0"))

        print("Predictions updated successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return True

def get_user_input():
    print("Yeni ürün ekleme formu:")
    siparis_numarasi = int(input("Sipariş Numarası: "))
    igne_sayisi = int(input("İğne Sayısı: "))
    siparis_miktari = int(input("Sipariş Miktarı: "))
    calisan_makine_sayisi = int(input("Çalışan Makine Sayısı: "))
    planlanan_bitis_gun = int(input("Planlanan Bitiş Tarihi (kaç gün sonra): "))
    tahmini_bitis_gun = int(input("Tahmini Bitiş Tarihi (kaç gün sonra): "))
    tahmini_uretim_suresi = float(input("Tahmini Üretim Süresi (saat): "))

    yeni_urun = {
        'SiparisNumarasi': siparis_numarasi,
        'IgneSayisi': igne_sayisi,
        'SiparisMiktari': siparis_miktari,
        'UretimMiktari': 0, 
        'FireMiktari': 0,  
        'FireOrani': 0,  # Yeni sipariş olduğu için 0
        'CalisanMakineSayisi': calisan_makine_sayisi,
        'PlanlananBitisTarihi': datetime.datetime.now() + datetime.timedelta(days=planlanan_bitis_gun),
        'TahminiBitisTarihi': datetime.datetime.now() + datetime.timedelta(days=tahmini_bitis_gun),
        'UretimBaslamaTarihi': datetime.datetime.now(),
        'TahminiUretimSuresi': tahmini_uretim_suresi
    }

    return yeni_urun

def add_new_product(yeni_urun):
    insert_query = text("""
    INSERT INTO Siparisler2 (
        SiparisNumarasi, IgneSayisi, SiparisMiktari, UretimMiktari, FireMiktari, 
        FireOrani, CalisanMakineSayisi, PlanlananBitisTarihi, TahminiBitisTarihi, 
        UretimBaslamaTarihi, TahminiUretimSuresi
    ) VALUES (
        :SiparisNumarasi, :IgneSayisi, :SiparisMiktari, :UretimMiktari, :FireMiktari, 
        :FireOrani, :CalisanMakineSayisi, :PlanlananBitisTarihi, :TahminiBitisTarihi, 
        :UretimBaslamaTarihi, :TahminiUretimSuresi
    )
    """)

    try:
        with engine.connect() as connection:
            result = connection.execute(insert_query, yeni_urun)
            connection.commit()
        print("Yeni ürün başarıyla eklendi.")
    except Exception as e:
        print(f"Hata oluştu: {e}")

def check_trigger():
    check_query = text("SELECT COUNT(*) FROM NewProducts WHERE Processed = 0")
    try:
        with engine.connect() as connection:
            result = connection.execute(check_query)
            count = result.scalar()
        print(f"NewProducts tablosunda {count} adet işlenmemiş yeni ürün bulundu.")
        if count > 0:
            print("Triger başarıyla çalıştı!")
        else:
            print("Triger çalışmamış olabilir. Lütfen kontrol edin.")
    except Exception as e:
        print(f"Kontrol sırasında hata oluştu: {e}")

if __name__ == "__main__":
    while True:
        print("\n1. Yeni ürün ekle")
        print("2. Yeni ürünleri işle")
        print("3. Çıkış")
        choice = input("Seçiminiz (1/2/3): ")

        if choice == '1':
            yeni_urun = get_user_input()
            add_new_product(yeni_urun)
            check_trigger()
        elif choice == '2':
            if process_new_products():
                print("Yeni ürünler işlendi.")
            else:
                print("İşlenecek yeni ürün bulunamadı.")
        elif choice == '3':
            print("Programdan çıkılıyor...")
            break
        else:
            print("Geçersiz seçim. Lütfen tekrar deneyin.")

    engine.dispose()
