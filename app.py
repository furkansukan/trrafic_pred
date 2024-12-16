import joblib
import folium
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
pd.set_option('display.max_columns', None)

# Verileri yükle
df_ = pd.read_csv(
    "traffic_density__202407.csv")
df = df_.copy()
df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"], format="%Y-%m-%d %H:%M:%S", errors="coerce")

dfpark_ = pd.read_excel(
    "istanbul-park-ve-yeil-alan-koordinatlar.xlsx")
dfpark = dfpark_.copy()

# Sidebar ile sayfa seçimi
st.sidebar.title("NETAŞ")
image = Image.open('logo2.png')  # 'logo.png' dosyasının doğru yerde olduğundan emin olun
with st.sidebar:  # 'with' bloğu düzeltildi
    st.image(image, caption='NETAŞ', use_column_width=True)
    st.write("Yeteneklerimizi Sizin için Kullanıyoruz!")
page = st.sidebar.radio("Gitmek istediğiniz sayfayı seçin:", ["Data Table", "Acil Durum Alanları", "İletişim"])

# Sayfa 1: Data Tablosu
if page == "Data Table":
    st.title("Akıllı Acil Durum Yönetimi")
    st.write("Trafik Yoğunluğu Verisi")
    st.dataframe(df)

    st.write("Park ve Yeşil Alan Verisi")
    st.dataframe(dfpark)


    def time_series_feature_engineering(df):
        # DATE_TIME kolonunu datetime formatına çevir
        df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])

        # 1. Zaman Bileşenleri
        df['year'] = df['DATE_TIME'].dt.year
        df['month'] = df['DATE_TIME'].dt.month
        df['day'] = df['DATE_TIME'].dt.day
        df['hour'] = df['DATE_TIME'].dt.hour
        df['day_of_week'] = df['DATE_TIME'].dt.dayofweek  # 0 = Pazartesi, 6 = Pazar
        df['weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)  # Hafta sonu bilgisi

        # 2. Dönemsellik Özellikleri
        df['day_of_year'] = df['DATE_TIME'].dt.dayofyear
        df['week_of_year'] = df['DATE_TIME'].dt.isocalendar().week

        # 3. Hareketli İstatistikler
        df['rolling_avg_speed'] = df['AVERAGE_SPEED'].rolling(window=3,
                                                              min_periods=1).mean()  # 3 zaman dilimi hareketli ortalama
        df['rolling_min_speed'] = df['MINIMUM_SPEED'].rolling(window=3, min_periods=1).min()
        df['rolling_max_speed'] = df['MAXIMUM_SPEED'].rolling(window=3, min_periods=1).max()
        df['rolling_vehicle_count'] = df['NUMBER_OF_VEHICLES'].rolling(window=3,
                                                                       min_periods=1).sum()  # Araç sayısının toplamı

        # 4. Lag (Gecikme) Özellikleri
        df['lag_1_avg_speed'] = df['AVERAGE_SPEED'].shift(1)  # Bir önceki zaman dilimindeki ortalama hız
        df['lag_1_vehicle_count'] = df['NUMBER_OF_VEHICLES'].shift(1)  # Bir önceki araç sayısı

        # 5. Hız Trendleri
        df['speed_diff'] = df['AVERAGE_SPEED'] - df['lag_1_avg_speed']  # Hız değişimi
        df['vehicle_diff'] = df['NUMBER_OF_VEHICLES'] - df['lag_1_vehicle_count']  # Araç sayısı değişimi

        # Boş değerler varsa doldur (örneğin gecikme kolonlarından kaynaklı)
        df.fillna(0, inplace=True)

        return df


    df = time_series_feature_engineering(df)


    def preprocess_data(df):
        """
        Preprocessing adımlarını içeren fonksiyon.
        Args:
        df: DataFrame
        Returns:
        df_preprocessed: Preprocessed DataFrame
        """
        # 1. String olan kolonları sayısallaştır
        # GEOHASH gibi kolonları sayısallaştırmak için Label Encoding uyguladık
        label_enc = LabelEncoder()
        df['GEOHASH'] = label_enc.fit_transform(df['GEOHASH'].astype(str))

        # 3. Özellikleri normalize veya standardize et
        # Normalizasyon yerine StandardScaler tercih ettik
        feature_columns = df.drop(["LATITUDE", "LONGITUDE", "DATE_TIME", "NUMBER_OF_VEHICLES"], axis=1).columns
        scaler = StandardScaler()

        df[feature_columns] = scaler.fit_transform(df[feature_columns])

        return df


    # Data preprocessing adımını çağır
    df = preprocess_data(df)
    df = df.head(1999)
    df_ = df_.head(1999)
    st.session_state.df = df
    st.session_state.df_ = df_

# Sayfa 2: Acil Durum Alanları
elif page == "Acil Durum Alanları":
    st.title("Acil Durum Toplanma Alanları")

    df = st.session_state.df
    df_ = st.session_state.df_


    def train_and_evaluate_model(df, target_column):
        """
        Eğitim ve test verilerini bölüp RandomForestRegressor modeli eğitir ve değerlendirir.
        Args:
        df: DataFrame, verisetini temsil eder.
        feature_columns: List, modelde kullanılacak özellik kolonları.
        target_column: String, hedef kolon adı (NUMBER_OF_VEHICLES).
        Returns:
        model: Eğitilmiş RandomForestRegressor nesnesi.
        y_test: Test verisi hedef değerleri.
        y_pred: Test verisi tahminleri.
        """
        feature_columns = df.drop(["LATITUDE", "LONGITUDE", "DATE_TIME", "NUMBER_OF_VEHICLES"], axis=1).columns
        # Özellikleri ve hedefi oluştur
        X = df[feature_columns]
        y = df[target_column]

        # Eğitim ve test verilerini böl
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # RandomForestRegressor modelini oluştur ve eğit
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)

        # Test verisi üzerinde tahminleme yap
        y_pred = model.predict(X_test)

        # Model performansını değerlendir
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("\nModel Performansı:")
        print("Mean Absolute Error:", mae)
        print("Mean Squared Error:", mse)
        print("R² Score:", r2)
        st.write("\n**Model Performansı:**")
        st.write(f"✅ Mean Absolute Error (MAE): {mae}")
        st.write(f"✅ Mean Squared Error (MSE): {mse}")
        st.write(f"✅ R² Score: {r2}")

        # Grafikle sonuçları görselleştir
        plt.figure(figsize=(10, 5))
        plt.plot(y_test.values, label='True Vehicle Counts')
        plt.plot(y_pred, label='Predicted Vehicle Counts', alpha=0.7)
        plt.title("True vs Predicted Vehicle Counts")
        plt.legend(loc="upper left")
        plt.show()

        return model, X_test, y_test, y_pred


    # Modeli eğit ve değerlendir

    target_column = 'NUMBER_OF_VEHICLES'

    # Fonksiyonu çağır
    model, X_test, y_test, y_pred = train_and_evaluate_model(df, target_column)


    def predict_and_find_dense_areas(model, df, df_):
        # Model tahminlemesini yap ve yeni sütunlara ekle
        df_['predicted_vehicle_count'] = model.predict(
            df.drop(["LATITUDE", "LONGITUDE", "DATE_TIME", "NUMBER_OF_VEHICLES"], axis=1))

        # Tahmin edilen verileri tablo şeklinde göster
        prediction_table = df_[['DATE_TIME', 'GEOHASH', 'LATITUDE', 'LONGITUDE', 'predicted_vehicle_count']]
        print("Tahminlenen Araç Yoğunlukları:")
        print(prediction_table.head())

        # Yoğun bölgeleri tespit et
        # Burada araç sayısı ortalamanın üzerinde ve hız ortalamasının altında olan bölgeleri buluyoruz
        # Model tahminlemesi yap

        # Her geohash için ortalama araç sayısını hesapla
        geohash_avg_vehicle_count = df_.groupby('GEOHASH')['NUMBER_OF_VEHICLES'].mean().to_dict()

        # Tahmin edilen verileri kontrol et ve yoğun bölgeleri ayır
        dense_areas = []
        for idx, row in df_.iterrows():
            geohash = row['GEOHASH']
            predicted_count = row['predicted_vehicle_count']
            avg_count = geohash_avg_vehicle_count.get(geohash, 0)

            # Eğer tahmin edilen araç sayısı, ortalama araç sayısından büyükse yoğun olabilir
            if predicted_count > avg_count:
                dense_areas.append(row)

        # Yoğun bölgeleri yeni DataFrame'e al
        dense_df = pd.DataFrame(dense_areas)

        # Sonuçları kontrolle
        print("Yoğun bölgelerin DataFrame'i:")
        print(dense_df[['DATE_TIME', 'GEOHASH', 'LATITUDE', 'LONGITUDE', 'predicted_vehicle_count']])

        return prediction_table, dense_df, df_


    prediction_table, dense_df, df_ = predict_and_find_dense_areas(model, df, df_)
    st.dataframe(prediction_table)

    # NaN değerleri kaldırma
    dense_df.dropna(subset=['LATITUDE', 'LONGITUDE'], inplace=True)
    dfpark.dropna(subset=['LATITUDE', 'LONGITUDE'], inplace=True)
    dfpark.columns = dfpark.columns.str.strip()


    st.title("Acil durum esnasında en yakın park ve yeşil alan")
    matched_df = pd.read_csv("matched_data.csv")
    st.dataframe(matched_df)

    # Streamlit Başlığı
    st.title('Park ve Dense Eşleşmeleri Haritası')
    # Rastgele bir eşleşme seç
    if not matched_df.empty:
        sampled_matched = matched_df.sample(n=1)

        # Sütunları sayıya çevir ve NaN değerleri temizle
        columns_to_convert = ['LATITUDE_DENSE', 'LONGITUDE_DENSE', 'LATITUDE_PARK', 'LONGITUDE_PARK']
        for col in columns_to_convert:
            sampled_matched[col] = pd.to_numeric(sampled_matched[col], errors='coerce')

        sampled_matched = sampled_matched.dropna(subset=columns_to_convert)

        # Eğer geçerli satır kalmadıysa çıkış yap
        if sampled_matched.empty:
            st.write("Geçerli veri bulunamadı.")
        else:
            row = sampled_matched.iloc[0]
            dense_lat, dense_lon = row['LATITUDE_DENSE'], row['LONGITUDE_DENSE']
            park_lat, park_lon = row['LATITUDE_PARK'], row['LONGITUDE_PARK']

            # Folium haritası oluştur
            m = folium.Map(location=[(dense_lat + park_lat) / 2, (dense_lon + park_lon) / 2], zoom_start=15)

            # Yol noktalarını ekle
            folium.Marker([dense_lat, dense_lon], popup='Dense (Road)', icon=folium.Icon(color='red')).add_to(m)
            folium.Marker([park_lat, park_lon], popup='Park', icon=folium.Icon(color='green')).add_to(m)

            # Park ve Dense noktalarını bağlayan çizgi ekle
            folium.PolyLine([(dense_lat, dense_lon), (park_lat, park_lon)], color='blue', weight=2).add_to(m)

            # Mesafe hesaplama
            distance = geodesic((dense_lat, dense_lon), (park_lat, park_lon)).km

            st.write(f"Park ve Dense noktası arasındaki mesafe: {distance:.2f} km")

            # Haritayı kaydet ve göster
            m.save('map.html')
            st.components.v1.html(open('map.html', 'r').read(), height=600)

    else:
        st.write("Veri bulunamadı.")
# Sayfa 3: İletişim
elif page == "İletişim":
    st.title("İletişim")
    st.write("""
                    **Daha Fazla Soru ve İletişim İçin**  
                    Bu projeyle ilgili herhangi bir sorunuz veya geri bildiriminiz olursa benimle iletişime geçmekten çekinmeyin! Aşağıdaki platformlar üzerinden ulaşabilirsiniz:
                    """)

    st.write("📧 **E-posta**: furkansukan10@gmail.com")
    st.write("🪪 **LinkedIn**: https://www.linkedin.com/in/furkansukan/")
    st.write("🔗 **Kaggle**: https://www.kaggle.com/furkansukan")
    st.write("🐙 **GitHub**: https://github.com/furkansukan")  # Buraya bağlantı ekleyebilirsiniz
    # Buraya bağlantı ekleyebilirsiniz

    st.write("""
                    Görüş ve önerilerinizi duymaktan mutluluk duyarım!
                    """)
