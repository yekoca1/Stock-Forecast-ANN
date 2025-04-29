import pandas as pd

hisse_kodu = input("Analiz yapılacak hisse kodunu giriniz (örnek: AAPL): ").upper()

csv_dosya_adi = f"{hisse_kodu}_veri.csv"

# CSV dosyasını oku, ilk satırı (başlık satırı) atla
veri = pd.read_csv(csv_dosya_adi, skiprows=2)

# Sütun isimlerini düzelt
veri.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

# Tarih kolonunu index yap ve tarih olarak ayarla
veri['Date'] = pd.to_datetime(veri['Date'])
veri.set_index('Date', inplace=True)

# Sayı kolonlarını float'a çevir
for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
    veri[col] = pd.to_numeric(veri[col], errors='coerce')

# Eksik verileri göster
print("\nEksik Veriler (Düzeltilmiş):")
print(veri.isnull().sum())

# Eksik verileri sil
veri = veri.dropna()

# Son hali göster
print("\nİlk 5 Satır (Temizlenmiş):")
print(veri.head())

print("\nVeri Bilgisi (Temizlenmiş):")
print(veri.info())

print("\nTemel İstatistikler (Temizlenmiş):")
print(veri.describe())
