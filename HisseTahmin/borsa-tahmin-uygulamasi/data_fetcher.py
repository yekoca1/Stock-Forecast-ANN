import yfinance as yf
import matplotlib.pyplot as plt

# Kullanıcıdan hisse kodu alınır
hisse_kodu = input("Tahmin yapılacak hisse kodunu giriniz (örnek: AAPL): ").upper()

# Son 1 yılın verisini çek
veri = yf.download(hisse_kodu, period="1y")

# Veriyi terminalde göster
print(veri)

# Çekilen veriyi CSV dosyasına kaydet
csv_dosya_adi = f"{hisse_kodu}_veri.csv"
veri.to_csv(csv_dosya_adi)
print(f"\nVeriler '{csv_dosya_adi}' dosyasına kaydedildi!")

# Grafik çizimi: Kapanış fiyatına göre
veri['Close'].plot(figsize=(12, 6))
plt.title(f"{hisse_kodu} Hisse Kapanış Fiyatı (Son 1 Yıl)")
plt.xlabel("Tarih")
plt.ylabel("Fiyat (USD)")
plt.grid(True)
plt.show()
