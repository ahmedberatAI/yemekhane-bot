# 🥗 Ankara Yemekhane Botu

Telegram üzerinden **Ankara KYK** ve **Ankara Üniversitesi** yemekhane menülerini gösteren,
aynı zamanda her sabah menüyü otomatik olarak gönderen bir bottur.

- 📅 Günlük menü gönderimi (schedule)
- 🔎 Belirli bir tarihin menüsünü görme
- 🍰 Belirli bir yemeği ay boyunca hangi günlerde çıktığını arama
- ☁️ Render üzerinde 7/24 çalışan deployment

---

## 📌 Özellikler

- **/bugun** → Bugünün menüsünü getirir.
- **/yarin** → Yarın çıkacak menüyü gösterir.
- **/menu** → Kısa yol, bugünün menüsü.
- **/tarih GG/AA/YYYY** → Belirli bir tarihin menüsünü gösterir.
- **/ara <yemek adı>** → İçinde o yemeğin geçtiği tüm günleri listeler (o ay içinde).
- **/start** → Botun tanıtımını ve temel komutları açıklar.
- **/help, /yardim, /komutlar** → Tüm komutların açıklamalı listesi.
- **Otomatik gönderim** → Her sabah 08:00’de günlük menü mesajı.

Veriler CSV dosyalarından okunur ve şu anda:

- `Ankara KYK`
- `Ankara Üniversitesi` (öğle + akşam)

için tanımlanmıştır. Yeni kurumlar eklemek kolayca mümkündür.

---

## 🧱 Kullanılan Teknolojiler

- **Python** (önerilen: 3.11+)
- [python-telegram-bot 21.x](https://docs.python-telegram-bot.org/)
- [pandas](https://pandas.pydata.org/)
- [APScheduler](https://apscheduler.readthedocs.io/)
- [python-dotenv](https://github.com/theskumar/python-dotenv)
- [Flask](https://flask.palletsprojects.com/) (Render health-check için basit keep-alive endpoint’i)

---

## 📁 Proje Yapısı (Örnek)

```text
.
├── yemekTelegram.py          # Botun ana kodu
├── requirements.txt          # Python bağımlılıkları
├── ankara_kyk_yemekleri.csv  # Ankara KYK menüleri
├── kyk_aksam_yemekleri.csv   # (varsa) KYK akşam menüleri
├── ankara_universitesi_ogle_yemekleri.csv
├── ankara_uni_aksam_yemekleri.csv
└── .env                      # Ortam değişkenleri (local’de sen ekleyeceksin)
