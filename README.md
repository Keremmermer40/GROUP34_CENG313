# GROUP34_CENG313 — Jigsaw: Community Rule Violation Prediction

Bu projede bir Reddit yorumunun verilen topluluk kuralını ihlal etme olasılığını tahmin ediyoruz. Hedef değişken `rule_violation`: `1` ihlal, `0` ihlal yok.

- **Ders:** CENG313 — Introduction to Data Science
- **Grup:** 34
- **Yarışma:** [Jigsaw — Agile Community Rules](https://www.kaggle.com/competitions/jigsaw-agile-community-rules)
- **Ders yönergesi:** [ceng313-2026](https://github.com/ceng313-2026)

## Grup üyeleri

| Ad soyad | Öğrenci numarası | GitHub kullanıcı adı |
|---|---|---|
| Hacı Kerem Meremr | 23118080769 | [Keremmermer40](https://github.com/Keremmermer40) |
| Necdet Atakan Aydın | 24118080793 | [necdetatakanaydin](https://github.com/necdetatakanaydin) |

## Veri kaynağı ve indirme

1. Kaggle hesabınızla yarışma sayfasındaki kuralları kabul edin.
2. Yarışmanın **Data** sekmesinden verileri indirin.
3. İndirilen arşivi açıp CSV dosyalarını proje kökündeki `data/` klasörüne yerleştirin:

```text
data/
├── train.csv
├── test.csv
└── sample_submission.csv
```

Veriler ve veri içeren ZIP paketleri GitHub'a yüklenmemelidir. `.gitignore` bu dosyaları dışlar; önceden Git takibine alınmış dosyalar ayrıca takipten çıkarılmalıdır.

Mevcut yerel eğitim dosyası **2.029**, test dosyası **10** satırdır. Bu test dosyası örnek test verisidir. Etiketleri bulunmadığından bu 10 satır üzerinden başarı metriği hesaplanmaz.

| Sütun | Açıklama |
|---|---|
| `row_id` | Satır kimliği |
| `body` | Değerlendirilen Reddit yorumu |
| `rule` | Yorumun değerlendirileceği kural |
| `subreddit` | Yorumun ait olduğu topluluk |
| `positive_example_1`, `positive_example_2` | Kuralı ihlal eden örnekler |
| `negative_example_1`, `negative_example_2` | Kuralı ihlal etmeyen örnekler |
| `rule_violation` | Yalnızca eğitim verisinde bulunan hedef etiket |

## Proje yapısı

Mevcut dosyalar:

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── train_baseline.py          # Yerel eğitim ve değerlendirme
├── assignment1_colab.ipynb    # src/ kodunu çalıştıran Colab yardımcısı
├── jigsaw_colab.ipynb         # Kodun hücrelerde bulunduğu alternatif notebook
├── prepare_colab.py          # Yerel Colab paketi hazırlama
├── src/
│   ├── __init__.py
│   ├── data.py               # Veri okuma, metin hazırlama ve bölme
│   ├── models.py             # Baseline modelleri
│   └── utils.py              # Metrikler ve tablo yazdırma
├── submissions/              # Skor kanıtları
├── data/                     # Yalnızca yerel veri
└── outputs/                  # Üretilen sonuçlar
```

**Teslim notebook'u henüz hazır değildir.** İstenen `assignment1_eda_baseline.ipynb`; problem tanımı, veri özeti, yorumlu EDA grafikleri, ön işleme gerekçeleri, baseline değerlendirmesi, skor kanıtı ve sonuç bölümlerini içermelidir. Ortak kod `src/` üzerinden kullanılmalıdır. Mevcut Colab yardımcıları bu teslimin yerine geçmez. İkinci teslim için `assignment2_solution.ipynb` hazırlanacaktır.

## Yerelde çalıştırma

Verileri `data/` klasörüne koyduktan sonra proje kökünde çalıştırın.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train_baseline.py
```

Linux/macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python train_baseline.py
```

Betik proje klasörünü kendi konumundan belirler. Gerekli paketler `requirements.txt` dosyasında listelenmiştir.

### Google Colab

Mevcut `src/` tabanlı çalıştırma yardımcısını kullanmak için:

1. Verileri yerelde `data/` klasörüne koyun ve `python prepare_colab.py` çalıştırın.
2. Oluşan `assignment1_colab.ipynb` dosyasını Colab'da açın.
3. İlk yükleme hücresinde yerelde oluşan `jigsaw_colab.zip` dosyasını seçin.
4. CPU çalışma zamanında hücreleri sırayla çalıştırın ve sonuç paketini indirin.

Bu ZIP veri içerir; kişisel çalıştırma içindir ve repoya eklenmemelidir. Sonuç paketini indirmek notebook'u kaydetmez. Notebook çıktıları ayrıca kaydedilmelidir.

## Baseline yaklaşımı

`body` ve `rule` sütunlarındaki eksik metinler boş dizeyle doldurulur. Model girdisi `RULE: <kural> COMMENT: <yorum>` biçimindedir. Subreddit ve örnek yorum sütunları mevcut modellerde kullanılmaz.

Üç model aynı veri ayrımlarında karşılaştırılır:

| Model | Ayarlar |
|---|---|
| DummyClassifier | `strategy="prior"`; sınıf oranına dayalı referans |
| Kelime TF-IDF + Logistic Regression | 1–2 kelimelik n-gramlar |
| Karakter TF-IDF + Logistic Regression | `char_wb`, 3–5 karakterlik n-gramlar |

İki TF-IDF modelinde `min_df=2` ve `max_features=30000`; Logistic Regression için `max_iter=1000`, `random_state=42` kullanılır. TF-IDF yalnızca ilgili eğitim bölümünde öğrenilir.

## Yerel doğrulama sonuçları

Yorumlar küçük harfe çevrilip kenar boşlukları temizlenerek gruplandırılır. `GroupShuffleSplit`, aynı normalize edilmiş yorumun eğitim ve doğrulamaya birlikte girmesini önler. Yaklaşık %80 eğitim ve %20 doğrulama ayrımı **7, 21 ve 42** tohumlarıyla tekrarlanır.

Aşağıdaki değerler mevcut `outputs/results.csv` dosyasındaki üç ayrımın ortalamasıdır:

| Model | ROC-AUC | Ortalama kural AUC | Accuracy | F1 |
|---|---:|---:|---:|---:|
| Dummy | 0.5000 | 0.5000 | 0.4894 | 0.6570 |
| Kelime TF-IDF + Logistic Regression | 0.8068 | 0.7948 | **0.7173** | **0.7254** |
| Karakter TF-IDF + Logistic Regression | **0.8140** | **0.8006** | 0.7082 | 0.7137 |

Ortalama kural AUC, kural başına ROC-AUC değerlerinin eşit ağırlıklı ortalamasıdır; tek sınıf içeren grupların AUC'si tanımsız bırakılır. Accuracy ve F1 için eşik `0.5`'tir. Kod ortalama kural AUC'ye göre karakter modelini seçer ve tüm eğitim verisiyle yeniden eğitir.

Bu değerler **yerel doğrulama sonuçlarıdır; resmi Kaggle skoru değildir.** Ayrımlar örtüşebilir ve model seçiminde de kullanılır. Bağımsız test başarısını veya görülmemiş kurallara genellemeyi ölçmez.

### Kaggle skoru ve değerlendirme kanıtı

Bu repoda doğrulanabilir Kaggle skor kanıtı bulunmuyor. Skor alındığında tarih ve ekran görüntüsü `submissions/` altında paylaşılmalı; public/private skorlar burada belirtilmelidir.

Late submission kullanılamıyorsa yönergedeki sabit %20 held-out değerlendirmesi ayrıca tamamlanmalı ve yarışma metriğiyle liderlik tablosuna kıyaslanmalıdır. Mevcut model seçimi bu bağımsız değerlendirme olarak sunulmaz. Bu yol tamamlandığında yönergede istenen “Competition closed — evaluated on a held-out split.” açıklaması eklenmelidir.

## Üretilen dosyalar

| Dosya | İçerik |
|---|---|
| `outputs/results.csv` | Model ve tohum başına doğrulama metrikleri |
| `outputs/per_rule.csv` | Kural bazında AUC değerleri |
| `outputs/submission.csv` | Test satırları için ihlal olasılıkları |
| `jigsaw_outputs.zip` | Colab yardımcısının indirdiği sonuç paketi |

`submission.csv` üretimi Kaggle'a gönderim yapıldığı anlamına gelmez. Kod test ve örnek gönderimdeki `row_id` sırasını kontrol eder.

## Teslim kontrolü

[Ders yönergesine](https://github.com/ceng313-2026) ve Classroom duyurusuna göre:

- [ ] Kullanıcı adları ve grup bilgileri asistana iletildi; organizasyon davetleri kabul edildi.
- [ ] Repo `ceng313-2026` organizasyonuna `GROUP34_CENG313` adıyla taşındı; üyelerin yazma erişimi doğrulandı.
- [ ] Veri ve veri içeren ZIP dosyaları güncel teslim ağacından çıkarıldı.
- [ ] `assignment1_eda_baseline.ipynb` tamamlandı ve baştan çalıştırılarak çıktıları kaydedildi.
- [ ] Kaggle skor kanıtı veya yönergedeki held-out değerlendirmesi eklendi.
- [ ] Her üyenin anlamlı katkıları commit geçmişinde görünüyor.
- [ ] Son teslim commit'i `assignment1` etiketiyle organizasyondaki repoya gönderildi.

**Assignment 1 son teslim:** 13 Kasım 2026, 23:59. **Assignment 2 son teslim:** 30 Aralık 2026, 23:59. Organizasyon dışındaki repolar değerlendirilmiyor.

Kaggle notebook veya hücreleri kopyalanmamalıdır. Yararlanılan fikirlerin kaynakları notebook'ta belirtilmeli ve her üye kodu açıklayabilmelidir.
