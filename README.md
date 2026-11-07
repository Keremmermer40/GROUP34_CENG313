# GROUP34_CENG313 — Jigsaw: Community Rule Violation Prediction

Bu proje, bir Reddit yorumunun verilen topluluk kuralını ihlal etme olasılığını tahmin eder.
CENG313 dersi Assignment 1 kapsamında başlangıç modelleri ve yerel doğrulama sonuçları sunulur.

- **Kaggle yarışması:** [Jigsaw – Agile Community Rules](https://www.kaggle.com/competitions/jigsaw-agile-community-rules)
- **Colab notebook:** [Notebook'u Google Colab'da aç](https://colab.research.google.com/drive/1EiCsKMU84Mmgpvb7a2x8FNm7ajzNoT56)
- **Notebook dosyası:** [assignment1_eda_baseline.ipynb](assignment1_eda_baseline.ipynb)
- **Veri paketi:** [Data.zip](Data.zip)

## Grup üyeleri

| Grup | Ad soyad | Öğrenci numarası | GitHub kullanıcı adı |

| 34 | Hacı Kerem Meremr | 23118080769| Keremmermer40 |
| 34 | Necdet Atakan Aydın | 24118080793 | necdetatakanaydin |

## Veri seti

Veri kaynağı yukarıdaki Kaggle yarışmasıdır. `Data.zip` paketinde aşağıdaki CSV dosyaları bulunmalıdır:

| Dosya | İçerik |

| `train.csv` | Yorumlar, kurallar, örnek yorumlar, subreddit ve hedef etiket |
| `test.csv` | Etiketi bulunmayan test yorumları |
| `sample_submission.csv` | Beklenen tahmin sütunları ve satır sırası |

Mevcut çalıştırmada eğitim verisi **2.029**, test verisi **10** satırdır.
Bu test dosyası örnek test verisidir; tam gizli yarışma testindeki başarıyı göstermez.

Temel sütunlar:

- `row_id`: Satır kimliği.
- `body`: Değerlendirilen Reddit yorumu.
- `rule`: Yorumun değerlendirileceği topluluk kuralı.
- `subreddit`: Yorumun ait olduğu topluluk.
- `positive_example_1`, `positive_example_2`: Kuralı ihlal eden örnekler.
- `negative_example_1`, `negative_example_2`: Kuralı ihlal etmeyen örnekler.
- `rule_violation`: Eğitim hedefi; `1` ihlal, `0` ihlal yok.

## Yaklaşım

Başlangıç modellerinde kural ve yorum metni
`RULE: <kural> COMMENT: <yorum>` biçiminde birleştirilir.
Subreddit ve örnek yorum sütunları mevcut modellerde kullanılmaz.

Üç model karşılaştırılır:

1. **DummyClassifier:** Eğitim sınıf oranlarına dayalı referans (`strategy="prior"`).
2. **Kelime TF-IDF + Logistic Regression:** 1–2 kelimelik n-gramlar.
3. **Karakter TF-IDF + Logistic Regression:** `char_wb` ile 3–5 karakterlik n-gramlar.

İki TF-IDF modelinde `min_df=2` ve en fazla 30.000 özellik;
Logistic Regression için `max_iter=1000` ve `random_state=42` kullanılır.

## Doğrulama ve değerlendirme

Aynı yorumun eğitim ve doğrulamada birlikte bulunmasını engellemek için yorumlar küçük harfe
çevrilip kenar boşlukları temizlenerek gruplandırılır. `GroupShuffleSplit` ile yaklaşık
%80 eğitim ve %20 doğrulama ayrılır. Deneyler **7, 21 ve 42** tohumlarında tekrarlanır.
TF-IDF her ayrımda yalnız eğitim bölümünde öğrenilir.

Raporlanan ölçütler genel ROC-AUC, ortalama kural ROC-AUC, accuracy ve F1'dır.
Ortalama kural ROC-AUC, her kuralın AUC değerinin eşit ağırlıklı ortalamasıdır.
Accuracy ve F1 için olasılık eşiği **0,5** kullanılır.

## Sonuçlar

Üç veri ayrımındaki yerel doğrulama sonuçlarının ortalaması:

| Model | ROC-AUC | Ortalama kural AUC | Accuracy | F1 |
||---:|---:|---:|---:|
| Dummy | 0.5000 | 0.5000 | 0.4894 | 0.6570 |
| Kelime TF-IDF + Logistic Regression | 0.8068 | 0.7948 | 0.7173 | 0.7254 |
| Karakter TF-IDF + Logistic Regression | 0.8140 | **0.8006** | 0.7082 | 0.7137 |

Karakter modeli ortalama kural AUC ölçütüne göre seçilir ve tüm eğitim verisiyle tekrar eğitilir.
Kelime modeli accuracy ve F1 bakımından daha iyi sonuç verir.
Test yorumları için ihlal olasılıkları `submission.csv` dosyasına yazılır.

Bu sonuçlar **yerel doğrulama sonuçlarıdır; resmi Kaggle skoru değildir.**
Rastgele bölmeler örtüşebilir ve model seçiminde de kullanılır; bağımsız test sonucu sayılmaz.
Bu yöntem görülmemiş kurallara genellemeyi ölçmez. Test etiketleri olmadığı için
10 örnek test satırı üzerinde başarı metriği hesaplanmaz.

## Google Colab'da çalıştırma

1. Yukarıdaki Colab bağlantısını açın veya `assignment1_eda_baseline.ipynb` dosyasını
   indirip Colab'da **File → Upload notebook** ile açın.
2. Repodaki `Data.zip` dosyasını indirip **bilgisayarınızda ZIP'ten çıkarın**.
3. CPU çalışma zamanı kullanarak notebook hücrelerini sırayla çalıştırın.
4. Veri yükleme hücresi dosya istediğinde **`train.csv`, `test.csv` ve
   `sample_submission.csv` dosyalarını birlikte seçin**.
   Mevcut hücre ZIP dosyasını doğrudan açmaz; bu aşamada `Data.zip` seçmeyin.
5. Eğitim ve değerlendirme hücrelerini çalıştırıp sonuç tablosunu inceleyin.
6. Son hücreyle `jigsaw_outputs.zip` sonuç paketini indirin.

Gerekli kütüphaneler notebook'un ilk hücresinde kurulur: `pandas` ve `scikit-learn`.
Colab oturum diski geçicidir. Sonuç ZIP'ini indirmek notebook'u kaydetmez;
çıktıların göründüğü notebook'u ayrıca kaydedin.

## Üretilen dosyalar

- `outputs/results.csv`: Model ve tohum başına doğrulama metrikleri.
- `outputs/per_rule.csv`: Kural bazında AUC değerleri.
- `outputs/submission.csv`: Test yorumları için ihlal olasılıkları.
- `jigsaw_outputs.zip`: Bu sonuç dosyalarının indirme paketi.

## Assignment 1 durumu

Mevcut notebook baseline modellerini ve sonuçlarını içerir.
Teslimin EDA bölümünü tamamlamak için veri boyutu/türleri, eksik değerler,
tekrarlar, hedef dağılımı, kural ve subreddit dağılımları ile yorum uzunlukları
tablolar, grafikler ve kısa yorumlarla notebook'a eklenmelidir.

Ders yönergesindeki ortak kod şartı için tekrar kullanılan fonksiyon ve sınıflar
aynı repodaki `.py` dosyalarına taşınmalı ve notebook bu dosyaları kullanmalıdır.
Mevcut paylaşılan notebook fonksiyonları kendi hücrelerinde tanımlar.

## İlerleme günlüğü

| Tarih | Çalışma |
|---|---|
| 26 Ekim 2026 | İlk proje yüklemesi ve `.gitignore` düzenlemesi |
| 29 Ekim 2026 | Proje dosyaları ve verilerin eklenmesi |
| 7 Kasım 2026 | Baseline karşılaştırması ve Colab çalıştırma hazırlığı |
| 7 Kasım 2026 | GitHub teslimi için README hazırlığı |

Günlük geliştirme ilerledikçe güncellenecektir. Yeni çalışmalar anlamlı commit mesajlarıyla kaydedilmelidir.

