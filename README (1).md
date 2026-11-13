# Yerel değerlendirme kanıtları

Bu klasör Kaggle skorunu değil, notebook çalıştırmasında üretilen yerel held-out sonuçlarını içerir.

- `heldout_metrics.csv`: seçilen model, bölme boyutları ve test metrikleri.
- `heldout_per_rule.csv`: kural başına AUC ve satır sayıları.
- `development_results.csv`: yalnızca geliştirme bölümünde yapılan model karşılaştırması.
- `heldout_score.png`: gerçek yerel metriklerden üretilen sonuç görseli; Kaggle ekran görüntüsü değildir.

Ayrım: hedefe göre stratified %20 test, random_state=42. Testteki yorumlarla eşleşen geliştirme satırları çıkarılır. Model seçimi kalan geliştirme verisinde yapılır. Önceki proje keşifleri tüm veriyi içerdiğinden tarihsel test bağımsızlığı iddia edilmez.

Resmi Kaggle gönderimi ve skor ekran görüntüsü bulunmuyor. Birinci ve beşinci sıra çözüm raporlarından alınan skorlarla karşılaştırma notebook ve README içinde kaynaklarıyla verilmiştir.
