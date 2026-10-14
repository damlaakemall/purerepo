# Router sözleşmesi: ortak veri, makaleye özgü öğrenme

Mevcut `Router.fit(train, val)` ve `sweep(benchmark) -> list[Decision]` korunur.
BinaryRouter/MultiModelRouter yardımcıdır; her yöntemi bu yardımcıların
classifier/lambda kararına uydurmak gerekmez. Doğrudan Router subclass'ı kendi
mimarisini, loss'unu, eğitim prosedürünü ve inference politikasını uygulayabilir.
`prepare_labels` için ortak zorunlu bir format veya bütün router'lar için ortak
embedding/backbone dayatılmaz.

Her paper paketinde ayrı `labels.py`/adapter adımı tutulmalı. Bu adım train ve
val'in ortak sample skorlarından paper'ın kendi supervision sinyalini üretir.
Test label'ları checkpoint/threshold seçimine girmez. Label artifact manifesti;
benchmark dataset/content/fingerprint'ini, kullanılan metric'i, sample politikasını,
model sırasını, label-config hash'ini, paper versiyonunu ve seed'i kaydetmelidir.

Ekibin planladığı router'lar ve ortak release'in her birine verdiği veri (router'lar
henüz implemente edilmedi; bu tablo implementasyon iddiası değildir):

| # | makale | yayın | release'te hazır olan | implementasyonda dikkat |
|---|---|---|---|---|
| 1 | Hybrid LLM (2404.14618) | ICLR 2024 | `scores` (N, M, k=10) — r_prob/r_trans k sample ister | mevcut implementasyon; küçük/büyük çifti havuzdan seçilmeli |
| 3 | RouteLLM (2406.18665) | ICLR 2025 | sample-0 skorlarından her model çifti için battle | Elo/tier **yalnız train** battle'larından (referans repoda val/test sızıyor); GT judge |
| 4 | GraphRouter (2410.03834) | ICLR 2025 | `sources` = task düğümü; maliyet (model, kaynak) başına train ortalaması | test kenarları görünmez; test sorguları çıkarımda düğüm olarak eklenir |
| 5 | MixLLM | NAACL 2025 | `extras/seconds` (gecikme), `completion_tokens`, `sample_cost` | online/sürekli öğrenme için sabit bir sorgu akışı sırası seçilmeli |
| 6 | UniRoute (2502.08773) | ICLR 2026 | bütün havuzun train/val skorları | dinamik havuz: test LLM'leri yalnız val'den görülür; `evaluation_models()` hook'u değerlendirmeyi o alt kümeye daraltır |
| 7 | BEST-Route | ICML 2025 | `extras/armorm` (selector), `extras/sample_cost` (n sample'ın gerçek maliyeti), `sample_index` | best-of-n kalitesi selector'ın seçtiği sample'ın GT'si; `max(GT)` oracle'dır |
| 8 | PILOT (Adaptive LLM Routing Under Budget Constraints) | EMNLP 2025 | sample-0 skor/maliyet, `trace` alanı | bütçe ve kronolojik akış simülasyonu runner'da yok; test sırası sabitlenmeli |

Arxiv revision'ı ve resmi kod her makale için implementasyondan önce sabitlenmelidir.

Model alt kümesi: bir router `fit` sonrası `evaluation_models()` döndürürse runner o
router'ı, tek modelleri ve bütün baseline'ları yalnız o modeller üzerinde ölçer
(UniRoute'un test LLM'leri). Varsayılan `None` = bütün aday modeller.

`Decision.choice` döndürülen modeli, opsiyonel `sample_index` döndürülen gerçek
sample'ı gösterir. `Decision.cost` bütün çağrıların toplam ödenmiş maliyetidir.
Opsiyonel `trace` ileride çağrı/feedback sırasını taşımak içindir; henüz kronolojik
online simulator yoktur. Router'lar kendi `sweep` uygulamasıyla native karar
kuralını koruyabilir. Ortak evaluator yalnız dönen sample'ın kalitesini ve toplam
maliyetini ölçer; best-of-N'e ücretsiz oracle selection vermez.

Mevcut baseline/Hybrid LLM davranışı değişmez; sadece NaN padding'e sahip
probabilistic label paydası düzeltilmiştir. Shared runner her metric/model pool için
sabit complete-case cohort kullanır, eksilen soruları raporlar. Tam cohort gerekirken
seçilmiş modelin eksik kalite skorunu sessizce atlamak yasaktır. Native evaluation
ve gerçek online protokoller ilgili paper implementasyonuyla ayrıca eklenmelidir.
