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
| 1 | Hybrid LLM (2404.14618) | ICLR 2024 | `views/bartscore` (referans repo'nun BARTScore'u) veya `views/ground_truth`; `scores` (N, M, k=10) — r_prob/r_trans k sample ister | önceki implementasyon silindi, yeni veriyle baştan yazılacak; küçük/büyük çifti havuzdan seçilmeli, NaN padding paydaya girmemeli |
| 3 | RouteLLM (2406.18665) | ICLR 2025 | sample-0 skorlarından her model çifti için battle | Elo/tier **yalnız train** battle'larından (referans repoda val/test sızıyor); GT judge |
| 4 | GraphRouter (2410.03834) | ICLR 2025 | `sources` = görev; `prompt_tokens` + train model/kaynak çıktı uzunluğu ve sabit input/output tarifesi | test kenarları görünmez; test sorguları çıkarımda düğüm olarak eklenir |
| 5 | MixLLM | NAACL 2025 | `extras/seconds` (gecikme), `completion_tokens`, `sample_cost` | online/sürekli öğrenme için sabit bir sorgu akışı sırası seçilmeli |
| 6 | UniRoute (2502.08773) | ICLR 2026 | bütün havuzun train/val skorları | dinamik havuz: test LLM'leri yalnız val'den görülür; `evaluation_models()` hook'u değerlendirmeyi o alt kümeye daraltır |
| 7 | BEST-Route | ICML 2025 | `extras/armorm` (selector), `extras/sample_cost` (n sample'ın gerçek maliyeti), `sample_index` | best-of-n kalitesi selector'ın seçtiği sample'ın GT'si; `max(GT)` oracle'dır |
| 8 | PILOT (Adaptive LLM Routing Under Budget Constraints) | EMNLP 2025 | sample-0 skor/maliyet, `trace` alanı | bütçe ve kronolojik akış simülasyonu runner'da yok; test sırası sabitlenmeli |

Arxiv revision'ı ve resmi kod her makale için implementasyondan önce sabitlenmelidir.

Hybrid LLM, GraphRouter ve UniRoute için makale/resmi kod incelemesi, referans repoda
taşınmaması gereken davranışlar ve `shared_v3` eğitim önerisi:
[üç router eğitim planı](uc_router_egitim_plani.md). Bu bir plan; yukarıdaki
router'ların implementasyon veya eğitim durumunu değiştirmez.

Model alt kümesi: bir router `fit` sonrası `evaluation_models()` döndürürse runner o
router'ı, tek modelleri ve bütün baseline'ları yalnız o modeller üzerinde ölçer
(UniRoute'un test LLM'leri). Varsayılan `None` = bütün aday modeller.

Eğitim havuzu: router `training_models()` döndürürse (config'ten, `fit`'ten önce belli)
runner `fit`'e yalnız o modellerin sütunlarını taşıyan train split'ini verir; val bütün
adayları taşır, çünkü yeni modelleri tanıtan küçük etiketli kümedir. Bu durumda
baseline'lar yalnız val'de (test havuzu) fit edilir, sonuçta `fit_on: val` yazar ve
leaderboard bu koşuları statik koşulardan ayrı gruplar.
`runner.training_splits()` ve `evaluation_split()` hem eğitimde hem denetimde
kullanılır. Test cohort'u, `fit` sonrasında değerlendirme havuzu belli olunca
hesaplanır; eğitim modellerinin test skorlarının eksik olması yeni modelleri
değerlendirmeyi engellemez. Boş model listesi hata, `None` bütün adaylar demektir.
Dinamik havuzda val yoksa train'e geri dönülmez. Her split'in query ID'leri ve
cohort hash'i kaydedilir; farklı cohort'lar leaderboard'da birleştirilmez.

Modeller hakkında bilgi: `bench.meta["model_info"]` her modelin HF ID'sini, revision'ını
ve `parameters_b` ile tarihli `pricing` tarifesini taşır (v1/v2: maliyet proxy'si).
Fiyatlar release config hash'iyle bağlıdır; `cost_definition` ve maliyetler fingerprint'tedir.
`model_info` yüklemede bu doğrulanmış config'ten eklenir. Makalenin istediği başka sabit bilgi (GraphRouter'ın model/görev açıklamaları)
router config'inde durur ve kamuya açık model kartlarından yazılır.

Tek operating point: `sweep` tek elemanlı liste döndürebilir (bir GraphRouter senaryosu).
O durumda AIQ boş kalır; kalite/maliyet noktası raporlanır.
`cost_at_max_quality` tek noktanın maliyetini de tabloya taşır; grafikte tek nokta
işaretçiyle gösterilir.

Otomatik kontrol: `llmr check-router configs/routers/<paper>.yaml`, CPU geliştirme
verisinde (`configs/benchmarks/synthetic_shared_v3.yaml`: ekibin 12 model adı, 7 kaynak,
0/1 ve token-F1 skorları, k=10) test skor/maliyetleri, bütün `extras` çıktı kanalları (`prompt_tokens` sabit kalır) ve
gold referansları karıştırılınca kararların değişmediğini sınar. Dinamik havuzda
yeni modellerin train skor/maliyet/extras değerleri NaN yapılarak eksikliğin de
eğitimi etkilemediği kontrol edilir. `describe()` runner ile aynı JSON yazıcısıyla
denetlenir; NumPy dizi/skalerlerini önce Python listesi/sayısına dönüştürün.
Bu kontrol yanıt üretilmeden seçim yapan üç router içindir; cascade'lerde ücretli
çağrı sırası ve erişilebilen yanıtlar ayrıca makaleye özgü test gerektirir.

`Decision.choice` döndürülen modeli, opsiyonel `sample_index` döndürülen gerçek
sample'ı gösterir. `Decision.cost` bütün çağrıların toplam ödenmiş maliyetidir.
Opsiyonel `trace` ileride çağrı/feedback sırasını taşımak içindir; henüz kronolojik
online simulator yoktur. Router'lar kendi `sweep` uygulamasıyla native karar
kuralını koruyabilir. Ortak evaluator yalnız dönen sample'ın kalitesini ve toplam
maliyetini ölçer; best-of-N'e ücretsiz oracle selection vermez.

Shared runner her metric/model pool için
sabit complete-case cohort kullanır, eksilen soruları raporlar. Tam cohort gerekirken
seçilmiş modelin eksik kalite skorunu sessizce atlamak yasaktır. Native evaluation
ve gerçek online protokoller ilgili paper implementasyonuyla ayrıca eklenmelidir.
