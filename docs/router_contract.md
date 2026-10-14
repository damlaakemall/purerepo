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

| yöntem | implementasyondan önce çözülmesi gereken sözleşme |
|---|---|
| RouteLLM | strong/weak ve preference/battle construction, resmi varyant, embedding, objective, calibration |
| GraphRouter | graph node/edge/features, görülebilir train/val edges, message passing, reward/label ve loss |
| UniRoute | query ve model genellemesi/pool split, model-native representation, risk/cost objective ve karar kuralı |
| BEST-Route | ragged sample supervision, `(model,N)` action, proxy/selector eğitimi, actual returned sample, bütün üretim ve selector cost'u |
| MixLLM | ilgili paper/revision netleştirme, offline/online feedback varsayımları, label/architecture/loss ve inference |
| PILOT | ilgili paper/revision netleştirme, observation/feedback zamanlaması, training/inference state ve cost |

Bu tablo algorithm tanımı veya implementasyon iddiası değildir. Özellikle aynı
adla birden fazla çalışma bulunabildiği için paper DOI/arXiv revision ve resmi kod
önce sabitlenmelidir. Yeni router'lar bu değişiklikte yazılmadı.

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
