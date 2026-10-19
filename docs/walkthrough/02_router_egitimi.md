# Adım 2: router eğitimine hazırlık

> **Durum — 19 Ekim 2026:** ortak altyapı CPU'da doğrulandı. Donmuş bir test-double
> release'i mevcut eğitim komutuyla okunabiliyor. Hybrid LLM, GraphRouter ve UniRoute
> henüz implemente edilmedi; gerçek veriyle veya Tosun GPU'sunda router eğitilmedi.

## Büyük resim

```text
üye cevapları + ayrı kalite skorları
    → doğrulanmış, donmuş release
    → Benchmark görünümü (bir kalite kanalı)
    → router'ın kendi labels.py adımı
    → fit(train, val)
    → sweep(test)
    → aynı cohort üzerinde kalite/maliyet ve baseline'lar
```

SU Lab referans projesindeki ortak matris → makaleye özgü adapter → eğitim →
değerlendirme ayrımını koruyoruz. Bizde adapter `routers/<paper>/labels.py`, ortak
girdi `Benchmark`, eğitim girişi `fit(train, val)`. Yeni bir eğitim framework'ü,
router'lara ortak bir mimari veya ortak bir label tanımı eklenmedi. Referans
incelemesinin commit'leri ve gerekli yöntem farkları [eğitim planında](../team/uc_router_egitim_plani.md).

## 1. Aynı veri sınırı eğitimde ve denetimde

`runner.training_splits()` yalnız izin verilen modellerin train sütunlarını seçer;
sonra eksik sample-0 kalite/maliyet kayıtlarını ortak cohort'tan çıkarır. Val bütün
adayları taşır. `fit` test verisini almaz.

`runner.evaluation_split()` fit sonrasında test model havuzunu seçer, ardından bu
havuzun eksiksiz test sorularını belirler. Böylece UniRoute'un öğrenme modellerinin
test etiketleri eksik olsa da yeni modeller değerlendirilebilir. Baseline'lar aynı
test sorularında ölçülür; dinamik havuzda yalnız yeni modellerin val verisiyle eğitilir.
Her split'in soru ID'leri ve hash'i kaydedilir. Leaderboard farklı havuz veya
cohort'ların sonuçlarını birleştirmez.

Bu iki yardımcıyı `check-router` da kullanır; denetim ve gerçek eğitim birbirinden
farklı split kuralları uygulamaz. Testin eksikliği train seçimini değiştirmez.

## 2. Router implementasyonunu bağlayacağımız yerler

| yöntem | mevcut bağlantı | daha sonra kendi paketinde eklenecek |
|---|---|---|
| Hybrid LLM | `BinaryRouter`, soft-label classifier, val checkpoint ve val'de seçilmiş eşik raporu | det/prob/trans label'ları, model çifti ve encoder config'i |
| GraphRouter | doğrudan `Router`, tek `Decision`, kalite/maliyet noktası ve grafikte işaretçi | graph, kenar label'ları, senaryolar ve açıklamalar |
| UniRoute | `training_models()`, `evaluation_models()`, val-only baseline'lar ve frozen embedding yardımcısı | model ayrımı, K-means/LearnedMap, yeni model profilleri |

Makalelerin kendi karar kuralları korunacak. GraphRouter'ın tek noktası için AIQ
boş; `max_quality` ve `cost_at_max_quality` kalite/maliyet çiftini verir. Kaynak
bazında sonuçlar ve makaleye özgü ölçütler ilgili implementasyonla eklenecek.
Etiket üretiminde val/test sınırlarının doğruluğu ayrıca yöntemin testlerinde sınanmalı.

## 3. Kontrolün kapsamı

`llmr check-router` query ve source'u sabit tutup test skorlarını, maliyetlerini,
`extras` çıktı kanallarını (`prompt_tokens` sabit kalır) ve gold referanslarını karıştırır. Karar değişirse sonuçlardan
yararlanılmış olabilir. Dinamik havuzda yeni modellerin train skor/maliyet/extras
hücreleri NaN yapılır; bu değişiklik eğitim cohort'unu veya kararları etkilememeli.
`describe()` aynı JSON yazıcısıyla denetlenir; kaydedilemeyen NumPy dizisi kontrolü geçmez.

Regresyon testleri, test ArmoRM'sini doğrudan okuyan router'ın reddedildiğini ve
öğrenme modellerinin test skorları eksikken geçerli dinamik router'ın çalıştığını
gösterir. Bu, makaleye uygunluk ispatı veya ücretli çağrı sıralı cascade denetimi değildir.

## Nasıl çalıştırdık

Repo kökünde, yerel `.venv` ile CPU'da aşağıdaki komutlar çalıştırıldı. `/tmp` yolu
bu doğrulama için kullanıldı; smoke release'i yeniden oluşturmak için yeni yol seçilir.

```bash
.venv/bin/python -m pytest -q
.venv/bin/llmr check-router configs/routers/quality_regression.yaml
.venv/bin/llmr train configs/experiments/quality_regression__synthetic_shared.yaml output_dir=/tmp/llmr-router-readiness-20261019/runs benchmark.path=/tmp/llmr-router-readiness-20261019/bench
.venv/bin/llmr smoke-pipeline --output /tmp/llmr-router-readiness-20261019/release
```

Donmuş release'i eğitim komutuna bağlamak için `/tmp/llmr-router-readiness-20261019/release_training.yaml`
geçici config'i oluşturuldu. İçeriği:

```yaml
name: quality_regression__release_readiness
seed: 0
benchmark: /tmp/llmr-router-readiness-20261019/release/frozen/smoke-v1/config.json
router: configs/routers/quality_regression.yaml
baselines: [oracle]
wandb: {enabled: false}
output_dir: /tmp/llmr-router-readiness-20261019/runs
```

```bash
.venv/bin/llmr train /tmp/llmr-router-readiness-20261019/release_training.yaml
```

## Örnek: gerçek bir altyapı doğrulaması

| kontrol | sonuç |
|---|---|
| testler | 97 geçti |
| `check-router`, quality regression | bütün kontroller geçti |
| shared biçimli sentetik eğitim | 12 model, k=10; 1684 train, 278 val; 900 test sorusunun 859'u complete-case |
| koleksiyon smoke | 30 soru, 2 model, 120 test-double cevap; release doğrulandı |
| bu release'ten CLI eğitimi | 18 train / 6 val / 6 test; `metrics.json`, `curve.csv`, `curve.png` yazıldı |

Bu sayılar sentetik altyapı kontrolüdür; gerçek model başarısı değildir. Testlerde
release oluşturulduktan sonra üretim/puanlama fonksiyonları kapatılarak eğitimin
yalnız release okuduğu da doğrulanıyor. CI, testlere ek olarak contract kontrolünü,
iki CPU eğitim smoke'unu ve koleksiyon smoke'unu çalıştırır.

## V3 fiyatlı havuzun CPU kontrolü

Bu adımın devamında yeni koleksiyon `shared_v3` olarak hazırlandı. [Havuz ve fiyatlar](01_egitim_verisi.md#shared_v3-fiyatı-doğrulanabilen-model-havuzu)
ayrı ayrı pinli; `synthetic_shared_v3` aynı 12 adı ve tarifeyi taşır, skorları/token
uzunlukları tamamen uydurmadır. Varsayılan `check-router` ve CI geliştirme eğitimi
artık bu config'i kullanır. Önceki bölümdeki 97-test kaydı önceki hazırlığa aittir.

19 Ekim 2026, yerel CPU'da gerçekten çalıştırılan komutlar:

```bash
.venv/bin/pytest -q
.venv/bin/llmr check-router configs/routers/quality_regression.yaml
.venv/bin/llmr train configs/experiments/quality_regression__synthetic_shared_v3.yaml output_dir=/tmp/llmr-v3-readiness-20261019-verified/runs benchmark.path=/tmp/llmr-v3-readiness-20261019-verified/bench
.venv/bin/llmr smoke-pipeline --output /tmp/llmr-v3-readiness-20261019-verified/collection
```

| kontrol | gözlenen sonuç |
|---|---|
| testler | 110 geçti; 11.32 saniye |
| contract | kayıt, geçerli kararlar, test çıktılarından bağımsızlık, JSON: hepsi geçti |
| v3 sentetik veri | 3000 soru, 12 model, k=10; fingerprint `129c3868120a7516`, kaydet–yükle korundu |
| v3 CPU eğitim | 1684 train / 278 val; 900 test sorusunun 859'u complete-case |
| eğitim artifact'leri | `metrics.json`, `curve.csv`, `curve.png`; tahmini API USD etiketi ve `cost_definition` |
| koleksiyon smoke | 30 soru, 2 model, 120 test-double cevap, immutable release doğrulandı |
| USD release regresyonu | 30 soru × 2 fiyatlı model × 2 sample = 120 yanıt; assemble → validate → load → evaluate → baseline train geçti |
| muhasebe | 1000 input + 2000 output için Llama3.2-3B `$0.00071`; tarife değişmiş config ile release yükleme reddedildi |

Yeni testler ayrıca eksik/negatif/NaN/sonsuz fiyatın generation'dan önce reddini,
yeni havuzun 6 modeli revision'larıyla korumasını, kaynak/split/scoring protokolünün
korunmasını, public prompt uzunluğu ile gizli completion ayrımını ve HF conditional
model sınıfını test double ile kontrol eder. Gerçek ağırlık/tokenizer indirilmedi.
Sentetik başarı skorları model/router başarımı diye raporlanmaz; gerçek v3 GPU
pilotundan sonra toplanan veriyle üç router'ın implementasyon/eğitimi yapılacak.

## Sırada ne var?

Veri tamamlanınca aynı loader, seçilen kalite görünümünü donmuş release'ten okuyacak;
eğitim sırasında üretim/puanlama başlamayacak. Gerçek Tosun job akışı
[cluster rehberinde](../../cluster/tosun/README.md); her router kendi deney YAML'ini
ekleyince mevcut `submit_train.sh` üzerinden en az üç seed ile çalışacak.

Şimdilik açık kalanlar [DECISIONS.md](../../DECISIONS.md) içinde: veri/kalite kanalı,
Hybrid çifti ve label politikası, UniRoute model ayrımı/encoder'ı. GraphRouter ve
UniRoute'un çekirdek kodunda SU Lab referansını başlangıç alma tercihi kullanıcı
yönlendirmesiyle kaydedildi; korunacak fonksiyonlar ve zorunlu adapter farkları
[eğitim planında](../team/uc_router_egitim_plani.md#referansa-yakın-uygulama-sınırı).
Üç implementasyon ve gerçek koşular tamamlandıkça aynı yazıya yöntem bölümleri eklenecek.
