# Ekip çalışma rehberi

Çalışma reposu `damlaakemall/purerepo`; hedef SU Lab LLM-Routing-Project'in ekip için
ayrı sürümü. Başlangıç: [teknik protokol](shared_protocol.md).

## 1. Laptop'ta güvenli kontrol

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
llmr smoke-pipeline --output /tmp/llmr-shared-smoke
```

Smoke; 30 küçük soru, iki sahte model, değişken sample sayısı, iki üye ve iki shard
ile 120 sentetik yanıtı skorlar, paketler, birleştirir, doğrular ve grafik üretir.
Gerçek model, GPU, API veya router training kullanılmaz. Aynı output'a ikinci
kez release yazılmaz; yeniden test için yeni/boş bir output yolu verin.
`/tmp/llmr-shared-smoke/reports/smoke-v1/` altında raporlar vardır.
Mevcut testler mevcut küçük CPU router eğitimlerini de doğrular; bunlar büyük model
veya deney eğitimi değildir.

## 2. Koordinatör: benchmark'ı kararlaştır ve bir kez hazırla

`configs/benchmarks/shared_v1.yaml` ve `configs/model_pools/team.yaml` taslaktır.
Dataset/model/ArmoRM `PIN_*` değerlerini sabit commit SHA'larıyla değiştirin.
Model havuzunu, her modelin sample sayısını, prompt/system ayarlarını, GT evaluator'ı,
scoring text policy'sini ve ortak cost tanımını kararlaştırın. Proxy maliyetle
başlarsanız raporlarda token_proxy kullanılır; USD sonuçları için doğrulanmış
tarihli input/output fiyatlarıyla `cost.definition: usd_tokens` seçin.

```bash
export LLMR_DATA_ROOT=/shared/team/llmr_data
llmr build-queries configs/benchmarks/shared_v1.yaml
```

Bu komut gerçek dataset indirir, generation yapmaz. Üyelere YAML/model pool ile
`$LLMR_DATA_ROOT/benchmarks/shared_v1/dataset/` klasörünün byte-identical kopyasını
verin (paylaşımlı Tosun dizini veya rsync). Her üye kendi yazılabilir collection
path'inde aynı dataset kopyasını kullanabilir. Dataset tekrar üretilmesin.
Üye-model atamalarını ekip kayıtlarında tutun; model adlarını değiştirmeyin.

## 3. Üye: sadece atandığın modeli üret

Gerekli backend'i kurun: HF için `pip install -e '.[gpu]'`, vLLM için ayrı uyumlu
CUDA ortamında `pip install -e '.[vllm]'`, API için `pip install -e '.[api]'`.
ArmoRM ortamını model kartıyla uyumlu torch/transformers sürümleriyle sabitleyin;
`pip freeze` ve GPU bilgilerini ilk pilotta saklayın. Burada gerçek GPU uyumluluğu
doğrulanmadı. Tosun GPU/driver/modül ayarları site ayarlarıyla doğrulanmalıdır.

Aşağıdaki komutlar gerçek generation/scoring başlatır; ilk pilot aşamasına kadar
çalıştırılmamalıdır. Her shard için aynı dört argümanı kullanın:

```bash
CFG=configs/benchmarks/shared_v1.yaml
MODEL=qwen2.5-0.5b
MEMBER=ibrahim
llmr generate "$CFG" --model "$MODEL" --member "$MEMBER" --num-shards 4 --shard-index 0 --max-retries 2
llmr score "$CFG" --model "$MODEL" --member "$MEMBER" --num-shards 4 --shard-index 0 --scorer ground_truth
llmr score "$CFG" --model "$MODEL" --member "$MEMBER" --num-shards 4 --shard-index 0 --scorer armorm
llmr package-contribution "$CFG" --model "$MODEL" --member "$MEMBER" --num-shards 4 --shard-index 0
```

Diğer işler shard index 1, 2, 3'ü üretir. Aynı komutu tekrar çalıştırmak resume'dur;
başarılı sample/score yeniden üretilmez. Config değiştirip eski işe resume etmeyin.
Model/sample ayarını değiştirmek yeni benchmark veya yeni sürüm demektir. Global
ayarlarla uyumsuz contribution birleştirmede reddedilir. API anahtarını yalnız
YAML'ın işaret ettiği environment değişkenine koyun. Batch/Sampling parametrelerini
model servisi desteklemiyorsa ilgili API ayarlarını açıkça yapılandırın.

Tosun'da `cluster/tosun/config.sh` içindeki site ayarlarını doğruladıktan sonra:

```bash
bash cluster/tosun/submit_member.sh configs/benchmarks/shared_v1.yaml qwen2.5-0.5b ibrahim 4
```

Bu wrapper generation → CPU GT → GPU ArmoRM → paketleme bağımlılıklarını submit
eder; router training submit etmez. Her aşama aynı shard array'ini kullanır.
API generation GPU istemez; offline ArmoRM ayrı GPU aşamasıdır. Zaman aşımında
başarısız stage'i aynı argümanlarla yeniden submit edin; tamamlanmış paketleri
yeniden paketlemeyin. `SLURM` kuyruk/hesap/GRES değerleri bu makinede sınanmadı.

## 4. Paketleri teslim et ve birleştir

Üyenin `exports/` altındaki tamamlanmış paket dizinlerini ortak collection'ın
`exports/` dizinine rsync ile taşıyın. Eksik kopya veya değişmiş dosya hash kontrolünde
reddedilir. Alternatif olarak YAML'da `contribution_paths` glob listesi belirtin.
Paket bir kez teslim edildikten sonra içerik değiştirmeyin.

```bash
llmr assemble configs/benchmarks/shared_v1.yaml
llmr validate-data configs/benchmarks/shared_v1.yaml
llmr evaluate-data configs/benchmarks/shared_v1.yaml
```

Çıktılar `frozen/v1/` altında, raporlar `reports/v1/` altındadır. Assemble eksik
model/sample/score, çakışma veya uyumsuz provenance bulursa yayımlamaz. Eksik
bir prototip release bilinçli isteniyorsa `allow_partial=true` kullanılabilir;
coverage raporlarını kontrol edin. Yeni veri veya fiyat düzeltmesi eski release
üzerine yazılmaz: yeni YAML/version/path oluşturulur.

## 5. HF Hub veya Tosun ile paylaş

Tosun için bütün `frozen/v1/` klasörünü salt okunur bir release dizinine kopyalayın;
manifestteki bütün dosyaları birlikte taşıyın. `data/`, weights ve matrisler Git'e
girmez; sadece kod/config/küçük fixture'lar commit edilir.

Hub için benchmark YAML'ında özel bir dataset reposu belirleyin; her benchmark
sürümü için ayrı Hub repo kullanın. HF token'ını environment/login üzerinden verin:

```bash
llmr push-data configs/benchmarks/shared_v1.yaml
# Komutun yazdığı commit SHA'yı hub_revision alanına koyun.
llmr pull-data configs/benchmarks/shared_v1.yaml
```

`pull-data` boş bir yerel release yoluna, sabit `hub_revision` commit'inden indirir,
manifesti doğrular ve atomik yerleştirir. Mevcut release üzerine yazmaz. Push/pull
bu çalışma sırasında gerçekten yapılmadı. Katkı paketleri de aynı Hub helper'ının
`push(bundle_dir, repo_id)` fonksiyonuyla ayrı private dataset reposunda paylaşılabilir;
benchmark release'i için `push-data` yalnız frozen içeriği yükler.

## 6. Makalelere sırayla geç

[Router sözleşmesi](router_contract.md) ve [paper şablonu](papers/_template.md) ile
her makale için önce label → architecture → loss → train → inference protokolünü
çıkarın; paper/released code farklarını kaydedin. Önerilen sıra: RouteLLM,
GraphRouter, UniRoute, BEST-Route, MixLLM, PILOT. Bu sıra uygulama değildir ve
paper-faithfulness iddiası taşımaz; makale başına doğrulama ayrıca yapılacak.

Mevcut bir router'ı ortak veri üzerinde ileride çalıştırırken experiment config'inde
benchmark olarak shared YAML'ı seçin. `quality_metric: ground_truth` ve `armorm` için
ayrı deneyler yapın; label metriğini paper gerekçesi olmadan değiştirmeyin. Eşik ve
checkpoint val'den, son rapor aynı test sorularından seçilir. Paper-native metrikleri
ortak quality/cost raporunun yanına ekleyin. En az üç seed ile raporlayın.
