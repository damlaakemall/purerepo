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
Gerçek model, GPU veya router training kullanılmaz. Aynı output'a ikinci
kez release yazılmaz; yeniden test için yeni/boş bir output yolu verin.
`/tmp/llmr-shared-smoke/reports/smoke-v1/` altında raporlar vardır.
Mevcut testler mevcut küçük CPU router eğitimlerini de doğrular; bunlar büyük model
veya deney eğitimi değildir.

## 2. Koordinatör: benchmark'ı bir kez hazırla

Ücretli API yok; her şey Tosun GPU'larında çalışır. `configs/benchmarks/shared_v1.yaml`
(7 kaynak, 60/10/30 split, `k: 10`) ve `configs/model_pools/team.yaml` (9 açık ağırlıklı
model) bütün dataset/model/ArmoRM revision'ları sabitlenmiş halde hazırdır. Maliyet
`token_proxy`: aktif parametre sayısı × token; USD değildir (DECISIONS.md #2).

```bash
source cluster/tosun/config.sh && llmr_activate
llmr build-queries configs/benchmarks/shared_v1.yaml
```

Bu komut dataset'leri indirir (login node'da yeterli), generation yapmaz. Dataset
`$LLMR_DATA_ROOT/benchmarks/shared_v1/dataset/` altında bir kez donar; üyeler bu klasörün
byte-identical kopyasını kullanır (paylaşımlı Tosun dizini veya rsync). Dataset tekrar
üretilmez. Üye-model atamalarını ekip kayıtlarında tutun; model adlarını değiştirmeyin.

## 3. Üye: sadece atandığın modeli üret

Kurulum bir kez: `bash cluster/tosun/setup_env.sh` (venv, CUDA 12.8 torch, vLLM).
ArmoRM ve vLLM bu repoda henüz gerçek GPU'da çalıştırılmadı: ilk iş küçük bir pilot
olsun (ör. `n` değerleri küçük bir kopya config ile tek shard) ve `pip freeze` ile GPU
bilgisini saklayın.

```bash
bash cluster/tosun/submit_member.sh configs/benchmarks/shared_v1.yaml qwen3-4b ibrahim 4
```

Wrapper modelin `gpus` değerine göre GPU ister (27B ve 26B-A4B için 2), generation'dan
sonra CPU ground-truth ve GPU ArmoRM skorlamasını paralel, ardından paketlemeyi submit
eder; router training submit etmez. `LLMR_BATCH_SIZE` vLLM'e tek seferde giden istek
sayısıdır (varsayılan 256). Elle çalıştırmak isterseniz her shard için aynı dört argüman:

```bash
CFG=configs/benchmarks/shared_v1.yaml; MODEL=qwen3-4b; MEMBER=ibrahim
llmr generate "$CFG" --model "$MODEL" --member "$MEMBER" --num-shards 4 --shard-index 0 --max-retries 2 --batch-size 256
llmr score "$CFG" --model "$MODEL" --member "$MEMBER" --num-shards 4 --shard-index 0 --scorer ground_truth
llmr score "$CFG" --model "$MODEL" --member "$MEMBER" --num-shards 4 --shard-index 0 --scorer armorm
llmr package-contribution "$CFG" --model "$MODEL" --member "$MEMBER" --num-shards 4 --shard-index 0
```

Aynı komutu tekrar çalıştırmak resume'dur; başarılı sample/score yeniden üretilmez.
Zaman aşımında başarısız stage'i aynı argümanlarla yeniden submit edin; tamamlanmış
paketleri yeniden paketlemeyin. Config değiştirip eski işe resume etmeyin: model/sample
ayarını değiştirmek yeni benchmark sürümü demektir.

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
çıkarın; paper/released code farklarını kaydedin. Planlanan router'lar: Hybrid LLM
(hazır), RouteLLM, GraphRouter, MixLLM, UniRoute, BEST-Route, PILOT. Her birinin
release'ten ne aldığı router sözleşmesindeki tabloda.

Mevcut bir router'ı ortak veri üzerinde ileride çalıştırırken experiment config'inde
benchmark olarak shared YAML'ı seçin. `quality_metric: ground_truth` ve `armorm` için
ayrı deneyler yapın; label metriğini paper gerekçesi olmadan değiştirmeyin. Eşik ve
checkpoint val'den, son rapor aynı test sorularından seçilir. Paper-native metrikleri
ortak quality/cost raporunun yanına ekleyin. En az üç seed ile raporlayın.
