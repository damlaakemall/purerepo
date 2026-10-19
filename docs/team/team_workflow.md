# Ekip çalışma rehberi

Çalışma reposu `damlaakemall/purerepo`; hedef SU Lab LLM-Routing-Project'in ekip için
ayrı sürümü. Başlangıç: [teknik protokol](shared_protocol.md).

## 0. Üç kişiyiz: kim ne yapar

12 model üç kişiye dörder dörder dağıtıldı. Atamaların tek kaynağı
[`cluster/tosun/team.yaml`](../../cluster/tosun/team.yaml); `tests/test_team.py` her modelin tam bir kez
atandığını kontrol eder.

| kişi | `--member` | modeller | toplam |
|---|---|---|---|
| İbrahim G (koordinatör) | `ibrahimg` | qwen2.5-32b (2 GPU), olmo-3-7b, mistral-7b-v0.3, phi-4-mini | 51.1B |
| Damla | `damla` | gemma-3-27b (2 GPU), llama-3.1-8b, qwen2.5-7b, llama-3.2-3b | 46.2B |
| İbrahim B | `ibrahimb` | mistral-small-24b (2 GPU), phi-4, deepseek-r1-distill-llama-8b, smollm3-3b | 49.4B |

Neden böyle: herkese bir büyük (2 GPU'luk) model düşer ve parametre toplamı (kabaca GPU süresi) yakın
tutulur. Lisans isteyen (gated) üç model, iki Llama ve Gemma, Damla'dadır; Llama ve Gemma lisansını
huggingface.co'da yalnız Damla kabul eder. İbrahim B'deki phi-4 en az 40 GB'lık kart ister,
deepseek-r1-distill-llama-8b ise uzun düşünür, o yüzden işi uzun sürer.

`--member` adı veriye yazılır (`contributions/<member>/`, `exports/<member>-...`); toplama başladıktan sonra
değiştirmeyin. Model değiş tokuşu gerekirse `team.yaml`'ı düzenleyin ve modeli henüz kimse üretmeye başlamamış olsun.

Sıra:

1. **Koordinatör** (İbrahim G) pilot ve `shared_v1` dataset'ini bir kez üretir ve diğer ikisiyle paylaşır (bölüm 2).
2. **Herkes** kendi dört modelinin pilotunu koşar ve kontrol eder:
   `bash cluster/tosun/submit_my_models.sh configs/benchmarks/shared_pilot.yaml <member> 1`,
   ardından her model için `python cluster/tosun/inspect_shard.py configs/benchmarks/shared_pilot.yaml <model> <member>`.
3. Pilotlar temizse **herkes** tam toplamayı başlatır:
   `bash cluster/tosun/submit_my_models.sh configs/benchmarks/shared_v1.yaml <member> 4` (bölüm 3).
4. **Herkes** paketlerini koordinatöre teslim eder; **koordinatör** birleştirir ve doğrular (bölüm 4).

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
(7 kaynak, 60/10/30 split, `k: 10`) ve `configs/model_pools/team.yaml` (referanstan farklı
12 açık ağırlıklı model, 3B-33B) bütün dataset/model/ArmoRM revision'ları sabitlenmiş halde hazırdır. Maliyet
`token_proxy`: parametre sayısı × token; USD değildir (DECISIONS.md #2).

```bash
source cluster/tosun/config.sh && llmr_activate
llmr build-queries configs/benchmarks/shared_v1.yaml
```

Bu komut dataset'leri indirir ve MiniLM ile yakın kopyaları gruplar, generation yapmaz; login node'un bellek
sınırı yüzünden `sbatch` ile compute node'da çalıştırın ([anlatım](../walkthrough/01_egitim_verisi.md#3-cevap-üretme-tosunda-bir-modeli-çalıştırmak)). Dataset
`$LLMR_DATA_ROOT/benchmarks/shared_v1/dataset/` altında bir kez donar; Damla ve İbrahim B bu klasörün
byte-identical kopyasını kendi `$LLMR_DATA_ROOT/benchmarks/shared_v1/dataset/` yoluna alır (koordinatörün
okumaya açtığı bir Tosun dizininden `rsync -a` ile veya bir arşiv olarak). Pilot için aynısı
`benchmarks/shared_pilot/dataset/` ile yapılır. Dataset tekrar üretilmez; kopya bozuksa
`submit_member.sh` GPU istemeden önce hash kontrolünde durur. Üye-model atamaları
[`cluster/tosun/team.yaml`](../../cluster/tosun/team.yaml)'dadır; model adlarını değiştirmeyin.

## 3. Üye: sadece atandığın modeli üret

Kurulum bir kez, compute node'da: `setup_env.sh` (venv, CUDA 12.9 torch ve vLLM; Tosun sürücüsü 570). İlk iş
küçük bir pilot olsun: `configs/benchmarks/shared_pilot.yaml` (30 soru, k = 2). Bütün akışın gerçek bir
koşuyla anlatımı: [Adım 1: eğitim verisi](../walkthrough/01_egitim_verisi.md).

Kendi dört modelini tek komutla gönder (her model için `submit_member.sh` çağırır):

```bash
bash cluster/tosun/submit_my_models.sh configs/benchmarks/shared_v1.yaml ibrahimb 4
```

Tek bir modeli yeniden göndermek için: `bash cluster/tosun/submit_member.sh configs/benchmarks/shared_v1.yaml smollm3-3b ibrahimb 4`.
Wrapper modelin `gpus` değerine göre GPU ister (24B, 27B ve 32B için 2; Llama ve Gemma için önce HF'de lisansı kabul edin), generation'dan
sonra CPU ground-truth ve GPU ArmoRM skorlamasını paralel, ardından paketlemeyi submit
eder; router training submit etmez. `LLMR_BATCH_SIZE` vLLM'e tek seferde giden istek
sayısıdır (varsayılan 256). Elle çalıştırmak isterseniz her shard için aynı dört argüman:

```bash
CFG=configs/benchmarks/shared_v1.yaml; MODEL=smollm3-3b; MEMBER=ibrahimb
llmr generate "$CFG" --model "$MODEL" --member "$MEMBER" --num-shards 4 --shard-index 0 --max-retries 2 --batch-size 256
llmr score "$CFG" --model "$MODEL" --member "$MEMBER" --num-shards 4 --shard-index 0 --scorer ground_truth
llmr score "$CFG" --model "$MODEL" --member "$MEMBER" --num-shards 4 --shard-index 0 --scorer armorm
llmr score "$CFG" --model "$MODEL" --member "$MEMBER" --num-shards 4 --shard-index 0 --scorer bartscore
llmr package-contribution "$CFG" --model "$MODEL" --member "$MEMBER" --num-shards 4 --shard-index 0
```

Aynı komutu tekrar çalıştırmak resume'dur; başarılı sample/score yeniden üretilmez.
Zaman aşımında başarısız stage'i aynı argümanlarla yeniden submit edin; tamamlanmış
paketleri yeniden paketlemeyin. Config değiştirip eski işe resume etmeyin: model/sample
ayarını değiştirmek yeni benchmark sürümü demektir.

## 4. Paketleri teslim et ve birleştir

Damla ve İbrahim B, `$LLMR_DATA_ROOT/benchmarks/shared_v1/exports/` altındaki tamamlanmış paket
dizinlerini koordinatörün aynı `exports/` dizinine rsync ile taşır. Paket adları üye adıyla başladığı için
(`damla-...`, `ibrahimb-...`) çakışmaz. Koordinatör `ls .../exports/` ile 12 modelin hepsinin geldiğini görür. Eksik kopya veya değişmiş dosya hash kontrolünde
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

[Router sözleşmesi](router_contract.md) ve [paper şablonu](../papers/_template.md) ile
her makale için önce label → architecture → loss → train → inference protokolünü
çıkarın; paper/released code farklarını kaydedin. Planlanan router'lar: Hybrid LLM,
RouteLLM, GraphRouter, MixLLM, UniRoute, BEST-Route, PILOT. Her birinin
release'ten ne aldığı router sözleşmesindeki tabloda.

Mevcut bir router'ı ortak veri üzerinde ileride çalıştırırken experiment config'inde
benchmark olarak shared YAML'ı seçin. `quality_metric: ground_truth`, `armorm` ve `bartscore` için
ayrı deneyler yapın; label metriğini paper gerekçesi olmadan değiştirmeyin. Eşik ve
checkpoint val'den, son rapor aynı test sorularından seçilir. Paper-native metrikleri
ortak quality/cost raporunun yanına ekleyin. En az üç seed ile raporlayın.
