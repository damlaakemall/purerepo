# Adım 1: Router'lar için eğitim verisi üretmek

> Bir router "bu soruyu hangi model cevaplasın?" kararını öğrenir. Bunu öğrenmesi için elimizde bir tablo
> olmalı: **her soru için, her modelin cevabı ne kadar iyi ve ne kadar pahalı.** Bu adım o tabloyu üretir.
> Soruları seçeriz, 12 modele cevaplatırız (Tosun GPU'larında), her cevabı puanlar ve maliyetini sayarız.
> Sonraki adımda router'lar bu tablonun train kısmıyla eğitilip test kısmında karşılaştırılacak.
>
> **Durum (26 Ekim 2026):** İbrahim G’nin v6 katkısı tamamlandı: 306080 cevap ve 16 paket
> doğrulandı; ArmoRM taşması yok. GT’de iki izinli matematik süre aşımı vardır.
> Damla’nın Qwen2.5-Coder ve Phi-4 pilotları gönderildi; iki Gemma modeli erişim bekliyor.
> Önceki GT2 katkısı korunur. 12 modellik ortak release henüz tamamlanmadı.

## shared_v6: ArmoRM sınırı üretime uygulanır — 23 Ekim 2026

Kullanıcı “ArmoRM sınırı”nın üretimde de geçerli olmasını netleştirdi. Yeni v6,
12 modelin tamamında **ArmoRM tokenizer’ı ve chat template’iyle soru + cevap
+ şablon toplamını 8192 token içinde** tutar. Üreten modelin tokenizer’ı farklı
olduğu için onun çıktı tavanını 8192 yapmak tek başına yeterli değildir.

Üretim sırasında aynı vLLM engine’in `add_request / step / abort_request`
akışı kullanılır; her kısmi cevap ArmoRM tokenizer’ıyla ölçülür. Engine core
`VLLM_ENABLE_V1_MULTIPROCESSING=0` ve `async_scheduling=False` ile eşzamanlı
adımlanır. Böylece CPU ölçüm yaparken arkada görünmeyen ek token üretilmez.
Tokenizer’dan `return_dict=False` ile gerçek düz ID listesi istenir; bir sözlüğün
anahtar sayısı token sayısı olarak kabul edilmez. Sınırdan
1024 token önce durdurma tetiklenir. Bu pay, üretici token’ın farklı tokenizer’da
birden çok token’a açılmasına ve engine adımına alan bırakır. Üretici vocabulary’sindeki
en uzun tek-token byte açılımı bu paya göre denetlenir; gerçek bir adım payı
buna rağmen aşarsa üretim hata verir. Son nihai metin ayrıca ArmoRM ile tekrar
ölçülür: 8192 üstü cevap başarılı veri olarak kaydedilemez. Bu doğrulama, pilot
ve tam paket denetiminde de zorunludur; yaklaşık token oranı garanti diye sunulmaz.

HF backend aynı kontrolü `StoppingCriteria` ile yapar. Her iki backend’de de
sadece durma kararı değişir: ek system prompt, logits değişikliği veya sonradan
metin kırpma yoktur. Ham üretilen token ID’leri sayılır; durdurma nedeni
`armorm_budget`, metadata `armorm_generation_budget.tokens/limit/reserve_tokens`
olarak kaydedilir. EOS önce gelirse normal `stop` geçerlidir. Üreticinin kendi
çıktı tavanı da bütün modellerde 8192’dir; model context’i ayrıca prompt’a yer ayırır.
Soru tek başına güvenli üretim alanı bırakmıyorsa açık hata verilir; soru kırpılmaz.

Bu yeni üretim protokolüdür: `shared_v6.yaml`, `shared_v6_pilot.yaml`,
`team_v6.yaml` ayrı sürüm ve veri kökü kullanır. Tamamlanmış v5/GT2 cevapları ve
puanları korunur; yeni bütçeye uydurmak için kırpılmaz veya yalnız taşan cevaplar
seçilerek tekrar üretilmez. Tutarlı v6 veri için İbrahim G’nin dört modeli yeni
pilot sonrasında yeni sürümde üretilir. 24 Ekim’de kullanıcı, İbrahim G’nin katkısı
tamamen doğrulandıktan sonra Damla’nın atanmış dört modelinin de bu hesapta,
`member=damla` katkı yollarında toplanmasını istedi. İbrahim B’nin modelleri bu yetkinin dışındadır.
Fiyatlar, model revision’ları, kaynaklar, split kuralı, non-thinking Qwen ve
GT’nin600s/eksik math-timeout politikası değişmedi. ArmoRM taşması v6’da
`on_context_overflow: error` olduğu için sessiz eksik skora dönüşmez.

```bash
llmr build-queries configs/benchmarks/shared_v6_pilot.yaml
bash cluster/tosun/submit_my_models.sh configs/benchmarks/shared_v6_pilot.yaml ibrahimg 1
# Dört pilotta bütün generation-budget metadata ve üç scoring kanalı doğrulanmadan tam üretime geçilmez.
```

### V6 hazırlığı ve gönderilen pilotlar

PR8, son kaynak commit’i `c8e155f833cdd222bf5c886c4251ca4608bb8ae0` üzerinde
CI sonrası main `3978bbc039e9876597a31f50624fea2185e50ea9` olarak birleşti.
Çalışan Tosun checkout’u `~/purerepo-v6-live`; Python kaynakları iş başladıktan
sonra pull/reset edilmez. Ön kontrolde 7652 sorunun tamamı ArmoRM’nin pinned
tokenizer’ıyla ölçüldü: **en uzun soru + boş assistant şablonu 2849 token**,
limit 8192 ve güvenlik payı 1024. Hiçbir soru kırpılmadı veya elenmedi.
30 soruluk v6 pilot seti donduruldu.

| model | generation | GT / ArmoRM / BART | package |
|---|---:|---:|---:|
| Qwen3-32B | 4201225 | 4201226 / 4201227 / 4201228 | 4201229 |
| Qwen3-14B | 4201230 | 4201231 / 4201232 / 4201233 | 4201234 |
| Mistral-Nemo-12B | 4201235 | 4201236 / 4201237 / 4201238 | 4201239 |
| Llama-3.2-3B | 4201240 | 4201241 / 4201242 / 4201243 | 4201244 |

Dört pakete bağımlı **4201245** doğrulaması, `logs/check-v6-pilot.py` ile
hash’leri, her modelde beklenen 60 anahtarı, gerçek token/context bütçelerini,
non-thinking ayarını, nihai cevapları ve üç kanalı denetler. ArmoRM’nin gerçek
scorer token sayısı, üretim metadata’sındaki ölçümle birebir eşleşmelidir;
ArmoRM’de taşma veya eksik skor bulunursa tam üretim açılmaz. İzinli GT
math-timeout durumu ayrı kaydedilir. `length` / `armorm_budget` bitişleri model
sonucu olarak tutulur; iyi cevap gelene kadar seçerek tekrar üretim yapılmaz.

Yalnız dört pilot doğrulanırsa `logs/v6-validate.sh`, `logs/v6-full.sh` işini
bir kez gönderir. Bu iş tam v6 soru setini compute node’da build eder, 7652 soru
ve aynı `664b6fe9a7e07132f88ef219d1315415010abc4924e9c565539489ccc81a5e0a`
fingerprintini doğrular, sonra yalnız İbrahim G’nin dört modelini dört shard
ile gönderir. İş numarası `logs/v6-full-job.txt`, gönderimler
`logs/v6-full-submissions.txt`, pilot raporu `logs/v6-pilot-report.json` içine
kaydedilir. Mevcut dosya/kuyruk kontrol edilmeden bu işler yeniden gönderilmez.

### Doğrulanan V6 pilotu ve tam toplamaya geçiş

**4201245** doğrulaması geçti; `logs/v6-pilot-report.json` gerçek dört paket
sonucunu kaydeder. Her modelde 30 soru × 2 örnek = 60 cevap vardır. Dosya
hash’leri, beklenen anahtarlar, model/revision, non-thinking Qwen ayarı,
nihai cevaplar ve gerçek token/context bütçeleri denetlendi. ArmoRM scorer’ının
sayımı ve şablon hash’i üretim metadata’sıyla her cevapta eşleşti.

| model | cevap | stop / length / armorm_budget | en büyük ArmoRM chat token sayısı | GT / ArmoRM / BART |
|---|---:|---:|---:|---:|
| Qwen3-32B | 60 | 60 / 0 / 0 | 4254 | 60 / 60 / 60 |
| Qwen3-14B | 60 | 60 / 0 / 0 | 4239 | 60 / 60 / 60 |
| Mistral-Nemo-12B | 60 | 58 / 1 / 1 | 7168 | 60 / 60 / 60 |
| Llama-3.2-3B | 60 | 59 / 0 / 1 | 7168 | 60 / 60 / 60 |

Toplam **240 cevap ve her kanalda 240 skor** vardır; izinli eksik skor ve ArmoRM
taşması yoktur. İki `armorm_budget` bitişi güvenlik payında durduruldu; Nemo’nun
bir `length` bitişi de model sonucu olarak korundu. Hiçbiri seçilerek yeniden
üretilmedi. Pilotun toplam tahmini API maliyeti **0.027635962 USD**’dir; gerçek
prompt ve toplam completion token’larına dondurulmuş tarifeler uygulanmıştır.
Bu bir GPU faturası değildir.

Başarılı doğrulama `logs/v6-full.sh` işini **4201274** olarak bir kez gönderdi.
Tam set **7652 soru** (train 4633, val 753, test 2266) ve yukarıdaki aynı dataset
fingerprintiyle kuruldu. Her model için k=10 ile **76520**, dört model için
**306080 cevap** planlanır. Aşağıdaki bütün işler 0–3 shard array’leridir:

| model | generation | GT / ArmoRM / BART | package |
|---|---:|---:|---:|
| Qwen3-32B | 4201275 | 4201276 / 4201277 / 4201278 | 4201279 |
| Qwen3-14B | 4201281 | 4201282 / 4201283 / 4201284 | 4201285 |
| Mistral-Nemo-12B | 4201286 | 4201287 / 4201288 / 4201289 | 4201290 |
| Llama-3.2-3B | 4201291 | 4201292 / 4201293 / 4201294 | 4201295 |

Çalışan checkout’ta durumu görmek için kullanılan komutlar:

```bash
cd ~/purerepo-v6-live
cat logs/v6-pilot-report.json
cat logs/v6-full-job.txt logs/v6-full-submissions.txt
squeue -u "$USER" -o '%.22i %.12T %.12M %.35R'
```

İşler zaten gönderilmiştir; bu kayıt yeni bir `submit_my_models.sh` çağrısı
gerektirmez. Bağımlı puanlama işlerinin beklemesi tek başına hata değildir.

### İbrahim G’nin tamamlanan V6 katkısı — 26 Ekim 2026

**16 paketin tamamı ve 306080 cevap doğrulandı.** Compute node üzerinde
`logs/check-v6-full.py` manifest/dosya hash’lerini, her shard’ın beklenen tam
anahtar kümesini, response/score/evaluator hash’lerini, soru metnini, seed’i,
model/revision’ı, native token ayrımını, context ve ArmoRM bütçesini denetledi.
`logs/audit-v6-origin.py` ayrıca bütün modellerde 76520 benzersiz anahtarı ve
üretim/puanlama kaynak hash’lerinin eşitliğini kontrol etti. Canlı Python
`c8e155f833cdd222bf5c886c4251ca4608bb8ae0` olarak kaldı; kaynak hash’i
`1f7b790d0164f009aa804d7f6877460c8bddf40bd227a2c29fe0f3b0c602611f`.

| model | cevap | stop / length / armorm_budget | GT / ArmoRM / BART | en büyük ArmoRM token |
|---|---:|---:|---:|---:|
| Qwen3-32B | 76520 | 76441 / 31 / 48 | 76520 / 76520 / 76520 | 7169 |
| Qwen3-14B | 76520 | 76410 / 42 / 68 | 76520 / 76520 / 76520 | 7169 |
| Mistral-Nemo-12B | 76520 | 76369 / 34 / 117 | 76518 / 76520 / 76520 | 7169 |
| Llama-3.2-3B | 76520 | 75564 / 0 / 956 | 76520 / 76520 / 76520 | 7170 |

ArmoRM ve BARTScore’un her birinde **306080 skor**, GT’de **306078 skor ve
iki izinli eksik** vardır. İki eksik de Nemo’nun `math_timeout`, `seconds=600`
kayıtlarıdır; sıfır veya başka kanaldan skor yazılmadı. Manifest kapsamı ve
izinli eksik nedenleri doğrulandı. ArmoRM taşması ve nihai-cevap-yokluğu eksikleri
yoktur. Her ArmoRM skorunun gerçek token sayısı ve şablon hash’i üretim
metadata’sıyla eşleşir. 1189 `armorm_budget` ve 107 `length` bitişi model
sonucu olarak tutuldu; seçerek yeniden üretim yapılmadı.

| model | prompt token | toplam completion token | tahmini API USD |
|---|---:|---:|---:|
| Qwen3-32B | 16810000 | 38025682 | 11.99199096 |
| Qwen3-14B | 16810000 | 37162918 | 9.85684196 |
| Mistral-Nemo-12B | 16760360 | 21703800 | 0.95280048 |
| Llama-3.2-3B | 18147370 | 32218638 | 11.53951904 |
| toplam | 68527730 | 129111038 | 34.34115244 |

Maliyet, dondurulmuş 2026-10-19 tarifeleri × gerçek prompt ve düşünme dahil
toplam completion token’ıdır; GPU faturası veya ücretli inference harcaması değildir.
Tam raporlar Tosun’da `logs/v6-full-verified-ibrahimg.json`,
`logs/v6-full-summary-ibrahimg.json` ve `logs/v6-origin-audit.json` içinde saklanır.
Paket kökü `~/llmr_data/benchmarks/shared_v6/exports/ibrahimg-*`;
önceki v5/GT2 cevapları değiştirilmedi.

25 Ekim’de Qwen14’ün BART shard 0/1/3 ve Nemo’nun dört GT işi kısmi skorlarla
kesilmişti; scheduler geçmişi erişilemediği için kesin kesilme nedeni saptanamadı.
Yalnız eksik skorlar **4251705** (BART array 0,1,3) ve **4251706** (GT array 0–3)
ile devam ettirildi. Tamamlanmış kayıtlar ve izinli süre aşımları korundu,
mevcut downstream bağımlılıkları düzeltildi. Yeni cevap üretimi veya protokol
değişikliği yapılmadı. Son doğrulama bu devam ettirilen kayıtları da kapsar.

```bash
# Aynı frozen checkout ve aktif llmr ortamında, compute node üzerinde:
python logs/check-v6-full.py qwen3-32b qwen3-14b mistral-nemo-12b llama-3.2-3b
python logs/audit-v6-origin.py
```

### Damla’nın katkısına geçiş

Kullanıcının 24 Ekim’de kesinleştirdiği sıra uygulandı: önce İbrahim G’nin tam
katkısı doğrulandı, sonra Damla’nın pilotuna geçildi. Atama `team_v3.yaml` ile
aynıdır: `qwen2.5-coder-32b`, `phi-4`, `gemma-3-12b`, `gemma-3-4b`.
Yeni kaynak/protokol oluşturulmadı; aynı frozen v6 soru seti, fiyatlar ve
model revision’ları kullanılır. Pilot k=2 ve 30 sorudur; tam katkı k=10 ve dört
shard ile ancak **Damla’nın dört pilotu da** doğrulanırsa başlar.

Gönderim işi **4252034**, erişilebilen iki modelin pilot zincirini gönderdi:

| model | generation | GT / ArmoRM / BART | package |
|---|---:|---:|---:|
| Qwen2.5-Coder-32B | 4252035 | 4252036 / 4252037 / 4252038 | 4252039 |
| Phi-4 | 4252040 | 4252041 / 4252042 / 4252043 | 4252044 |

Gemma-3-12B ve Gemma-3-4B’nin pinned `tokenizer_config.json` erişim denetimi
Hugging Face’te **403 GatedRepoError** döndürdü. Bu iki iş henüz gönderilmedi;
erişim açılınca aynı revision ile devam edilecek. Token görüntülenmedi ve model
değiştirilmedi. `logs/v6-damla-pilot-<model>.txt` gerçek gönderimleri,
`logs/check-v6-damla-pilot.py` dört pilotun ortak doğrulamasını tutar.
Henüz Damla’nın pilot sonucu, tam katkısı veya 12 modellik release tamamlandı denemez.

Üretimdeki engine arayüzü Tosun’un kurulu vLLM kaynağından denetlendi;
[LLM engine API](https://docs.vllm.ai/en/latest/api/vllm/v1/engine/llm_engine/)
ve [HF durdurma kriteri](https://huggingface.co/docs/transformers/main/internal/generation_utils#transformers.StoppingCriteria)
aynı mekanizmaları açıklar. Yukarıdaki sonuçlar yalnız İbrahim G’nin dört
modelinin pilotuna aittir; diğer sekiz modelin pilotu veya ortak release sonucu değildir.

## shared_v5_gt2: kesinleşen matematik süre sınırı — 22 Ekim 2026

Bütün 12 model için **OlympiadBench cevap kontrolü başına 600 saniye** duvar
saati sınırı kararlaştırıldı. Daha erken biten karşılaştırma aynı skoru alır;
süreyi aşan karşılaştırma `score=null`, `reason=math_timeout`, `seconds=600`
olarak kaydedilir. Kontrolün bitmemesi cevabın yanlış olduğunu kanıtlamaz;
GT’ye sıfır yazılmaz ve başka kalite kanalından doldurulmaz. MBPP’nin mevcut
5 saniyelik kod testi sınırı ve süre aşımını yanlış cevap sayması değişmedi.
ArmoRM’nin 8192 context sınırı/eksik skoru ve toplam token maliyeti de değişmedi.

Karar anındaki gerçek v5 durumu: dört modelin toplam **306080 cevabı** üretildi.
Üç modelin dörder paketinde dosya hash’leri, anahtarlar, token/context bütçeleri
ve kanallar bağımsız doğrulandı:

| model | cevap | stop / length | GT / ArmoRM / BART | izinli ArmoRM eksik |
|---|---:|---:|---:|---:|
| Qwen3-32B | 76520 | 76464 / 56 | 76520 / 76454 / 76520 | 66 |
| Qwen3-14B | 76520 | 76446 / 74 | 76520 / 76422 / 76520 | 98 |
| Mistral-Nemo-12B | 76520 | 76367 / 153 | 76520 / 76362 / 76520 | 158 |

Bunlar **eski v5** sayılarıdır; yeni GT2 sonuçları olarak sunulmaz. Llama’nın
GT shard 0’ında 19390 cevaptan 18256 kayıt yazıldıktan sonra ilerleme durdu;
1134 kayıt kaldı. Takılan `4121908_0` (somut job `4121990`) durduruldu;
üretilmiş cevaplar ve önceki skorlar silinmedi. O anda diğer Llama puanlama aşamaları
bu işin başarısına bağımlı olduğundan tamamlanmamıştı.

Bu kararın nedeni Llama’nın `olympiadbench/673dbb00cc49c41098c5c9e9`, sample 6
cevabındaki binlerce tekrarın SymPy sayısal karşılaştırmasını 20 saatten uzun
kilitlemesidir. Üretim bitmişti; takılan model değil puanlayıcıydı. Hesaplamayı
ayrı süreçte yürütmek, referans kodun geniş `except` bloklarının süre sınırını
yutmasını önler. Süre sonunda alt süreç sonlandırılır ve temizlenir.

Eski `shared_v5` configleri ve sonuçları donmuş kalır. Yeni configler
`shared_v5_gt2.yaml` / `shared_v5_gt2_pilot.yaml` ayrı veri kökleri kullanır.
GT bütün modeller için yeniden hesaplanır; yalnız takılan Llama örneğine özel
kural uygulanmaz. Aynı donmuş cevaplar, sorular, splitler, modeller, üretim
ayarları ve fiyatlar kullanılır; yeniden cevap üretilmez. Aktarım özgün
`run.json` ortamını ve cevap dosyasının hash’ini `generation_origin` içinde
saklar. Kaynak Python veya kaynak skor dosyaları değiştirilmez.

Yeni kod ayrı Tosun checkout’unda kurulunca, compute node’da örnek aktarım:

```bash
# Önce kaynak shard'ın çalışan yazarı olmadığını doğrula; kilidi körlemesine silme.
python cluster/tosun/rescore_v5_gt2.py llama-3.2-3b ibrahimg --num-shards 4 --shard-index 0
llmr score configs/benchmarks/shared_v5_gt2.yaml --model llama-3.2-3b --member ibrahimg --num-shards 4 --shard-index 0 --scorer ground_truth
```

Aktarım aynı soruları/splitleri ve aynı generation spec’ini denetler, tam ve
benzersiz başarılı cevapları hash’leriyle kontrol eder, GT’yi kopyalamaz.
Değişmeyen ArmoRM/BARTScore kayıtları varsa korunup kaldığı yerden sürdürülür.
İş gönderimi yalnız üyeye atanmış dört model içindir; diğer üyeler aynı kararı
kendi modellerine uygular. Sırf tekrar için `submit_my_models.sh` çalıştırılmaz.
Bu bölüm ortak karar ve prosedürdür. İbrahim G’nin tamamlanan GT2 katkısının
gerçek sayıları ve kanal kapsamı aşağıda ayrıca raporlanır; diğer üyelerin
modelleri tamamlanmış sayılmaz.

### İbrahim G’nin tamamlanan GT2 katkısı — 23 Ekim 2026

**306080/306080 cevap, 16/16 paket doğrulandı.** Aynı 7652 soru (train 4633,
val 753, test 2266), dört model ve soru başına k=10 kullanıldı. Dataset kimliği
`664b6fe9a7e07132f88ef219d1315415010abc4924e9c565539489ccc81a5e0a`.
Bu, İbrahim G’nin dört modellik katkısıdır; diğer üyelerin sekiz modeli ve
12 modellik ortak frozen release henüz tamamlanmış sayılmaz. Router eğitimi başlatılmadı.

| model | cevap | stop / length | GT skor / eksik | ArmoRM skor / eksik | BARTScore | tahmini API USD |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-32B | 76520 | 76464 / 56 | 76520 / 0 | 76454 / 66 | 76520 | 12.21868848 |
| Qwen3-14B | 76520 | 76446 / 74 | 76520 / 0 | 76422 / 98 | 76520 | 10.00016360 |
| Mistral-Nemo-12B | 76520 | 76367 / 153 | 76518 / 2 | 76362 / 158 | 76520 | 0.99826545 |
| Llama-3.2-3B | 76520 | 75793 / 727 | 76519 / 1 | 75544 / 976 | 76520 | 14.32077455 |

GT kapsamı **306077/306080 (%99.9990)**, ArmoRM kapsamı
**304782/306080 (%99.5759)**, BARTScore **306080/306080 (%100)**.
Üç GT eksikliği `math_timeout` (Nemo 2, Llama 1; sınır 600 saniye),
1298 ArmoRM eksikliği ölçülen `context_overflow` durumudur. Hiçbiri sıfırla
veya başka kanalla doldurulmadı. Başka eksik anahtar veya başarısız scorer kaydı
bulunmadı. Toplam 1010 `length` sonucu tutuldu; örnekler seçilerek yeniden
üretilmedi. Doğal `stop`, cevabın doğru veya tekrarsız olduğu anlamına gelmez.

Toplam **68527730 prompt token**, **140515680 completion token** ve
**37.53789208 estimated API USD** hesaplandı. Bu Tosun gideri veya provider
faturası değildir; aynı donmuş 2026-10-19 input/output tarifeleri kullanıldı.
Bütün cevaplarda token ayrımı `generated_ids` olarak çözüldü; gerçek
reasoning+answer toplamı completion sayısına eşit ve context/etkin çıktı
bütçesi içindeydi. Ham cevap, model/revision ve Qwen non-thinking ayarı doğrulandı.

İş zinciri: aktarım **4178857**, GT **4178858** (array 0–15), yalnız Llama için
ArmoRM **4178859** ve BARTScore **4178860** (0–3), paket **4178861** (0–15).
GT/package indeksinde `index//4` sırasıyla Qwen32, Qwen14, Nemo, Llama;
`index%4` shard’dır. Diğer üç modelin değişmeyen reward kayıtları korunmuştur.
Tüm modellerin GT’si tek yeni politika ile yeniden hesaplandı.

Son denetim CPU compute node üzerinde çalıştırıldı:

```bash
cd ~/purerepo-v5-gt2
export LLMR_REPO="$PWD" PYTHONPATH="$PWD"
source cluster/tosun/config.sh
llmr_activate
srun --input=none --partition=cuda --qos=cuda --mem=8G --cpus-per-task=2 --time=00:20:00 python logs/validate-gt2.py
# ALL_VERIFIED 306080
```

Denetim, 16 manifestin bütün dosya hash’lerini; her shard’ın beklenen soru/model/
örnek anahtarlarını; cevap ve skor hash’lerini; evaluator kimliği ve kanalların
tam kayıtlarını; yalnız izinli eksik nedenlerini; context/token/model/pricing
bilgilerini kontrol etti. `generation_origin` içindeki özgün run/ortam ayrıca
eski v5 `run.json` ile birebir karşılaştırıldı. Cevap dosyalarının byte hash’leri
hem yeni paketle hem özgün v5 kaynakla aynıdır: yeniden üretim veya ham veri
üzerine yazma yoktur.

Paketler Tosun’da `~/llmr_data/benchmarks/shared_v5_gt2/exports/ibrahimg-<model>-<shard>-of-00004`
altındadır. Rapor `~/purerepo-v5-gt2/logs/v5-gt2-final-report.json`, denetim
çıktısı `logs/v5-gt2-final-validation-compute.txt` içinde saklanır. Ham cevap,
JSONL veya ağırlıklar GitHub’a eklenmez. Bu katkı tamamlandığı için zamanlanmış
takip kapatılır; ortak 12 modellik assembly diğer katkıları bekler.

## shared_v5: bütün havuza uygulanan bütçe ve nihai cevap politikası

**21 Ekim 2026 — kullanıcının netleştirdiği ortak karar.** Uzun cevap ve thinking
politikası yalnız İbrahim G’nin dört modeline değil, 12 modelin tamamına uygulanır.
Yeni configler `shared_v5.yaml`, `shared_v5_pilot.yaml` ve `model_pools/team_v5.yaml`;
atamalar aynı `cluster/tosun/team_v3.yaml` dosyasındadır. V3/v4 verisi ve configleri
korunur. Tosun’da yalnız İbrahim G’nin dört modeli çalıştırılır; diğer üyeler aynı
ortak configleri kendi hesaplarında kullanır.

| modeller | çıktı tavanı | toplam context | düşünme kontrolü |
|---|---:|---:|---|
| Qwen3-14B / Qwen3-32B | 16384 | 32768 | pinned tokenizer şablonunda `enable_thinking: false` |
| Phi-4 | 12288 | 16384 | ayrı bir thinking anahtarı yok; kendi instruct şablonu |
| kalan 9 instruct model | 16384 | 32768 | doğrulanmış ayrı anahtar yok; kendi instruct şablonları |

Phi-4’ün gerçek toplam context’i 16384’tür; yalnız config sayılarını büyütmek modelin
eğitim sınırını büyütmez. `context_policy: cap_output`, tam prompt’u koruyarak
örnek bazında `min(çıktı_tavanı, toplam_context - prompt_tokens)` uygular. Etkin
çıktı bütçesi ve context nedeniyle azalıp azalmadığı metadata’da kaydedilir.
Maliyet yine **gerçek** prompt/output token sayıları × dondurulmuş 2026-10-19
input/output tarifesidir; bütçe tavanları faturalandırılmış token sayısı değildir.
Gemma/Llama gated modellerinin indirme erişimini ilgili üyeler kendi hesaplarında
sağlar; İbrahim’in hesabıyla diğer üyelerin modelleri çalıştırılmadı.

**Thinking ayrımı.** Qwen3 iki modu yerel şablon üzerinden destekler. Diğer
modellerde sıradan bir uzun açıklamayı gizli düşünce saymak veya etkisiz bir
`enable_thinking: false` alanı eklemek doğru değildir. `<think>`, `<thinking>`,
`<reasoning>` gibi tanınan bloklar puanlama görünümünden çıkarılır; ham cevap
saklanır. V5’te `text_policy: final_only`, yalnız düşünce içeren cevapları
ArmoRM/BARTScore için ham düşünceye geri döndürmez: GT=0, reward skorları eksik
kalır. Bu, referansın ham metne geri dönen reward davranışından belgelenmiş bir
ayrımdır ve kullanıcının yalnız nihai cevaba bakma isteğini uygular.

Token ayrımı yeniden tokenize edilmiş metin parçası tahmini değildir. Üretilen
gerçek ID dizisindeki tanınan açma/kapama etiketleri üzerinden `reasoning_tokens`
ve `answer_tokens` kaydedilir; etiket token’ları reasoning’e dahildir. Prompt’un
sonunda zaten açılmış düşünce etiketi varsa prompt token’ı olarak kalır, output’a
iki kez sayılmaz. Çözülemeyen etiket yazımı/encoding için iki sayı `null`,
`token_partition: unavailable` olur; toplam `completion_tokens` yine gerçektir.
Maliyete düşünme + cevap toplamı girer. Sadece cevabı fiyatlamak yerel üretim
maliyetini eksik sayardı.

**ArmoRM sınırı.** V4’te bir Llama cevabının soru+cevap girdisi 8309 token oldu ve
sabitlenen ArmoRM’nin 8192 context sınırını aştı. Referans da bu durumu hata
olarak bırakıyor; kırpmıyor. Kullanıcı açıkça “Referans gibi taşan ArmoRM skorunu
eksik bırak” seçeneğini seçti. V5’in `on_context_overflow: missing` politikası:

- Tam cevabı ve maliyeti korur; GT ve BARTScore kendi tanımlarıyla çalışır.
- Yalnız ölçülen ArmoRM context taşmasını eksik skor olarak kaydeder; token
  uzunluğu/sınırı ve neden kaybolmaz. View’de değer NaN, sıfır değildir.
- Aynı bilinen eksik skoru her resume’da yeniden denemez.
- Eksik üretim, GPU/model yükleme hatası ve başka scorer hataları hâlâ başarısızdır.
  `allow_partial: false` korunur; genel hata affı açılmaz.
- Release manifesti `allowed_missing_scores` ile kanal başına bu eksikleri sayar.
  Analizlerde ArmoRM kapsama oranı ayrıca raporlanmalı; eksiklerin uzun cevaplarda
  yoğunlaşması nedeniyle ArmoRM ortalaması tüm örneklerin ortalaması değildir.

**110 CPU testi geçti** (75.95 s); uçtan uca test-double koleksiyonunda 30 soru,
120 cevap ve doğrulanmış release üretildi. Testler, eksik ArmoRM ile tam GT/maliyet korunmasını; yalnız düşünceyi reward
puanlayıcısına göndermemeyi; diğer hataların gizlenmemesini ve token ID ayrımını
kontrol eder. Gerçek v5 pilot sonucu aşağıda kaydedilmiştir. V4 tek örnek Llama
16k bütçe tanısı 965 token’da EOS verdi; tekli batch/çalışma koşulları değiştiği
ve aynı seed tek başına birebir GPU determinism sağlamadığı için bu sonuç
“bütçe artışı kesin çözdü” kanıtı veya seçilmiş replacement sample değildir.
Tanı cevabı hiçbir benchmark paketine katılmadı.

```bash
# Yeni ve izole v5 checkout’ta, compute node build adımı:
llmr build-queries configs/benchmarks/shared_v5_pilot.yaml
bash cluster/tosun/submit_my_models.sh configs/benchmarks/shared_v5_pilot.yaml ibrahimg 1
python cluster/tosun/inspect_shard.py configs/benchmarks/shared_v5_pilot.yaml qwen3-32b ibrahimg
# Her modelde tam üretim + GT/BART tamamlanması, ArmoRM başarı/izinli eksik
# ayrımı ve kesilme oranı kontrol edilir. Başarılı örnek seçilerek tekrar üretilmez.
```

### İbrahim G’nin doğrulanan v5 pilotu ve tam toplama geçişi — 21 Ekim 2026

Aynı 30 donmuş soru dört modelde k=2 ile çalıştırıldı: **240/240 üretim başarılı**,
GT **240/240**, BARTScore **240/240**, ArmoRM **238/240**. İki ArmoRM eksikliği
kullanıcının seçtiği `context_overflow` politikasıdır; başka puanlama hatası veya
boş nihai cevap yoktur. ArmoRM kapsamı **%99,17**; bu iki skora sıfır yazılmadı,
cevaplar ve token bazlı tahmini maliyetleri veri içinde tutuldu.

| model | cevap | stop / length | en uzun çıktı | GT / ArmoRM / BART | tahmini API USD |
|---|---:|---:|---:|---:|---:|
| Qwen3-32B | 60 | 60 / 0 | 3519 | 60 / 60 / 60 | 0.00932252 |
| Qwen3-14B | 60 | 60 / 0 | 4662 | 60 / 60 / 60 | 0.00748058 |
| Mistral-Nemo-12B | 60 | 59 / 1 | 16384 | 60 / 59 / 60 | 0.00127309 |
| Llama-3.2-3B | 60 | 60 / 0 | 11827 | 60 / 59 / 60 | 0.01093005 |

Toplam **118111 output token** ve **0.02900624 estimated API USD** ölçüldü;
bu değer Tosun GPU gideri veya provider faturası değildir. Model başına ham
hesap, `logs/v5-pilot-report.json` içinde daha yüksek hassasiyetle saklanır.
Mistral’ın bir cevabı 16384 tavanında kesildi; bu cevap seçilerek tekrar
üretilmedi veya paket dışına atılmadı. Llama’daki uzun cevap doğal stop verdi
fakat açıklama uzayıp bozuldu; doğal stop doğruluk/kalite garantisi değildir.
İki durumda da ArmoRM sınırı aşan tam metin kırpılmadan korundu.

Dört paketin manifestleri/dosya hash’leri ve donmuş dataset kimliği ayrıca
doğrulandı. Otomatik kontrol, her modelde 60 cevap ve aynı anahtarlara bağlı
üç kanalın tam kaydını, yalnız izinli eksikleri, pinned revision’ları, gerçek
context/etkin çıktı bütçelerini ve Qwen3 non-thinking ayarını denetledi.
Token ayrımı 240 örneğin tamamında generated_ids olarak çözüldü; etiketli düşünce token sayısı hepsinde sıfırdı. Bu, normal cevabın çözüm açıklaması içermediği anlamına gelmez. Ham cevaplar saklanır.

Pilot generation / package iş numaraları sırasıyla Qwen32 **4115805 / 4115809**,
Qwen14 **4115810 / 4115814**, Nemo **4115824 / 4115828**, Llama
**4115829 / 4115833**. Dört paket sonrasındaki **4115834** doğrulama işi başarılı
olunca **4121890** tam soru setini compute node’da hazırladı ve tam üretim
zincirlerini gönderdi. Donmuş tam set **7652 soru** içeriyor: train **4633**,
val **753**, test **2266**. Dataset kimliği
`664b6fe9a7e07132f88ef219d1315415010abc4924e9c565539489ccc81a5e0a`.
Kaynak sayıları: Alpaca 1000, BBH 2160, GSM8K 1319, MBPP 499,
MMLU-Pro 1500, Multi-News 500, OlympiadBench 674.
Her soru k=10 ile yanıtlanacak: model başına **76520**, İbrahim G’nin dört modeli
için toplam **306080 planlanan cevap**. Bu sayılar tamamlanmış üretim sayısı değildir.
Her model dört shard’a ayrıldı; ilk kontrolde Qwen32’nin bir ve Llama’nın iki
shard’ı çalışıyor, kalanlar GPU kuyruğunda bekliyordu.

| model | generation array | GT | ArmoRM | BARTScore | package |
|---|---:|---:|---:|---:|---:|
| Qwen3-32B | 4121891 | 4121892 | 4121893 | 4121894 | 4121895 |
| Qwen3-14B | 4121897 | 4121898 | 4121899 | 4121900 | 4121901 |
| Mistral-Nemo-12B | 4121902 | 4121903 | 4121904 | 4121905 | 4121906 |
| Llama-3.2-3B | 4121907 | 4121908 | 4121909 | 4121910 | 4121911 |

Generation ve puanlama array’leri `0-3` shard’larını kapsar; paket işi bunların
başarısına bağlıdır. Tam setin build ve gönderiminde kullanılan komutlar:

```bash
# ~/purerepo-v5-all-models içinde logs/v5-full.sh (compute node):
llmr build-queries configs/benchmarks/shared_v5.yaml
bash cluster/tosun/submit_my_models.sh configs/benchmarks/shared_v5.yaml ibrahimg 4
```

Tam toplama gönderim numaraları `logs/v5-full-submissions.txt` dosyasına yazılır;
mükerrer job göndermeden bu dosya ve `squeue` kontrol edilir. Donmuş Python
kaynakları çalışma sırasında değiştirilmez; sonraki doc commit’leri kaynak
hash’ini değiştirmek için çalışan checkout’a pull edilmez. Diğer üyelerin
sekiz modeli bu pilotla doğrulanmış sayılmaz; kendi pilotları gereklidir.

## shared_v4: referansın düşünmesiz Qwen ayarı ve daha uzun cevap bütçesi

**21 Ekim 2026 — kullanıcı kararı.** Kullanıcı, referansın Qwen ayarını uygulamayı
ve cevap bütçesi yetmediğinde daha uzun üretim beklemeyi istedi. Bu, kesilmeleri
kabul ederek v3’e devam etme kararı değildir. V3 pilotu ve yapılandırmaları korunur;
yeni sürüm `configs/benchmarks/shared_v4.yaml`, pilotu `shared_v4_pilot.yaml`, havuzu
`configs/model_pools/team_v4.yaml` dosyalarındadır. Üye atamaları aynı
`cluster/tosun/team_v3.yaml` dosyasından okunur.

| değişiklik | v3 | v4 | gerekçe |
|---|---|---|---|
| Qwen3-32B thinking | açık | kapalı | sabitlenen referans `e6578414` Qwen politikasını izlemek |
| Qwen3-32B çıktı / toplam context | 8192 / 16384 | 16384 / 32768 | önceki pilotta düşünme sırasında 5 kesilme; cevaba daha çok yer |
| Mistral-Nemo çıktı / toplam context | 8192 / 16384 | 16384 / 32768 | önceki pilotta 1 kesilme |
| diğer modellerin bütçesi | 8192 / 16384 | aynı | bu adımda değişiklik gerekmiyor |

Bu sayılar token tavanlarıdır. **Maliyet = gerçek prompt token’ı × dondurulmuş
input tarifesi + gerçek üretilen token × dondurulmuş output tarifesi** (tarifeler
milyon token başına olduğundan toplam 1e6’ya bölünür). Örneğin tavan 16384 iken
model 900 token’da EOS verirse 900 output token fiyatlanır. Boş düşünce etiketi
gibi gerçekten üretilen token’lar da bu sayımın içindedir. 2026-10-19 fiyatları,
model kimlik/revision’ları, kaynaklar, bölmeler, seed, T=0.7, top_p=1.0 ve üç ayrı
kalite kanalı aynıdır; fiyatlar tekrar çekilmedi. Bu yine yerel üretime uygulanmış
**estimated API USD** hesabıdır, Tosun GPU faturası değildir. Modeller arasında
çıktı tavanı farklıdır; karşılaştırmada v4’ün bu farkı açıkça raporlanmalıdır.

Referans yerel HF provider’ının varsayılanı 4096’dır; **16384 bu varsayılanın kopyası
değil, kullanıcının daha uzun cevap isteğiyle alınan v4 kararıdır.** Referans
aynı zamanda kapanmamış düşünceyi sıfır GT olarak değerlendirir. V4 bunun yerine
başarısız cevapları veri dışına atmaz ve başarılı örneği daha iyi cevap gelene
kadar tekrar üretmez; yeni pilotun kesilme sayısını ayrıca kontrol eder.

Tosun’da ayrı checkout `~/purerepo-v4-nonthinking` kullanılır. V3’te doğrulanan native
Mistral tokenizer adaptörü ve shard kilidine uygun sıralı GT → ArmoRM → BARTScore
akışı bu checkout’a alınır; önceki katkıların source hash’i değiştirilmez.
Yerelde bütün testler **99 passed** (16.32 s), Tosun build işinde de **99 passed**
(124.24 s). Build işi `4111079`, ardından pilot gönderimi `4111080` olarak kuyruğa
alındı. `4111176`, dört paket tamamlanınca doğrulama işini planlar. Bu iş
`logs/v4-pilot-report.json` dosyasına model başına bitiş nedeni, token kullanımı ve
gerçek token’lardan hesaplanan estimated API USD özetini yazar. Bütün paketler
bütünse, cevaplar doğal EOS ile bitmişse ve düşünce temizliği sonrası boş cevap
yoksa tam v4 soru setini compute node’da build edip yalnız İbrahim G’nin dört
modelini 4 shard ile gönderir. Devam eden kesilmede tam toplamayı başlatmaz;
tekrarlı çıktı ile anlamlı devam ayrıştırılarak yeni bütçe/sürüm değerlendirilir.
V4 tam toplama başlatılmadı: Llama’nın 8309 token’lık ArmoRM girdisi 8192 sınırını aştı. Bu durum için kullanıcının eksik-skor kararı ve havuz geneli bütçeler v5’te uygulanır; v3 sayıları v4/v5 sonucu olarak kullanılamaz.

```bash
export LLMR_REPO="$HOME/purerepo-v4-nonthinking"
export PYTHONPATH="$LLMR_REPO"
cd "$LLMR_REPO"
source cluster/tosun/config.sh
llmr_activate
# Compute node üzerindeki build job’unda:
llmr build-queries configs/benchmarks/shared_v4_pilot.yaml
# Build başarılı olduktan sonra:
bash cluster/tosun/submit_my_models.sh configs/benchmarks/shared_v4_pilot.yaml ibrahimg 1
python cluster/tosun/inspect_shard.py configs/benchmarks/shared_v4_pilot.yaml qwen3-32b ibrahimg
# Aynı kontrol diğer üç model için de yapılır; sonuç görülmeden tam toplama başlatılmaz.
```

## shared_v3: fiyatı doğrulanabilen model havuzu

**19 Ekim 2026 — yeni koleksiyonun hazırlığı.** Kullanıcının fiyatı bulunabilen
modellerle çalışma isteğiyle ana sonraki sürüm v3 oldu. V1/v2 dosyaları ve sonuçları
değiştirilmedi. Bu bölüm aşağıdaki tarihsel v1/v2 kayıtlarından önce okunmalı;
21 Ekim 2026 itibarıyla İbrahim G’nin dört modeli için gerçek GPU pilotu tamamlandı;
tam koleksiyon henüz başlatılmadı. Aşağıdaki pilot sonuçları, 12 modelin tamamı için doğrulama değildir.

### Neyi değiştirdik, neden?

Havuz yine 12 açık ağırlıklı model. Korunan altı: Llama3.2-3B, Qwen2.5-7B,
Llama3.1-8B, Phi-4, Mistral Small 24B, Gemma3-27B. Çıkan altı: SmolLM3-3B,
Phi-4-mini, Mistral7B-v0.3, OLMo3-7B, R1-Distill-Llama8B, Qwen2.5-32B.
Yerlerine Gemma3-4B/12B, Mistral-Nemo12B, Qwen3-14B/32B ve Qwen2.5-Coder32B
kondu: güncel OpenRouter kataloğunda açık HF kimliğiyle eşleşen ve pozitif fiyatlı
sağlayıcısı doğrulanan modeller. Bu seçim yalnız kapasite/fiyat çeşitliliği sağlar;
model kalitesi veya Pareto üstünlüğü iddiası değildir. Özellikle daha büyük modelin
API fiyatının daha yüksek olması gerekmez.

`configs/model_pools/team_v3.yaml` her modelin HF revision'ını, gerçek parametre
sayısını (`parameters_b`, HF safetensors metadata'sından), BF16 yerel üretim ayarını ve
ayrı `pricing` kaydını taşır. Parametre sayısı artık dolar tarifesi yerine kullanılmaz.
Havuz referans SU Lab `e6578414` havuzundan yine ayrıdır. Yedi kaynak, formatter'lar,
60/10/30 split, near-duplicate gruplama, üç ayrı skor kanalı ve k=10 örnek korunur.
Ortak üretim T=0.7, top_p=1.0, en çok 8192 yeni token, toplam 16384 bağlamdır.
Qwen3-14B'de thinking kapalı, Qwen3-32B'de açık; ikinci model düşünme çeşitliliğini
korur. Düşünme token'ları çıktı maliyetine dahildir, puanlamadan önce izler temizlenir.

Referansın 12 model kimliğiyle karşılaştırmada **ortak model sayısı 0/12**.
Yukarıdaki “korunan altı” bizim eski v2 havuzumuzdan korunanlardır; referanstan
alınmış modeller değildir. Örneğin referansta Qwen3-4B/8B ve Gemma4 varken bizde
Qwen3-14B/32B ve Gemma3 var. Aynı model ailesinin farklı checkpoint'leri bulunabilir;
birebir model kimliği ortak değildir. Bu, incelenen referans commit'iyle model
seçimi sırasında yapılan karşılaştırmanın sonucudur.

### Donmuş fiyat tablosu

**USD / 1 milyon token, 2026-10-19.** Bunlar model kataloğunun değişken “en düşük”
fiyatından değil, belirli sağlayıcı endpoint'inin tarifesinden alınmıştır.

| model | parametre (B) | sağlayıcı / hosted precision | input $/M | output $/M | kaynak |
|---|---:|---|---:|---:|---|
| `llama-3.2-3b` | 3.213 | Parasail / `bf16` | 0.05 | 0.33 | [tarife](https://openrouter.ai/api/v1/models/meta-llama/llama-3.2-3b-instruct/endpoints) |
| `gemma-3-4b` | 4.300 | Parasail / `bf16` | 0.04 | 0.05 | [tarife](https://openrouter.ai/api/v1/models/google/gemma-3-4b-it/endpoints) |
| `qwen2.5-7b` | 7.616 | Phala / `unknown` | 0.1 | 0.2 | [tarife](https://openrouter.ai/api/v1/models/qwen/qwen-2.5-7b-instruct/endpoints) |
| `llama-3.1-8b` | 8.030 | DeepInfra / `fp8` | 0.02 | 0.04 | [tarife](https://openrouter.ai/api/v1/models/meta-llama/llama-3.1-8b-instruct/endpoints) |
| `mistral-nemo-12b` | 12.248 | DekaLLM / `fp8` | 0.018 | 0.03 | [tarife](https://openrouter.ai/api/v1/models/mistralai/mistral-nemo/endpoints) |
| `gemma-3-12b` | 12.187 | DeepInfra / `bf16` | 0.05 | 0.15 | [tarife](https://openrouter.ai/api/v1/models/google/gemma-3-12b-it/endpoints) |
| `qwen3-14b` | 14.768 | NextBit / `int4` | 0.1 | 0.22 | [tarife](https://openrouter.ai/api/v1/models/qwen/qwen3-14b/endpoints) |
| `phi-4` | 14.660 | DeepInfra / `bf16` | 0.07 | 0.14 | [tarife](https://openrouter.ai/api/v1/models/microsoft/phi-4/endpoints) |
| `mistral-small-24b` | 23.572 | DeepInfra / `fp8` | 0.05 | 0.08 | [tarife](https://openrouter.ai/api/v1/models/mistralai/mistral-small-24b-instruct-2501/endpoints) |
| `gemma-3-27b` | 27.432 | DeepInfra / `fp8` | 0.08 | 0.16 | [tarife](https://openrouter.ai/api/v1/models/google/gemma-3-27b-it/endpoints) |
| `qwen2.5-coder-32b` | 32.764 | Cloudflare / `unknown` | 0.66 | 1 | [tarife](https://openrouter.ai/api/v1/models/qwen/qwen-2.5-coder-32b-instruct/endpoints) |
| `qwen3-32b` | 32.762 | DeepInfra / `fp8` | 0.08 | 0.28 | [tarife](https://openrouter.ai/api/v1/models/qwen/qwen3-32b/endpoints) |

Kaynakların herkese açık `GET /api/v1/models/{id}/endpoints` yanıtları ve HF metadata'sı
okundu; inference çağrısı yapılmadı, ağırlık indirilmedi. OpenRouter ham `prompt` ve
`completion` fiyatları **token başına** USD'dir; config'e bir milyonla çarpılarak yazıldı.
[Resmî fiyat şeması](https://openrouter.ai/docs/guides/overview/models).

Seçim: erişilebilir (`status=0`), normal service tier, pozitif iki tarife, en az
16384 context ve 8192 max completion; ek request/internal-reasoning ücreti veya fiyat
override'ı olmayan ve indirimli olmayan endpoint'ler arasından `input+output` toplamı
en küçük olan (eşitlikte endpoint tag'i). Bu eşit input/output ağırlıklı sabit bir
seçimdir; her cevap uzunluğu için “en ucuz sağlayıcı” iddiası değildir. Tarih,
OpenRouter ID, katalogdaki HF ID, sağlayıcı, endpoint, quantization ve yayınlanmış
context/prompt/completion sınırları config'tedir. Llama3.1'in katalogdaki `Meta-Llama`
HF alias'ı canonical `Llama-3.1-8B-Instruct` reposuna çözülerek revision sabitlendi.

Yerel BF16 çıktıyı bu tarifeyle fiyatlamak **tahmini API USD** hesabıdır. Hosted FP8,
INT4 veya yayınlanmamış precision yerel BF16 ile aynı kalite ölçümü değildir; sağlayıcı
HF revision'ını da garanti etmiyor. Phi-4 sağlayıcısında ayrıca 4096 input-token
sınırı var: daha uzun yerel sorular için bu tarife yalnız karşıolgusal hesap olur,
tüm benchmark'ın o endpoint'te çalışabileceği anlamına gelmez. Gerçek deployment
iddiası için bu sorular ayrıca kapsam dışı raporlanmalı veya yeni sürümde sınırlandırılmalı.
Cache/batch indirimleri, servis/ödeme ücretleri ve gerçek Tosun gideri hesaba katılmaz.
Bu yüzden “gerçek API faturası” veya “Tosun'da dolar harcaması” diye sunulmaz.

### Maliyet ve router bağlantısı

```text
realized_cost = (prompt_tokens * input_per_1m + completion_tokens * output_per_1m) / 1e6
GraphRouter cost_hat = (prompt_tokens * input_per_1m
                      + E_train[completion_tokens | model, source] * output_per_1m) / 1e6
UniRoute: eğitim modellerinin maliyet profili train'den, yeni modellerinki val'den
```

Örnek: Llama3.2-3B için 1000 input + 2000 output token =
`1000*0.05/1e6 + 2000*0.33/1e6 = $0.00071`.
`collection/assemble.py` zaten iki tarifeli hesabı destekliyordu; yeni bir maliyet
motoru eklenmedi. Eksik/geçersiz fiyatlar generation işinden önce reddedilir.
Fiyatlar resolved release config hash'ine bağlanır; yüklerken değişmiş tarife kabul
edilmez. Eğitim internete bağlanıp fiyat güncellemez; yeni fiyat yeni release ister.

V3 view'ında `extras/prompt_tokens` modelin chat template/tokenizer'ından gelen,
cevaptan **önce** bilinen girdidir. Eski v1/v2 view biçimi ve fingerprint'i değişmez.
`meta.model_info` sabit parametre ve tarifeyi verir. `completion_tokens`, kalite ve
gerçek sample maliyetleri testte seçim girdisi değildir. Contract kontrolü bu çıktı
kanallarını karıştırır, public prompt uzunluğunu sabit tutar.

GraphRouter/UniRoute'un SU Lab çekirdeği, native label/loss/karar kuralı ve makale–kod
farkları [eğitim planında](../team/uc_router_egitim_plani.md) kalır. Bu adım üç router'ı
implemente etmez; veri toplandıktan sonra bunları mevcut runner'a bağlayacak yapıyı
hazırlar. GraphRouter'ın maliyet formülü referansla artık aynı input/output USD
biçimindedir. Hybrid için yeni öneri Gemma3-4B/27B çifti; UniRoute 6/6 ayrımı ve
Hybrid label seçenekleri test açılmadan ayrıca sabitlenecek.

Gemma3'ün 4/12/27B checkpoint'i conditional model olduğundan HF fallback'te
`hf_loader: multimodal_lm` ile uygun model sınıfı seçilir, text-only girdi kullanılır.
`max_model_len` yalnız vLLM scheduler ayarıdır, HF model constructor'ına aktarılmaz.
[Gemma model dokümanı](https://huggingface.co/docs/transformers/model_doc/gemma3).
Gerçek model yükleme/bellek kontrolü CPU testinin kapsamı değildir.

### Tosun'da uygulanacak prosedür

Atama `cluster/tosun/team_v3.yaml`: İbrahim G 62.991B, Damla 63.911B, İbrahim B 66.650B;
herkeste dört model. Bu toplamlar runtime ölçümü değildir. BF16 ağırlıklar toplam
~387 GB; scorer ve cache için ek disk gerekir. Herkes kendi Llama/Gemma lisansını
kabul eder. 12B ve üzeri generation için bir 96GB RTX PRO 6000 istenir; scorer'lar ayrı
isteklerini korur. Pilot bellek/throughput sonucu alınmadan tam koleksiyon başlatılmaz.

```bash
# Koordinatör: bir kez build; çıktı bütün üyelerle byte-identical paylaşılır.
# Login node yerine compute node'da kaynak indirme/gruplama:
sbatch -p cuda --qos=cuda -c 4 --mem=16G -t 30 -o logs/build-v3-pilot-%j.out \
  --wrap "source cluster/tosun/config.sh; llmr_activate; llmr build-queries configs/benchmarks/shared_v3_pilot.yaml"
bash cluster/tosun/submit_my_models.sh configs/benchmarks/shared_v3_pilot.yaml <member> 1
python cluster/tosun/inspect_shard.py configs/benchmarks/shared_v3_pilot.yaml <model> <member>
# Pilot kontrolünden sonra full için yeni dataset build + her üyeden dört model:
sbatch -p cuda --qos=cuda -c 4 --mem=16G -t 60 -o logs/build-v3-%j.out \
  --wrap "source cluster/tosun/config.sh; llmr_activate; llmr build-queries configs/benchmarks/shared_v3.yaml"
bash cluster/tosun/submit_my_models.sh configs/benchmarks/shared_v3.yaml <member> 4
# Bütün katkılar toplandıktan sonra koordinatör:
llmr assemble configs/benchmarks/shared_v3.yaml
llmr validate-data configs/benchmarks/shared_v3.yaml
llmr evaluate-data configs/benchmarks/shared_v3.yaml
bash cluster/tosun/submit_train.sh configs/experiments/quality_regression__shared_v3.yaml 3
```

**Tam toplama ve router eğitimi henüz çalıştırılmadı.** Pilot özellikle Qwen3-32B düşünme
kesilmesi, Multi-News/Phi-4 context sınırı, Gemma text template'i, EOS, tokenizer
usage ve 12B+ bellek kullanımını denetler. Test split'i router ayarı seçmek için kullanılmaz.
CPU kontrolleri ve gerçek sayıları [adım 2](02_router_egitimi.md#v3-fiyatlı-havuzun-cpu-kontrolü) kaydeder.

### İbrahim G’nin gerçek v3 pilotu — 21 Ekim 2026

Damla’nın `64462f5` commit’i, Tosun’da mevcut değiştirilmiş `~/purerepo` korunarak
`~/purerepo-v3-64462f5` klasöründe çalıştırıldı. `LLMR_REPO` ve `PYTHONPATH` bu
klasöre ayarlandı; mevcut Python ortamı kullanıldı. Compute node üzerinde **98 test
geçti**, ardından `shared_v3_pilot` için **30 soru** donduruldu (build işi `4093298`).
Her modele aynı sorular için k=2 ile 60 cevap üretildi; tam toplamadaki k=10 değişmedi.

| model | başarılı cevap | EOS / token sınırı | en uzun çıktı | GT / ArmoRM / BARTScore |
|---|---:|---:|---:|---:|
| Qwen3-32B | 60 | 55 / 5 | 8192 | 60 / 60 / 60 |
| Qwen3-14B | 60 | 60 / 0 | 3844 | 60 / 60 / 60 |
| Mistral-Nemo-12B | 60 | 59 / 1 | 8192 | 60 / 60 / 60 |
| Llama-3.2-3B | 60 | 60 / 0 | 1992 | 60 / 60 / 60 |

Burada “başarılı cevap” üretim işleminin tamamlanmasıdır; doğru cevap veya kesilmemiş
cevap demek değildir. **Qwen3-32B’nin 8 OlympiadBench cevabından 5’i** düşünme
sırasında 8192 token sınırına ulaştı; kapanmayan düşünce bloğu mevcut `strip_think`
kuralıyla boş cevaba dönüştü. Bu gözlem nedeniyle tam toplama başlatılmadı.
8192 sınırı altında bu kesilmelerin kabul edilip edilmeyeceği kullanıcıyla
netleştirilmeli; token bütçesi veya model davranışı sessizce değiştirilmemeli.

Dört paketin manifest dosyaları ve bütün dosya hash’leri doğrulandı; her pakette
60 başarılı üretim anahtarıyla eşleşen, kanal başına 60 başarılı skor var, işlem
hatası yok. Prompt/çıktı token sayıları negatif olmayan tamsayılar ve seed uygulaması
kontrol edildi. Prompt token aralıkları sırasıyla 15–2247, 19–2251, 10–2518 ve
42–2229. Paketler `~/llmr_data/benchmarks/shared_v3_pilot/exports/` altında,
`ibrahimg-<model>-00000-of-00001` adlarıyla duruyor. Bu dört paket ortak 12 modellik
release olarak assemble edilmedi; diğer üyelerin işleri başlatılmadı.

**Uygulama düzeltmeleri.** Collector her shard için tek `.writer.lock` tuttuğundan
paralel scorer işleri çakıştı. Skorları yeniden üretmek yerine eksik BARTScore
adımları yeniden gönderildi; yeni zincirlerde GT → ArmoRM → BARTScore sırası
kullanıldı. Submit script’i de aynı sırayı uygular; kalite kanalları bağımsız kalır.
Llama lisans onayı gelince generation `4094814`, skorlar `4094815–4094817`, paket
`4094818` olarak tamamlandı.

vLLM’nin native `MistralTokenizer` nesnesi HF’nin `name_or_path`, `init_kwargs` ve
Jinja `chat_template` alanlarını sağlamadığı için küçük bir backend adaptörü gerekti.
Native `apply_chat_template(tokenize=True)` token ID’leri doğrudan kullanılır;
control token’lar metne çevrilip yeniden encode edilmez. Native template sınıfı/sürümü
ve istenen pinned tokenizer revision metadata’ya yazılır. Adaptör
`~/purerepo-v3-nemo-compat` içinde izole edildi; önceki modellerin source hash’i
değiştirilmedi. **10 backend testi** ve gerçek pinned tokenizer ile ID eşitliği
kontrolü geçti (`4094888`); Mistral üretimi `4094920`, skorları `4094921–4094923`,
paketi `4094924` olarak tamamlandı. Başarısız denemeler ayrı arşivde saklandı ve paketlere katılmadı. Bu adaptör ve test sürümün kod değişikliklerine dahildir.

Kontrol komutları (her model için; ilgili checkout etkinleştirilerek):

```bash
export LLMR_REPO="$HOME/purerepo-v3-64462f5"
export PYTHONPATH="$LLMR_REPO"
cd "$LLMR_REPO"
source cluster/tosun/config.sh
llmr_activate
python cluster/tosun/inspect_shard.py configs/benchmarks/shared_v3_pilot.yaml qwen3-32b ibrahimg --show-failed 0
# Diğer modeller için aynı komutta model adını değiştirin.
```

## Büyük resim

Sonunda elde edeceğimiz şey kabaca şu (sayılar örnek):

| soru | kaynak | split | smollm3-3b | ... | qwen2.5-32b |
|---|---|---|---|---|---|
| "Janet's ducks lay 16 eggs..." | GSM8K | train | GT 1, ArmoRM 0.18, maliyet 1.9 | ... | GT 1, ArmoRM 0.21, maliyet 20.1 |
| "Find the smallest n such that..." | OlympiadBench | test | GT 0, ... | ... | GT 1, ... |

Küçük model ilk soruda yetiyorsa büyük modele 10 kat ödemeye gerek yok; ikincide büyük model şart. Router
bu farkı sorunun metninden tahmin etmeyi öğrenir. Tabloyu üreten akış:

```text
1. sorular            llmr build-queries    7 kaynaktan 7652 soru, train/val/test'e bölünür ve dondurulur
2. modeller           team.yaml             12 model, sabit sürümler
3. cevap üretme       llmr generate         her (soru, model) için 10 cevap, Tosun GPU'sunda vLLM ile
4. puanlama           llmr score ×3         her cevap: ground truth, ArmoRM, BARTScore (ayrı ayrı)
5. kontrol + paket    inspect_shard.py,     sonuç okunur, doğrulanır, değişmez bir paket olarak teslim edilir
                      package-contribution
   (sonra)            llmr assemble         bütün modellerin paketleri tek tabloda birleşir
```

Adım 3–5 her model için bir üye tarafından ayrı yapılır; sorular herkes için aynıdır.

## 1. Sorular: hangi testler ve neden

[`shared_v1.yaml`](../../configs/benchmarks/shared_v1.yaml) yedi kaynak kullanır:

| kaynak | ne ölçüyor | soru | nasıl puanlanıyor |
|---|---|---|---|
| GSM8K | çok adımlı aritmetik problem | 1319 | sayı doğru mu, 0/1 |
| MMLU-Pro | 14 alanda bilgi ve akıl yürütme, 10 şık | 1500 | şık doğru mu, 0/1 |
| BBH | 27 zor akıl yürütme görevi (mantık, tarih, sayma, nesne takibi) | 2160 | cevap aynı mı, 0/1 |
| OlympiadBench | yarışma matematiği, açık uçlu | 674 | matematiksel eşdeğerlik, 0/1 |
| MBPP | kısa Python fonksiyonu yazma | 500 | testler geçiyor mu, 0/1 |
| Alpaca | genel talimat takibi | 1000 | referans cevapla kelime örtüşmesi, 0–1 |
| Multi-News | çok makaleli haber özeti, uzun girdi | 500 | referans özetle kelime örtüşmesi, 0–1 |

**Neden bunlar:**

- **Referansla aynı.** SU Lab LLM-Routing-Project bu karışımı kullanıyor. Biz farklı modellerle aynı soruları
  çözdüğümüz için sonuçlar yan yana okunabiliyor.
- **Zorluk ve beceri yelpazesi.** BBH'yi 3B'lik model de çözer, OlympiadBench'i 32B'lik model bile zor çözer.
  Aritmetik, bilgi, mantık, kod, serbest yazı, uzun metin var. Her soru aynı zorlukta olsaydı her soruya aynı
  model giderdi ve router'ın öğreneceği bir şey kalmazdı.
- **Programla kontrol edilebilir.** Beş kaynakta kesin bir doğru cevap var; insan ya da ücretli bir LLM
  hakemi gerekmiyor. Alpaca ve Multi-News'te kesin cevap yok; bunlar serbest metin kalitesi için.

**Prompt'lar** modelden cevabı, puanlayıcının ilk baktığı biçimde ister
([`formatters.py`](../../llm_routing/collection/formatters.py)): GSM8K'da son satır `Answer: <sayı>`,
şıklılarda `Answer: X`, OlympiadBench'te `\boxed{}`, MBPP'de tek bir ` ```python ` bloğu. Böylece cevap çoğunlukla
kesin bir kuralla okunur.

**Bölme** cevaplardan **önce** yapılır ve dondurulur:
1. Neredeyse aynı sorular (MiniLM benzerliği ≥ 0,99) aynı split'e düşer. Yoksa bir sorunun ikizi train'de,
   kendisi test'te olur ve router ezberle iyi görünür.
2. Her kaynak kendi içinde 60/10/30 bölünür.

Kural: router'lar `train` ile eğitilir, ayarlar `val` ile seçilir, `test` yalnız son raporda kullanılır.

## 2. Modeller: hangi 12 model ve neden

[`team.yaml`](../../configs/model_pools/team.yaml):

| grup | modeller |
|---|---|
| küçük (3–4B) | SmolLM3 3B, Llama 3.2 3B, Phi-4-mini |
| orta (7–8B) | Mistral 7B v0.3, OLMo 3 7B, Qwen2.5 7B, Llama 3.1 8B, DeepSeek-R1-Distill-Llama 8B |
| büyük (15–33B) | Phi-4, Mistral Small 24B, Gemma 3 27B, Qwen2.5 32B |

**Neden bunlar:**

- **Gerçek maliyet farkı.** 3B'den 33B'ye. Routing'in kazancı, ucuz modelin yettiği soruyu ucuza göndermekten
  geliyor; benzer boyutlu modellerle bu kazanç olmaz.
- **Farklı aileler.** Llama, Qwen, Mistral, Phi, Gemma, OLMo, SmolLM, DeepSeek. Aynı boyutta iki model farklı
  konularda güçlü olabilir; router için bir sinyal daha.
- **Referanstan tamamen farklı.** Aynı protokolü başka bir havuzla tekrarlamak, router sonuçlarının bir model
  setine bağlı olup olmadığını gösterir.
- **Bir reasoning modeli.** R1-Distill cevap vermeden önce uzun düşünür: daha pahalı ama zor soruda daha iyi
  olabilir. "Düşünen model ne zaman değer?" routing için iyi bir soru.
- **Ücretsiz ve Tosun'a sığıyor.** Hepsi açık ağırlıklı; en büyüğü tek bir 80/96 GB'lık karta sığıyor.

Bütün modeller aynı ayarlarla cevaplar: `temperature 0.7`, `top_p 0.95`, en fazla 4096 token, soru başına
**10 cevap** (bazı router'lar aynı modelden birkaç cevap üretip en iyisini seçiyor). Her model bir Hugging Face
commit'ine sabit; herkes birebir aynı ağırlıkları kullanır.

**Maliyet:** kendi GPU'larımızda dolar fiyatı yok, bir vekil kullanıyoruz:
`parametre sayısı (milyar) × (prompt + cevap token'ı) / 1000`. 32B'lik model token başına 3B'liğin ~10 katı
hesap yapar, uzun düşünen model de daha çok token harcar. Bu bir USD değildir.

## shared_v2: 19 Ekim 2026 hazırlığı

**Durum:** yeni config'ler yerelde hazır; R1 dahil bütün v2 havuzu için gerçek GPU pilotu henüz
çalıştırılmadı. Yukarıdaki v1 ayarları ve aşağıdaki R1 pilotu tarihsel olarak korunur. Yeni toplama
`configs/benchmarks/shared_v2.yaml`, kurulum pilotu `configs/benchmarks/shared_v2_pilot.yaml` ile yapılacak.
Veriler sırasıyla `benchmarks/shared_v2/` ve `benchmarks/shared_v2_pilot/` altında ayrı tutulur.

### Model ve ortak üretim bütçesi

**R1-Distill-Llama 8B dahil aynı 12 model ve aynı ağırlık revision'ları korunur.** Eski pilotta
4096-token sınırının cevapları kestiği görüldü; bu sonuç tek başına modelin değiştirilmesini
gerektirmez. R1'i tutmak model kimliğini sabit bırakır ve uzun düşünmenin kalite/maliyet katkısını
inceleme olanağı verir. Düşünme token'ları maliyete dahildir; puanlayıcılar nihai cevabı mevcut
`strip_think` politikasıyla ayırır. 8192 bütçesinin yeterli olduğu henüz kanıtlanmadı.
Yeni havuz `configs/model_pools/team_v2.yaml`, aynı üye/model atamaları
`cluster/tosun/team_v2.yaml` içindedir; R1 yine İbrahim B'dedir.

Herkes için `temperature: 0.7`, `top_p: 1.0`, **en fazla 8192 yeni token**; tam koleksiyonda k=10,
kurulum pilotunda k=2. Yerel referansın CLI varsayılanı top_p=1.0'dır; bu değerin seçilme gerekçesi
belgelenmemiştir. Eski 0.95 ayarının birebir referans eşleşmesi olduğu iddia edilmez.

Çıktı limiti modele prompt içinde söylenmez. Normal üretimde model erken EOS üretirse hemen durur:
600 token'da biten cevap, sınır 8192 diye 8192 token'a uzamaz. Üst sınırın artırılması tek başına
token olasılıklarını değiştirmez; backend/batch farklarından dolayı bit düzeyinde aynı cevap
garantisi verilmez. Yüksek sınır bir cevabın tamamlandığını da garanti etmez: tekrar döngüsü, erken
EOS veya `finish_reason: length` ayrı kontrol edilir. Maliyet ayrılan bütçeden değil, gerçek token
sayısından hesaplanır.

Toplam bağlam her model için 16384 olarak korunur; **prompt token'ları + 8192 ≤ 16384** olmalıdır.
Phi-4'ün doğal bağlamı 16384 olduğundan her modele gelişigüzel 32768 çıktı bütçesi verilemez. V2
pilotunda modelin tokenizer'ıyla gerçek prompt uzunluğu ve kesilme oranı kontrol edilmelidir.
Uzun cevapların ArmoRM bağlamına sığması ve BARTScore'un mevcut 512-token kesmesi de ayrı sınırlardır.

### Tosun'da canlı kontrol

19 Ekim 2026'da login node'da şu salt okuma komutları çalıştırıldı:

```bash
sinfo -p cuda -N -o "%N|%G|%m|%t"
scontrol show nodes
squeue -u "$USER" -o "%.18i %.15j %.9T %.10M %.30R"
```

| node | gerçek GPU kaynak adı | kart |
|---|---|---|
| cn02, cn03 | `rtx_pro_6000:1` | RTX PRO 6000, 96 GB |
| cn21 | `rtx_pro_6000:2` | iki RTX PRO 6000, 96 GB/kart |
| cn08 | `a100_80gb:1` | A100, 80 GB |
| cn18 | `h200_nvl:1`, `rtx_4500:2` | H200 141 GB ve iki 24 GB kart |

Kullanıcının kuyruğu o anda boştu; bu, kümenin boş olduğu anlamına gelmez. A100, H200 ve cn21'in
GPU'ları diğer işlere ayrılmıştı. cn02/cn03'te GPU ayrılmamış görünse de CPU kaynaklarının çoğu doluydu;
başlama zamanı garanti edilemez.

V2'de Phi-4 ve üç büyük modelin generation işi `gpu:rtx_pro_6000:1` ister. `submit_member.sh` modelin
`gpu_type` alanını yalnız generation'a uygular; ArmoRM/BARTScore küçük kartlarda kendi isteklerini korur.
`LLMR_GPU_TYPE` açıkça verilirse model ayarını geçersiz kılar; gerçek alternatif adlar `a100_80gb` ve
`h200_nvl`'dir. Tek kartta sığma ve throughput, gerçek pilot geçmeden doğrulanmış sayılmaz.

Tosun'daki mevcut `pilot_tosun` paketi ayrıca salt okunarak kontrol edildi: **60 başarılı cevap,
51 stop / 9 length, en uzun prompt 2199 token, en uzun çıktı 4096 token**. GT, ArmoRM ve BARTScore'un
her birinde 60 başarılı skor, sıfır işlem hatası vardı. Bunlar eski R1 pilotudur; v2 başarısı değildir.
Uzak checkout eski modül yerleşiminde ve kaydedilmemiş değişiklikler içeriyordu; üzerine dosya
yazılmadı, `git pull` veya yeni GPU işi başlatılmadı.

Yerel doğrulama: **79 test geçti**; test modelleriyle 30 soru / 120 cevap içeren uçtan uca smoke
release'i doğrulandı. V1/v2 kaynakları, scorer ayarları ve split kuralları eşit; release kimlikleri ve
veri yolları farklı olarak kontrol edildi. Güncellenen submit script'i ayrıca Tosun'un Bash 5.2.21
ortamında sahte scheduler ile çalıştırıldı: v1 GPU isteği, v2'nin generation/scorer ayrımı ve H200
environment override'ı geçti; bu kontrol gerçek iş submit etmedi.

Kalite kanalları ayrı kalır; mevcut `evaluate-data` kaynak bazında da sonuç verir. Beş doğrulanabilir
kaynağın doğruluğu ve Alpaca/Multi-News token F1'i raporda ayrı okunmalıdır. Son test skorları üretim
ayarlarını veya router eşiklerini seçmek için kullanılmaz.

## 3. Cevap üretme: Tosun'da bir modeli çalıştırmak

**Tosun'un yapısı.** İşler SLURM ile yürür: `sbatch` ile kuyruğa verilir, boş GPU bulununca bir compute
node'da çalışır. Login node yalnız giriş ve iş göndermek içindir (bellek sınırı var; büyük bir `pip install`
orada öldürülür). Bizim partition'ımız `cuda`. bf16 desteklemeyen V100/K80 node'ları ve 16 GB kartlı cn14
varsayılan olarak hariç ([`config.sh`](../../cluster/tosun/config.sh)). Kalan kartlar 24 GB (A5000, RTX 4500),
80 GB (A100), 96 GB (RTX PRO 6000), 141 GB (H200). 8B'lik model 24 GB'a, 32B'lik model 80 GB ve üstüne sığar.

**Bir kez yapılanlar:**

```bash
# laptop: SSH anahtarı (şifreyi bir kez siz girersiniz)
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_tosun
ssh-copy-id -i ~/.ssh/id_ed25519_tosun.pub <kullanıcı>@tosun.sabanciuniv.edu

# Tosun: ortam, bir compute node'da
cd ~/purerepo && mkdir -p logs
sbatch -p cuda --qos=cuda -c 8 --mem=32G -t 2:00:00 -o logs/setup_env-%j.out --wrap "bash cluster/tosun/setup_env.sh"
hf auth login        # gated modeller (Llama, Gemma) için; lisansı huggingface.co'da kabul edin
```

[`setup_env.sh`](../../cluster/tosun/setup_env.sh) sürümleri sabitler, çünkü Tosun'un GPU sürücüsü CUDA 12.x
(570) ve PyPI'deki en yeni vLLM/torch CUDA 13 için derlenmiş: o sürümler GPU'yu görmez. Biz CUDA 12.9 sürümlerini
kuruyoruz (torch 2.13.0+cu129, vLLM 0.31.0+cu129).

**Önce pilot, sonra tam koleksiyon.** Pilot ([`shared_pilot.yaml`](../../configs/benchmarks/shared_pilot.yaml))
aynı kaynak ve ayarlardan 30 soru, 2 cevap; bütün zinciri dakikalar içinde gerçek modellerle dener.

```bash
# soruları bir kez dondur (compute node'da)
sbatch -p cuda --qos=cuda -c 4 --mem=16G -t 30 -o logs/build-%j.out \
  --wrap "source cluster/tosun/config.sh; llmr_activate; llmr build-queries configs/benchmarks/shared_pilot.yaml"

# bir modeli çalıştır: config, model, üye adı, shard sayısı
bash cluster/tosun/submit_member.sh configs/benchmarks/shared_pilot.yaml deepseek-r1-distill-llama-8b ibrahim 1
```

`submit_member.sh` beş işi bağımlılıklarıyla kuyruğa verir:

```text
generate (GPU) ──┬──► ground_truth (CPU) ──┐
                 ├──► armorm (GPU)         ├──► package
                 └──► bartscore (GPU)      ┘
```

`generate` her (soru, cevap no) için ayrı bir seed'le istek oluşturur ve vLLM'e 256'lık gruplar halinde verir;
her cevap `responses.jsonl`'a bir satır olur. Yarıda kalan iş aynı komutla tekrar gönderilir, biten cevaplar
atlanır. Tam koleksiyonda büyük modeller için shard sayısı artırılır (ör. 4 paralel GPU işi).

İzlemek için `squeue -u $USER` ve `logs/llmr-member-<jobid>_0.out`. `squeue`'daki `DependencyNeverSatisfied`,
önceki işin başarısız olduğu anlamına gelir: logunu okuyun, kalanları `scancel` edin, düzeltip yeniden gönderin.

Pilotta generation H200'de ~7 dakika sürdü (16 GB model indirme dahil), ArmoRM ~4, BARTScore ~1 dakika.

## 4. Puanlama: bir cevabın iyi olduğuna nasıl karar veriyoruz

Her cevap üç **ayrı** kanalla puanlanır, çünkü makaleler farklı kalite sinyali kullanıyor ve her router kendi
makalesindekini alacak. Kanallar asla karıştırılmaz; birinin eksiği diğeriyle doldurulmaz.

| kanal | soru | ölçek | kim kullanıyor |
|---|---|---|---|
| **ground truth (GT)** | cevap doğru mu? | 0/1, serbest metinde 0–1 | çoğu router (doğruluk etiketi) |
| **ArmoRM** | insan tercihiyle eğitilmiş 8B'lik bir ödül modeli cevabı ne kadar beğeniyor? | sürekli, çoğunlukla 0–0.3 | BEST-Route |
| **BARTScore** | BART, bu prompt'tan sonra bu cevabı ne kadar olası buluyor? | negatif, 0'a yakın iyi | Hybrid LLM |

**GT nasıl karar veriyor.** Her soru bir cevap tipi taşır
([`evaluators.py`](../../llm_routing/ground_truth/evaluators.py)). Önce kesin kural denenir; olmazsa
yalnız şıklı sorularda düzyazıdan şıkkı kurtarmaya çalışan bir yedek kural:

- **sayı (GSM8K):** "Final Answer", `Answer: n`, `\boxed{}`, `#### n`; bulunamazsa son sayı. `1,000` = `1000`.
- **şık (MMLU-Pro), metin (BBH):** son `Answer:` satırı; olmazsa yedek kural.
- **OlympiadBench:** `\boxed{}` içi, sympy ile eşdeğerlik (`1/2` = `0.5`).
- **kod (MBPP):** kod ayrı bir süreçte gold testlerle çalıştırılır; 5 saniyede bitmezse yanlış.
- **serbest metin (Alpaca, Multi-News):** referansla token F1 (kelime örtüşmesi).

Her sonuç bir `mode` ile kaydedilir (`strict`, `fallback`, `failed`, `no_answer`, `timeout`, `token_f1`). Böylece
"yanlış cevap" ile "biçim yüzünden okunamadı" ayırt edilebilir.

**Düşünce izi.** R1-Distill'in cevabı `<think> ...uzun düşünce... </think> asıl cevap` biçiminde. Üç kanal da
yalnız asıl cevabı puanlar. Düşüncede "belki (D)" deyip sonunda "(B)" diyen model (D)'den puan almamalı. İz
4096 token'da kesildiyse ortada cevap yoktur: GT 0 (`no_answer`). Bir incelik var: bu modelde `<think>`
etiketini model değil chat template açar. Bu yüzden generation o öneki cevabın başına geri koyar; koymasaydı
iz hiç silinemezdi.

**Referansla aynı olduğunu nasıl gösterdik.** GT kodu referans repodan birebir alındı (üç belgelenmiş düzeltme
dışında). Pilotun 60 cevabı Tosun'da referans repo'nun **kendi** ArmoRM ve BARTScore fonksiyonlarıyla tekrar
puanlandı: iki repo her cevaba bit düzeyinde aynı skoru verdi (fark 0).

## 5. Sonucu kontrol etmek

Bir job'ın hatasız bitmesi sonucun doğru olduğunu göstermez. Model biçimi yanlış anlayıp her soruda 0 alabilir,
düşünce izi yanlışlıkla puanlanabilir. Paketlemeden önce shard'ı okuyoruz:

```bash
.venv/bin/python cluster/tosun/inspect_shard.py configs/benchmarks/shared_pilot.yaml deepseek-r1-distill-llama-8b ibrahim
```

R1-Distill pilotunun gerçek çıktısı (kısaltılmış):

```text
responses: 60  finish_reason: {'stop': 51, 'length': 9}
reasoning: response_prefix {"'<think>\\n'": 60}  trace closed 52  empty after strip 8
ground_truth: 60 scored, 0 errors   armorm: 60 scored, 0 errors   bartscore: 60 scored, 0 errors

  bbh            n=12  ground_truth 1.000  armorm 0.141  GT modes {'strict': 12}
  gsm8k          n=8   ground_truth 0.750  armorm 0.176  GT modes {'strict': 6, 'no_answer': 1, 'failed': 1}
  olympiadbench  n=8   ground_truth 0.250  armorm 0.067  GT modes {'strict': 2, 'no_answer': 6}
  ...
ArmoRM on 0/1 sources: GT-correct 0.161 (n=28)  GT-wrong 0.098 (n=16)

--- GT failed: gsm8k  gold 89  read as 147   ... **Final Answer: 147**
```

Ne kontrol ediyoruz:

1. **Hepsi puanlandı mı?** Her kanalda hata 0.
2. **Düşünce izi doğru ayrıldı mı?** 60 cevabın hepsinde önek var; 8'inde iz kapanmamış (kesilmiş) ve GT'de
   tam olarak 8 `no_answer` var. Sayılar tutuyor.
3. **`failed` olanlar gerçekten yanlış mı?** Birkaçını okuyun. Pilottakilerin hepsi gerçekten yanlıştı (model
   147 diyor, doğrusu 89). Çoğu cevap `failed` ise sorun büyük ihtimalle biçimdedir, modelde değil.
4. **Kanallar tutarlı mı?** ArmoRM, GT'ye göre doğru cevaplara daha yüksek puan vermeli (0,161 > 0,098).

Pilotun söylediği: zincir gerçek modellerle doğru çalışıyor. Bir de karar verilecek bir konu çıktı: R1-Distill
4096 token sınırında OlympiadBench sorularının çoğunda kesiliyor (8'de 6). Tam koleksiyonda bu modelin
OlympiadBench skoru büyük ölçüde token sınırını ölçecek. 8 soruluk ortalamalar gürültülü; pilot bir model
karşılaştırması değil, doğruluk kontrolü.

## Bilinen sorunlar

| belirti | sebep | çözüm |
|---|---|---|
| login node'da `pip` → `Killed` | login node bellek sınırı | kurulumu `sbatch` ile yapın |
| `torch ... sees no GPU` | CUDA 13 derlemesi, sürücü 12.x | `setup_env.sh` ile yeniden kurun |
| `libtorchcodec_image.so` / `sqlite3_deserialize` hatası | Tosun'daki paket/modül uyumsuzlukları | `setup_env.sh` ve `config.sh` düzeltiyor; job'larda `llmr_activate` kullanın |
| 8B model OOM | 16 GB'lık karta düştü | `LLMR_EXCLUDE`'u daraltmayın |
| 403 indirme | gated model | lisansı kabul edin, `hf auth login` |
| `AssocGrpCpuLimit` ile bekliyor | grubun o partition'daki kotası dolu | `cuda` partition'ını kullanın |

## Sırada ne var

1. Tam koleksiyon: üyeler modelleri `shared_v1.yaml` ile çalıştırır, koordinatör `llmr assemble` /
   `validate-data` / `evaluate-data` ile tek tabloyu yayımlar (kurallar: [ekip rehberi §4](../team/team_workflow.md#4-paketleri-teslim-et-ve-birleştir)).
2. Router'lar: her makale bu tablonun train kısmıyla eğitilir; her biri için ayrı bir anlatım yazılacak.

Ayrıntılı kurallar ve dosya biçimleri: [teknik protokol](../team/shared_protocol.md). Açık kararlar (maliyet tanımı,
büyük modeller için GPU sayısı): [DECISIONS.md](../../DECISIONS.md).
