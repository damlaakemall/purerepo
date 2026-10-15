# Adım 1: Router'lar için eğitim verisi üretmek

> Bir router "bu soruyu hangi model cevaplasın?" kararını öğrenir. Bunu öğrenmesi için elimizde bir tablo
> olmalı: **her soru için, her modelin cevabı ne kadar iyi ve ne kadar pahalı.** Bu adım o tabloyu üretir.
> Soruları seçeriz, 12 modele cevaplatırız (Tosun GPU'larında), her cevabı puanlar ve maliyetini sayarız.
> Sonraki adımda router'lar bu tablonun train kısmıyla eğitilip test kısmında karşılaştırılacak.
>
> **Durum (15 Ekim 2026):** akış hazır ve 30 soruluk bir pilotla gerçek GPU'da uçtan uca doğrulandı. Tam
> koleksiyon (7652 soru × 12 model × 10 cevap) henüz yapılmadı.

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
([`formatters.py`](../../llm_routing/pipeline/shared/formatters.py)): GSM8K'da son satır `Answer: <sayı>`,
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
([`evaluators.py`](../../llm_routing/pipeline/ground_truth/evaluators.py)). Önce kesin kural denenir; olmazsa
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
   `validate-data` / `evaluate-data` ile tek tabloyu yayımlar (kurallar: [ekip rehberi §4](../team_workflow.md#4-paketleri-teslim-et-ve-birleştir)).
2. Router'lar: her makale bu tablonun train kısmıyla eğitilir; her biri için ayrı bir anlatım yazılacak.

Ayrıntılı kurallar ve dosya biçimleri: [teknik protokol](../shared_protocol.md). Açık kararlar (maliyet tanımı,
büyük modeller için GPU sayısı): [DECISIONS.md](../../DECISIONS.md).
