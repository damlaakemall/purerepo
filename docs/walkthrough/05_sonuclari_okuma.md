# 5. Sonuçları okuma ve kontrol: çıkan sayılara nasıl güveniyoruz

> Bir job'ın hatasız bitmesi sonucun doğru olduğu anlamına gelmez. Bir model biçimi yanlış anlayıp her
> cevabı 0 alabilir, düşünce izi yanlışlıkla puanlanabilir, skorlayıcı sessizce bozuk olabilir. Bu adım,
> bir modelin skorlanmış cevaplarını paketlemeden ve birleştirmeden önce nasıl okuduğumuzu ve neyi kontrol
> ettiğimizi anlatır.

## Dosyalar ne içeriyor

Bir shard'ın klasöründe ([4. Tosun](04_tosunda_calistirma.md#örnek-r1-distill-pilotu-15-ekim-2026)) her satır bir
`(soru, model, örnek)`:

| dosya | önemli alanlar |
|---|---|
| `run.json` | dataset kimliği, modelin tam kaydı, generation ayarları, git commit, paket sürümleri |
| `responses.jsonl` | `prompt`, `response` (ham metin, hiç silinmez), `seed`, `success`; `metadata` içinde `finish_reason` (`stop` / `length`), `completion_tokens`, `prompt_tokens`, `response_prefix`, `formatted_prompt`, `batch_seconds_per_sample` |
| `ground_truth.jsonl` | `score` (0/1 veya F1), `details.mode` (`strict`, `fallback`, `failed`, `no_answer`, `timeout`, `token_f1`), `details.predicted` (cevaptan ne okundu) |
| `armorm.jsonl` | `score`, `details.rewards` (19 alt boyut), `details.tokens` (girdi uzunluğu) |
| `bartscore.jsonl` | `score`, `details.target_tokens` |

Her skor satırı `response_hash` ile bağlı olduğu cevaba, `evaluator_id` ile skorlayıcının ayarlarına
bağlıdır; cevap veya ayar değişirse eski skor kullanılamaz.

## Hızlı özet: `inspect_shard.py`

```bash
# Tosun login node'unda olur (yalnız dosya okur, model yüklemez)
cd ~/purerepo
.venv/bin/python cluster/tosun/inspect_shard.py configs/benchmarks/shared_pilot.yaml deepseek-r1-distill-llama-8b ibrahim
```

R1-Distill pilotunun gerçek çıktısı (15 Ekim 2026; kısaltılmış):

```text
responses: 60  finish_reason: {'stop': 51, 'length': 9}
completion tokens: min 215  median 840  max 4096
reasoning: response_prefix {"'<think>\\n'": 60}  trace closed 52  empty after strip 8
ground_truth: 60 scored, 0 errors
armorm: 60 scored, 0 errors
bartscore: 60 scored, 0 errors

per source (mean)
  alpaca         n=8   ground_truth 0.198  armorm 0.117  bartscore -4966.648  GT modes {'token_f1': 8}
  bbh            n=12  ground_truth 1.000  armorm 0.141  bartscore  -485.667  GT modes {'strict': 12}
  gsm8k          n=8   ground_truth 0.750  armorm 0.176  bartscore -3459.879  GT modes {'strict': 6, 'no_answer': 1, 'failed': 1}
  mbpp           n=8   ground_truth 0.250  armorm 0.129  bartscore -5010.168  GT modes {'failed': 5, 'no_answer': 1, 'strict': 2}
  mmlu_pro       n=8   ground_truth 0.750  armorm 0.173  bartscore -2545.191  GT modes {'strict': 6, 'failed': 2}
  multi_news     n=8   ground_truth 0.354  armorm 0.189  bartscore  -562.359  GT modes {'token_f1': 8}
  olympiadbench  n=8   ground_truth 0.250  armorm 0.067  bartscore -7338.141  GT modes {'strict': 2, 'no_answer': 6}

ArmoRM on 0/1 sources: GT-correct 0.161 (n=28)  GT-wrong 0.098 (n=16)  -- correct should score higher

--- GT failed: gsm8k  gold 89  read as 147
    ... Total buttons = 87 small buttons + 60 large buttons = 147 buttons. |  | **Final Answer: 147**
```

## Neye bakıyoruz, neden

Her satırı bir soruya cevap olarak okuyun:

1. **Hepsi skorlandı mı?** `60 scored, 0 errors` her kanal için. Hata varsa `error` alanı nedenini söyler
   (ör. ArmoRM bağlam aşımı). Hatalı satırlar aynı komutla yeniden denenir.
2. **Cevaplar nasıl bitti?** `length`, modelin `max_new_tokens`'a (4096) takılıp kesildiği anlamına gelir.
   Pilotta 60'tan 9'u. Normal bir modelde bu oran çok düşük olmalı; reasoning modelinde yüksek olabilir.
3. **Düşünce izi doğru ayrıldı mı?** (yalnız reasoning modellerinde) 60 cevabın hepsi `<think>\n` önekini
   taşıyor, 52'sinde iz kapanmış, 8'inde kapanmamış (kesilmiş). Kesilen 8 cevap GT'de `no_answer` olarak
   görünüyor (GSM8K 1 + MBPP 1 + OlympiadBench 6 = 8). Sayıların tutması, temizlemenin doğru çalıştığını
   gösteriyor. Önek olmasaydı cevaplarda yalnız `</think>` bulunurdu: temizleme hiçbir izi silemez, `empty
   after strip` 0 çıkar ve kesilmiş 8 düşünce cevap gibi puanlanırdı.
4. **GT modları mantıklı mı?** Doğru cevapların hepsi `strict` (biçim tutmuş, fallback'e gerek kalmamış).
   Bir kaynakta çoğu cevap `failed` ise önce birkaç örneği okuyun: model yanlış mı cevaplıyor, yoksa biçim mi
   tutmuyor? İkincisi bir prompt ya da evaluator sorunudur. Pilotta `failed` çıkanların hepsi gerçekten
   yanlıştı (yukarıdaki GSM8K örneği: model 147 diyor, doğrusu 89).
5. **Kanallar birbiriyle tutarlı mı?** ArmoRM, GT'ye göre doğru olan cevaplara ortalamada daha yüksek skor
   vermeli (0,161 > 0,098). Tersi olsaydı ArmoRM'a giden metinde (ör. düşünce izi) bir sorun olurdu.
6. **Kaynaklar arası farklar beklenen yönde mi?** BBH ve GSM8K kolay, OlympiadBench zor; BARTScore uzun
   cevaplarda (Alpaca, OlympiadBench) daha negatif. Bunlar modelin ve ölçünün bilinen davranışı.

### Pilotun bize söyledikleri

- Skorlama zinciri gerçek modellerle uçtan uca doğru çalışıyor (üç kanal da referansla aynı, bkz.
  [3. Skorlama](03_skorlama.md#referansla-aynı-olduğunu-nasıl-gösterdik)).
- R1-Distill, 4096 token sınırıyla OlympiadBench'te neredeyse hep kesiliyor (8'de 6). Tam koleksiyonda bu
  modelin OlympiadBench skoru büyük ölçüde token sınırını ölçecek. Sınırı reasoning modeli için yükseltmek
  maliyeti ve süreyi artırır; tam koşudan önce verilecek bir ekip kararı.
- 8 soruluk ortalamalar gürültülüdür; pilot bir model karşılaştırması değil, bir doğruluk kontrolüdür.

## Sonraki adım: birleştirme

Bütün modellerin paketleri hazır olunca koordinatör hepsini birleştirir, doğrular ve kalite/maliyet
raporlarını üretir:

```bash
llmr assemble configs/benchmarks/shared_v1.yaml       # frozen/v1/: long, wide, views/<kanal>/
llmr validate-data configs/benchmarks/shared_v1.yaml  # hash'ler, anahtarlar, view = long tutarlılığı
llmr evaluate-data configs/benchmarks/shared_v1.yaml  # her kanal için tek modeller, oracle, rastgele: reports/v1/
```

Bu adımın gerçek bir koşuyla anlatımı, tam koleksiyon yapılınca ayrı bir yazı olarak eklenecek. Kuralları
şimdiden [ekip rehberi §4](../team_workflow.md#4-paketleri-teslim-et-ve-birleştir) ve
[protokol §4](../shared_protocol.md#4-güvenli-birleştirme-ve-sürümleme) anlatıyor.

## İlgili dosyalar

- özet scripti: [`cluster/tosun/inspect_shard.py`](../../cluster/tosun/inspect_shard.py)
- kayıt biçimleri: [protokol §2–3](../shared_protocol.md#2-bağımsız-generation-yalnız-tosun-gpuları)
