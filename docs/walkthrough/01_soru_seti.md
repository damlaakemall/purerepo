# 1. Soru seti: hangi testler, neyi ölçüyor, neden bunlar

> Router'lar "bu soruyu hangi model cevaplasın?" kararını öğrenir. Bunun için modellerin gerçekten
> **farklı** başardığı, cevabı **kontrol edilebilir** sorulara ihtiyaç var. Bu adım o soruları bir kez seçer,
> biçimlendirir, train/val/test'e böler ve dondurur. Sonraki her adım bu donmuş soruları kullanır.

## Ne yapıyoruz

[`configs/benchmarks/shared_v1.yaml`](../../configs/benchmarks/shared_v1.yaml) yedi kaynak listeler.
`llmr build-queries` her kaynağı sabitlenmiş Hugging Face commit'inden indirir, seed 42 ile karıştırıp
`n` örnek alır, her örneği [`formatters.py`](../../llm_routing/pipeline/shared/formatters.py) ile bir
prompt + doğru cevap (gold) + cevap tipine çevirir ve `dataset/queries.jsonl` olarak dondurur.

| kaynak | ne ölçüyor | örnek sayısı | cevap tipi | nasıl puanlanıyor |
|---|---|---|---|---|
| GSM8K | ilkokul düzeyi çok adımlı aritmetik problem | 1319 (test'in tamamı) | `numeric` | sayı doğru mu, 0/1 |
| MMLU-Pro | 14 alanda bilgi + akıl yürütme, 10 şıklı | 1500 (12032'den) | `mcq_letter` | şık harfi doğru mu, 0/1 |
| BBH | 27 zor akıl yürütme görevi (mantık, tarih hesabı, sayma, nesne takibi...) | 27 görev × 80 = 2160 | `exact_match_text` | cevap metni/şık aynı mı, 0/1 |
| OlympiadBench | yarışma matematiği, açık uçlu, İngilizce, yalnız metin | 674 (tamamı) | `exact_match_text` | matematiksel eşdeğerlik (sympy), 0/1 |
| MBPP | kısa Python fonksiyonu yazma | 500 (test'in tamamı) | `code_tests` | kod testleri geçiyor mu, 0/1 |
| Alpaca | genel talimat takibi ("şu e-postayı yaz", "şunu açıkla") | 1000 (52k'dan) | `reference_text` | referans cevapla token F1, 0–1 |
| Multi-News | çok makaleli haber özetleme, uzun girdi | 500 (5622'den) | `reference_text` | referans özetle token F1, 0–1 |

Toplam 7652 soru (Alpaca'daki birebir tekrarlar ve çok uzun Multi-News örnekleri çıktıktan sonra).

## Neden bu yedi kaynak

1. **Referansla aynı karışım.** SU Lab LLM-Routing-Project bu yedi kaynağı, bu sayılarla kullanıyor.
   Modellerimiz farklı ama sorular ve protokol aynı olduğu için sonuçlarımız onlarınkiyle yan yana
   okunabiliyor (DECISIONS.md, 2026-10-15).
2. **Zorluk yelpazesi.** BBH ve GSM8K'yı 3B'lik bir model bile sıkça çözer, OlympiadBench'i 32B'lik model
   bile zor çözer. Router'ın öğrenecek bir şeyi olması için "küçük model yeter" ve "büyük model şart"
   sorularının ikisinin de bol olması gerekiyor. Tek tip zorlukta her soruya aynı model gider.
3. **Beceri yelpazesi.** Aritmetik, bilgi, mantık, ileri matematik, kod, serbest yazı, uzun metin özeti.
   Modeller bu becerilerde farklı güçlü (bir kod modeli matematikte zayıf olabilir), router'ın kullanacağı
   sinyal tam olarak bu farklılık.
4. **Kontrol edilebilir cevap.** Beş kaynakta kesin bir doğru cevap var, bir programla 0/1 karar
   verilebiliyor; insan ya da başka bir LLM hakemi gerekmiyor (ücretli API kullanmıyoruz). Alpaca ve
   Multi-News'te kesin cevap yok; bunlar serbest metin kalitesi için ve ayrıca ArmoRM/BARTScore ile de
   ölçülüyor ([3. Skorlama](03_skorlama.md)).

## Prompt'lar nasıl yazıldı

Her prompt, modelden cevabı **evaluator'ın ilk baktığı biçimde** vermesini ister. Böylece cevap çoğunlukla
sıkı (strict) kuralla okunur, düzyazıdan cevap çıkarmaya (fallback) nadiren gerek kalır. Örnek, GSM8K:

```text
Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins ...

Solve the problem step by step. End your response with a final line of the form 'Answer: <number>'.
```

| kaynak | istenen biçim |
|---|---|
| GSM8K | son satır `Answer: <sayı>` |
| MMLU-Pro | şıklar `(A) ... (J)`, son satır `Answer: X` |
| BBH | son satır `Answer: <cevap>`, şıklı görevde `Answer: (B)` |
| OlympiadBench | son cevap `\boxed{}` içinde; birim varsa kutunun dışında |
| MBPP | ilk test örnek olarak verilir, tek bir ` ```python ` bloğu |
| Alpaca, Multi-News | talimatın kendisi; biçim istenmez |

Prompt'lar modelin chat template'iyle sarılır (ör. Llama'da `<|start_header_id|>user...`). System prompt
kullanmıyoruz; her model aynı kullanıcı mesajını görür.

## Train / val / test bölmesi

Bölme generation'dan **önce** yapılır ve dondurulur; sonra hiçbir sonuç bölmeyi etkileyemez. İki adım
([`splits.py`](../../llm_routing/pipeline/shared/splits.py)):

1. **Neredeyse aynı sorular birlikte kalır.** Bütün sorular MiniLM ile vektöre çevrilir; benzerliği ≥ 0,99
   olanlar bir grup olur ve bir grup tek split'e düşer. Yoksa bir sorunun ikizi train'de, kendisi test'te
   olur ve router ezberle iyi görünür. Gerçek veride 753 soru gruplandı, 719'u BBH'nin şablonlu
   görevlerinden.
2. **Her kaynak kendi içinde 60/10/30.** Böylece test'te her kaynaktan aynı oranda soru var; bir kaynağın
   tamamen train'e düşmesi gibi bir şans olmaz.

**Kural:** router'lar yalnız `train` ile eğitilir, eşik ve hiperparametre `val` ile seçilir; `test` yalnız
son raporda kullanılır (AGENTS.md, kural 1).

## Nasıl çalıştırdık

```bash
# Tosun'da, compute node'da (datasets + MiniLM; login node'da bellek sınırı var)
sbatch -p cuda --qos=cuda -c 4 --mem=16G -t 30 -o logs/build-%j.out \
  --wrap "source cluster/tosun/config.sh; llmr_activate; llmr build-queries configs/benchmarks/shared_v1.yaml"
```

Pilot için aynı kaynaklardan küçük bir kopya kullandık:
[`shared_pilot.yaml`](../../configs/benchmarks/shared_pilot.yaml) (kaynak başına 4 soru, BBH'den 3 görev × 2,
k = 2; toplam 30 soru). Çıktı:

```text
[build-queries] 30 frozen queries -> .../benchmarks/shared_pilot/dataset; near-duplicate groups: 0 (0 prompts)
```

## Nasıl kontrol ettik

- Aynı config ile `build-queries` tekrar çalışınca yeni dataset yazmaz, mevcut olanı hash ile doğrular.
  Config değiştiyse hata verir: yeni bir sürüm (`shared_v2`) gerekir (AGENTS.md, kural 2).
- Bütün kaynak commit'lerinin Hugging Face'te var olduğu doğrulandı (2026-10-15).
- `tests/test_splits.py`: grupların bölünmediği ve oranların tuttuğu test ediliyor.

## İlgili dosyalar

- config: [`configs/benchmarks/shared_v1.yaml`](../../configs/benchmarks/shared_v1.yaml)
- kod: [`formatters.py`](../../llm_routing/pipeline/shared/formatters.py),
  [`dataset.py`](../../llm_routing/pipeline/shared/dataset.py), [`splits.py`](../../llm_routing/pipeline/shared/splits.py)
- ayrıntı: [protokol §1](../shared_protocol.md#1-dataset-ve-kimlik)
