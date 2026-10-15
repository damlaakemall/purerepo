# 3. Skorlama: bir cevabın doğru ya da iyi olduğuna nasıl karar veriyoruz

> Router'ın öğrendiği şey "bu model bu soruda ne kadar iyi". O "iyi"yi sayıya çeviren adım bu. Her cevap
> üç ayrı **kalite kanalı** ile puanlanır; kanallar hiç karıştırılmaz.

## Üç kanal

| kanal | soru | ölçek | hangi sorular |
|---|---|---|---|
| **ground truth (GT)** | cevap doğru mu? | 0/1, serbest metinde token F1 0–1 | hepsi (gold'u olan) |
| **ArmoRM** | bir ödül modeli bu cevabı ne kadar beğeniyor? | sürekli, çoğunlukla 0–0.3 | hepsi |
| **BARTScore** | BART bu cevabı bu prompt'tan sonra ne kadar olası buluyor? | negatif sürekli, 0'a yakın iyi | hepsi |

Neden üç tane: farklı makaleler farklı kalite sinyali kullanır. Doğruluk ya da tercih etiketiyle çalışanlar (RouteLLM, GraphRouter...) GT'yi,
BEST-Route ArmoRM'u, Hybrid LLM BARTScore'u kullanır. Hepsini aynı cevaplardan ayrı ayrı topluyoruz; her
router kendi makalesindeki sinyali alır. **Kural:** bir kanalın eksik değeri başka bir kanalla doldurulmaz
(AGENTS.md, kural 4).

Üç kanal da **referans repoyla aynı kodla** hesaplanıyor (aşağıda nasıl doğruladığımız var).

## Ground truth: cevap tipine göre karar

Her soru bir `answer_type` taşır; [`evaluators.py`](../../llm_routing/pipeline/ground_truth/evaluators.py) ona
göre karar verir. Her tipte önce **sıkı (strict)** kural denenir: cevap istenen biçimde mi ve doğru mu.
Olmazsa yalnız şıklı sorularda bir **fallback** düzyazıdan cevabı kurtarmaya çalışır.

| tip | strict | fallback | örnek |
|---|---|---|---|
| `numeric` (GSM8K) | önce açık biçimler ("Final Answer" başlığı, `Answer: n`, `\boxed{}`, `#### n`), bulunamazsa metindeki son sayı; sayı olarak karşılaştırılır (`1,000` = `1000`) | yok | `Answer: 18` ve gold 18 → 1 |
| `mcq_letter` (MMLU-Pro) | açık `Answer: X` sonucu | son satırdaki tek şık harfi | `**Answer: E**`, gold A → 0 |
| `exact_match_text` (BBH) | son `Answer:` satırı, büyük/küçük harf ve boşluk normalize | göreve özgü (şık metni, son satır) | `Answer: (B)` veya `Answer: B`, gold `(B)` → 1 |
| OlympiadBench | `\boxed{}` içi, sympy ile matematiksel eşdeğerlik (`1/2` = `0.5`) | yok (hakem zaten esnek) | |
| `code_tests` (MBPP) | ` ```python ` bloğu ayrı bir süreçte (`python -I`, 5 sn sınır) gold testlerle çalıştırılır | yok | tüm `assert`'ler geçerse 1; sonsuz döngü = 0 |
| `reference_text` (Alpaca, Multi-News) | token F1: küçük harf, noktalama ve artikeller atılır, kelime örtüşmesinin precision/recall harmonik ortalaması | — | 0–1 arası |

Sonuç her cevap için bir `mode` ile kaydedilir: `strict`, `fallback`, `failed` (cevap okundu ama yanlış ya da
okunamadı), `no_answer` (cevap yok), `timeout` (MBPP), `token_f1`. Bu sayede "yanlış cevap" ile "biçim
yüzünden okunamadı" ayrılabiliyor.

Referans repodaki evaluator'lar blind audit'lerle sağlamlaştırılmış; biz onları **birebir kopyaladık**, üç
belgelenmiş düzeltme dışında (DECISIONS.md): "A cache is..." cümlesi artık A şıkkı sayılmıyor; BBH'de gold
`(B)` iken `Answer: B` doğru; MBPP'de süre aşımı scoring hatası değil yanlış cevap.

## ArmoRM

[`RLHFlow/ArmoRM-Llama3-8B-v0.1`](https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1), insan tercihleriyle
eğitilmiş 8B'lik bir ödül modeli. Soru ve cevabı bir sohbet olarak okuyup tek bir skor verir (ve 19 alt boyut
puanı: yardımseverlik, doğruluk, güvenlik...). Referans cevaba ihtiyacı yoktur, bu yüzden her soruda çalışır.
Bağlamı 8192 token; aşan bir girdi kesilmez, hata olarak kaydedilir (sessizce yanlış sayı üretmektense).

## BARTScore

`facebook/bart-large` ile, prompt verildiğinde cevabın token log-olasılıklarının **toplamı**:
`Σ log p(cevap_t | cevap_<t, prompt)`. Referans repo böyle tanımlıyor; Hybrid LLM makalesindeki BARTScore ise
cevabı bir referans cevaba göre puanlar. Toplam olduğu için uzun cevaplar daha düşük (daha negatif) skor alır;
pilotta değerler −120 ile −10.000 arasındaydı. Cevap uzunluğuyla (`completion_tokens`) birlikte okunmalı.

## Düşünme izi

R1-Distill gibi bir reasoning modelinin cevabı `<think> ...uzun düşünce... </think> asıl cevap` biçimindedir.
Üç kanal da **yalnız asıl cevabı** puanlar (`text_policy: strip_think`):

- Düşüncede "belki (D)" yazıp sonunda "(B)" diyen model (D)'den puan almamalı.
- Bin token'lık düşünce, tek cümlelik referansla token F1'i sıfıra yaklaştırır.

Temizleme referanstaki `strip_reasoning` ile aynı: `<think>`, `<thinking>`, `<reasoning>` blokları silinir;
**kapanmamış** bir blok (model 4096 token'da kesildi) cevap yok demektir.

R1-Distill'in özel bir durumu var: `<think>` etiketini model değil chat template açar (prompt
`<｜Assistant｜><think>\n` ile biter). Bu yüzden modelin ürettiği metinde açılış etiketi yoktur. Generation bu
öneki cevabın başına geri koyar ve `metadata.response_prefix` alanına yazar. Bu yapılmasaydı düşünce izi hiç
silinmez ve puanlanırdı.

Geriye hiçbir şey kalmazsa (boş cevap ya da yalnız düşünce): GT **0** (`mode: no_answer`), ArmoRM ve BARTScore
referanstaki gibi ham metni puanlar. Boş metin puanlansaydı BARTScore'un verebileceği en iyi skoru alırdı.

## Referansla aynı olduğunu nasıl gösterdik

1. **Kod karşılaştırması.** `ground_truth/` dosyaları referansla `diff` edildi: yalnız üç belgelenmiş düzeltme
   farklı. ArmoRM ve BARTScore'un formülü, ayarları (bf16, 512 token, chat template, bağlam kontrolü) ve
   düşünce temizleme referansla satır satır karşılaştırıldı.
2. **Sayısal karşılaştırma, gerçek GPU'da.** Pilotta üretilen 60 R1-Distill cevabı, Tosun'da bir GPU job'ında
   referans repo'nun **kendi** `score_single_response` fonksiyonlarıyla tekrar puanlandı ve bizim kayıtlı
   skorlarımızla karşılaştırıldı:

   ```text
   bartscore: n=60 max_rel_diff=0.00e+00
   armorm:    n=60 max_rel_diff=0.00e+00
   ```

   Yani iki repo aynı cevaba bit düzeyinde aynı skoru veriyor.
3. **GT çıktısını elle okuma.** `failed` çıkan her cevap açılıp bakıldı: hepsi gerçekten yanlıştı (GSM8K'da
   147, doğrusu 89; MMLU-Pro'da E, doğrusu A; testi geçmeyen MBPP kodu). Evaluator'ın doğru bir cevabı
   kaçırdığı durum yoktu.

## İlgili dosyalar

- GT: [`llm_routing/pipeline/ground_truth/`](../../llm_routing/pipeline/ground_truth/), giriş noktası `score_query`
- ArmoRM, BARTScore, düşünce temizleme: [`collect.py`](../../llm_routing/pipeline/shared/collect.py) (`ArmoRM`, `BARTScore`, `answer_text`, `score`)
- `<think>` öneki: [`backends.py`](../../llm_routing/pipeline/shared/backends.py) (`_prompt`)
- testler: `tests/test_ground_truth.py`, `tests/test_shared_backends.py`, `tests/test_shared_pipeline.py`
- ayrıntı: [protokol §3](../shared_protocol.md#3-ayrı-scoring-kanalları-ground-truth-armorm-bartscore)
