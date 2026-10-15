# 2. Model havuzu: hangi 12 model, neden, maliyeti nasıl sayıyoruz

> Router'ın seçeceği modeller. Her soru her modelden k = 10 kez cevaplanır; router'lar "hangi model bu
> soruyu ucuza ve iyi cevaplar" sorusunu bu cevaplardan öğrenir.

## Ne yapıyoruz

[`configs/model_pools/team.yaml`](../../configs/model_pools/team.yaml) 12 açık ağırlıklı modeli listeler.
Her kayıtta Hugging Face kimliği, sabit commit (revision), maliyet katsayısı, gerekirse GPU sayısı ve chat
template ayarı var.

| grup | model | parametre (B) | not |
|---|---|---|---|
| küçük | SmolLM3 3B | 3.1 | düşünme modu kapalı (`enable_thinking: false`) |
| | Llama 3.2 3B Instruct | 3.2 | gated |
| | Phi-4-mini Instruct | 3.8 | |
| orta | Mistral 7B Instruct v0.3 | 7.2 | |
| | OLMo 3 7B Instruct | 7.3 | tamamen açık (veri + kod) |
| | Qwen2.5 7B Instruct | 7.6 | |
| | Llama 3.1 8B Instruct | 8.0 | gated |
| | DeepSeek-R1-Distill-Llama 8B | 8.0 | **reasoning modeli**: her zaman `<think>` ile düşünür |
| büyük | Phi-4 | 14.7 | bağlamı 16k |
| | Mistral Small 24B Instruct | 23.6 | |
| | Gemma 3 27B IT | 27.4 | gated, multimodal checkpoint, yalnız metin |
| | Qwen2.5 32B Instruct | 32.8 | |

Generation ayarları bütün modeller için aynı (`shared_v1.yaml`): `temperature 0.7`, `top_p 0.95`,
`max_new_tokens 4096`, soru başına `k = 10` örnek. Tam koleksiyon 7652 soru × 12 model × 10 örnek ≈ 918 bin
cevap demek.

## Neden bu modeller

1. **Referanstan tamamen farklı.** Referans repo Qwen3, Gemma 4, Cogito, R1-Distill-Qwen, Granite, Kimi,
   GLM, GPT kullanıyor. Aynı soru ve protokolü farklı bir havuzla tekrarlamak, router sonuçlarının belirli
   bir model setine bağlı olup olmadığını gösterir.
2. **Gerçek maliyet/kalite yayılımı (3B → 33B).** Routing'in anlamı, ucuz modelin yettiği soruyu ucuza
   göndermek. Havuzda yalnız benzer boyutlu modeller olsaydı yönlendirmenin kazandıracağı bir şey olmazdı.
   3 küçük, 5 orta, 4 büyük model ile her bütçe düzeyinde seçenek var.
3. **Farklı aileler ve eğitim tarifleri.** Llama, Qwen, Mistral, Phi, Gemma, OLMo, SmolLM, DeepSeek. Aynı
   boyutta iki model farklı konularda güçlü olabilir (ör. Qwen2.5 matematikte, Llama talimat takibinde);
   router'ın kullanabileceği bir sinyal.
4. **Bir reasoning modeli.** R1-Distill uzun düşünür, daha çok token harcar (daha pahalı) ama zor
   matematikte daha iyi olabilir. "Düşünen model ne zaman değer?" sorusu routing için ilginç.
5. **Tosun'da çalışabilir, ücretsiz.** Hepsi açık ağırlıklı ve vLLM'de destekleniyor; en büyüğü (32B, bf16
   ~66 GB) tek bir 80/96 GB'lık karta sığıyor. Ücretli API yok (AGENTS.md, kural 3).

## İki özel durum

- **R1-Distill her zaman düşünür.** Chat template'i prompt'u `<think>\n` ile bitirir; model düşünür,
  `</think>` yazar, sonra cevap verir. Skorlamada yalnız cevap kısmı puanlanır ([3. Skorlama](03_skorlama.md#düşünme-izi)).
  Düşünme 4096 token'ı aşarsa cevap hiç yazılmamış olur ve 0 alır; pilotta OlympiadBench'in 8 örneğinden 6'sı
  böyle kesildi.
- **SmolLM3 varsayılan olarak düşünür**; referanstaki gibi `enable_thinking: false` ile kapatıyoruz.
  Böylece havuzda tek bir bilinçli reasoning modeli kalıyor.

## Maliyeti nasıl sayıyoruz

Kendi GPU'larımızda çalıştırdığımız için bir dolar fiyatı yok. Maliyet bir **vekil** (proxy):

```text
maliyet = parametre sayısı (milyar) × (prompt token + cevap token) / 1000
```

Gerekçe: dense bir modelde token başına hesap yaklaşık parametre sayısıyla orantılıdır; 32B'lik model bir
token için 3B'lik modelin ~10 katı iş yapar. Uzun düşünen model de cevap token'larıyla daha pahalı çıkar.
Bu bir **USD değildir** ve öyle raporlanmaz (AGENTS.md, kural 5). Her cevap ayrıca ölçülen süreyi
(`batch_seconds_per_sample`) kaydeder; ölçülen GPU süresine geçip geçmeme kararı açık (DECISIONS.md #2).

## Neyi sabitledik

Her model, tokenizer ve chat template bir Hugging Face commit'ine bağlı (`revision`). Model sahibi
repoyu güncellese bile herkes aynı ağırlıkları ve aynı template'i kullanır. 12 commit'in hepsinin
Hugging Face'te var olduğu doğrulandı (2026-10-15).

**Gated modeller** (Llama ×2, Gemma): her üye huggingface.co'da lisansı kendi hesabıyla kabul eder ve Tosun'da
`hf auth login` yapar; yoksa indirme 403 verir.

## Hangi GPU, kaç GPU

bf16 ağırlık ≈ 2 GB × milyar parametre, üstüne KV cache. Tosun'daki kartlarla (Ekim 2026):

| model boyutu | ağırlık | uyan kartlar |
|---|---|---|
| 3–8B | 6–16 GB | 24 GB ve üstü (A30, A5000, RTX 4500, A100, RTX PRO 6000, H200) |
| 15B (Phi-4) | ~30 GB | 40 GB ve üstü (A100 80 GB, RTX PRO 6000 96 GB, H200) |
| 24–33B | 47–66 GB | A100 80 GB, RTX PRO 6000 96 GB, H200 141 GB (tek kart) |

V100 ve K80'lerde bf16 yok; 16 GB'lık A4000 (cn14) 8B modelleri taşımaz. Üçü de varsayılan olarak hariç
(`cluster/tosun/config.sh`). Büyük modeller için `gpus: 2` mi yoksa tek büyük kart mı kararı açık
(DECISIONS.md #6): aynı tipte iki büyük kartı olan tek node cn21.

## İlgili dosyalar

- havuz: [`configs/model_pools/team.yaml`](../../configs/model_pools/team.yaml)
- generation kodu: [`backends.py`](../../llm_routing/pipeline/shared/backends.py), [`collect.py`](../../llm_routing/pipeline/shared/collect.py)
- maliyet: [`assemble.py`](../../llm_routing/pipeline/shared/assemble.py) (`_cost`)
- kararlar: [DECISIONS.md](../../DECISIONS.md)
