# 4. Tosun'da çalıştırma: hesaptan skorlanmış cevaplara

> Bir modelin bütün sorulara cevap üretmesi ve bu cevapların üç kanalda puanlanması Tosun'un GPU'larında
> yapılır. Bu yazı sıfırdan başlayıp bir modelin katkı paketini çıkarana kadar olan yolu anlatır; örnek,
> 15 Ekim 2026'daki DeepSeek-R1-Distill-Llama-8B pilotu.

## Tosun nedir, nasıl çalışır

Tosun, Sabancı'nın paylaşımlı GPU kümesi. İş **SLURM** ile yürür: işinizi bir script olarak kuyruğa
verirsiniz (`sbatch`), SLURM uygun bir makinede boş GPU bulunca çalıştırır.

- **Login node** (`tosun.sabanciuniv.edu`): yalnız giriş, dosya düzenleme, iş gönderme. Burada ağır iş
  çalıştırılmaz; bellek sınırı vardır (büyük bir `pip install` burada öldürüldü).
- **Compute node'lar** (`cn02`...`cn21`, `gpu01`): işler burada koşar. İnternet erişimleri var (model ve
  veri indirme çalışıyor).
- **Partition / QOS:** bizim GPU işlerimiz `cuda`. Hesabımızda ayrıca `short_mdbf`, `mid_mdbf`, `long_mdbf`,
  `longer_mdbf` var. `cuda` partition'ında süre sınırı 15 gün.
- **Disk:** home `/cta/users/<kullanıcı>/`, herkesin paylaştığı `/cta` diskinde. Ekim 2026'da %96 doluydu
  (3,2 TB boş); 12 modelin ağırlıkları ~300 GB tutar, gereksiz kopya bırakmayın.

GPU'lar (`sinfo -p cuda -N -o "%10N %40G"`, Ekim 2026):

| node | GPU | bizim için |
|---|---|---|
| cn02, cn03, cn21 | RTX PRO 6000 96 GB (cn21'de 2 tane) | her model |
| cn08 | A100 80 GB | her model |
| cn18 | H200 NVL 141 GB + 2 × RTX 4500 24 GB | her model / ≤ 8B |
| cn07 | 2 × RTX A5000 24 GB | ≤ 8B |
| cn14 | A30 24 GB + A4000 16 GB | **hariç**: iş A4000'e düşerse 8B sığmaz |
| cn04, cn11, gpu01 | V100 | **hariç**: bf16 yok |
| cn05, cn06 | K80 | **hariç**: bf16 yok, çok eski |

## Bir kez: erişim

1. Hesap SUTicket ile istenir (FENS → HPC); kullanıcı adı ve ilk şifre e-postayla gelir. İlk girişte
   `passwd` ile şifre değiştirilir.
2. Şifre yerine SSH anahtarı kullanıyoruz; anahtar bir kez kendi şifrenizle yüklenir:

   ```bash
   ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_tosun
   ssh-copy-id -i ~/.ssh/id_ed25519_tosun.pub <kullanıcı>@tosun.sabanciuniv.edu
   ssh -i ~/.ssh/id_ed25519_tosun <kullanıcı>@tosun.sabanciuniv.edu
   ```

   Kampüs dışından bağlanılamazsa önce `ssh <kullanıcı>@flow.sabanciuniv.edu` üzerinden geçin
   ([SU HPC rehberi](https://su-hpc-tutorials.readthedocs.io)).

## Bir kez: Python ortamı

Repo `~/purerepo` altına klonlanır (veya laptop'tan `rsync` ile kopyalanır). Ortam bir compute node'da
kurulur, çünkü login node'un bellek sınırı torch'u kurarken pip'i öldürür:

```bash
cd ~/purerepo && mkdir -p logs
sbatch -p cuda --qos=cuda -c 8 --mem=32G -t 2:00:00 -o logs/setup_env-%j.out --wrap "bash cluster/tosun/setup_env.sh"
```

[`setup_env.sh`](../../cluster/tosun/setup_env.sh) ne kuruyor ve neden sabit sürümler:

- Tosun'un GPU sürücüsü **570** (CUDA 12.x). PyPI'deki en yeni vLLM ve torch **CUDA 13** için derlenmiş;
  bu sürücüde GPU'yu görmezler. Bu yüzden vLLM'in `+cu129` sürümünü ve torch'u PyTorch'un `cu129` index'inden
  kuruyoruz (CUDA 12.x sürücüsü 12.9 ile derlenmiş kodu çalıştırabilir). Sürümler sabit, herkes aynı yığını
  kullanır: torch 2.13.0+cu129, vLLM 0.31.0+cu129.
- `torchcodec`'in PyPI sürümü de CUDA 13'e bağlı ve `import vllm`'i bozuyor; metin modelleri için aynı sürümün
  CPU derlemesi yeterli.

Her job [`config.sh`](../../cluster/tosun/config.sh)'ı yükler: `python/3.12.9` ve `cuda/12.8` modülleri,
venv, veri ve model cache yolları, varsayılan partition ve hariç tutulan node'lar (bf16'sız V100/K80 ve cn14). Bir ince ayar daha var:
`python/3.12.9` modülünün `sqlite3`'ü bozuk (2020'den kalma bir kütüphaneye bağlı), vLLM de sqlite3 import
ediyor; `config.sh` `sqlite` modülünün yeni kütüphanesini önceden yükler.

Gated modeller için bir kez `hf auth login` (ve huggingface.co'da lisans kabulü).

GPU'nun görüldüğünü kontrol etmek için kısa bir iş:

```bash
sbatch -p cuda --qos=cuda -c 2 --mem=8G -t 10 --gres=gpu:1 -o logs/gpucheck-%j.out \
  --wrap "source cluster/tosun/config.sh; llmr_activate; llmr_require_gpu"
# beklenen: torch 2.13.0+cu129 (CUDA 12.9) -> 1 x NVIDIA H200 NVL
```

## Önce pilot

Rehberin ve bizim kuralımız: büyük iş göndermeden önce küçük bir deneme. Pilot config
[`shared_pilot.yaml`](../../configs/benchmarks/shared_pilot.yaml), `shared_v1` ile aynı kaynaklar ve
ayarlardan 30 soru, k = 2. Gerçek modeli, gerçek ArmoRM'u ve gerçek BART'ı küçük ölçekte çalıştırır;
kurulum, indirme ve skorlama hataları dakikalar içinde ortaya çıkar.

## Bir modeli koleksiyona eklemek

Sorular bir kez donar ([1. Soru seti](01_soru_seti.md)), sonra her üye kendisine atanan modeli çalıştırır:

```bash
cd ~/purerepo
bash cluster/tosun/submit_member.sh configs/benchmarks/shared_pilot.yaml deepseek-r1-distill-llama-8b ibrahim 1
# Collection submitted: generation=4042873 (1 GPU) scoring[ground_truth,armorm,bartscore]=4042874 4042875 4042876 package=4042877
```

Argümanlar: config, model adı (havuzdaki), üye adı, shard sayısı. Tam koleksiyonda büyük modeller için shard
sayısını artırın (ör. 4): sorular 4 parçaya bölünür, 4 GPU işi paralel koşar.

[`submit_member.sh`](../../cluster/tosun/submit_member.sh) beş işi bağımlılıklarıyla birlikte kuyruğa verir:

```text
generate (GPU) ──┬──► ground_truth (CPU, MBPP kodunu ayrı süreçte çalıştırır) ──┐
                 ├──► armorm (GPU, 8B ödül modeli)                              ├──► package
                 └──► bartscore (GPU, bart-large)                               ┘
```

- **generate:** vLLM modeli yükler, her (soru, örnek) için kendi seed'iyle bir istek oluşturur, 256'lık
  gruplar halinde üretir, her cevabı `responses.jsonl`'a bir satır olarak ekler.
- **üç skor işi** generation bitince paralel başlar; her biri kendi `<kanal>.jsonl` dosyasını yazar.
- **package** bu dosyaları hash'leriyle `exports/` altına değişmez bir paket olarak kopyalar. Koordinatör
  bütün paketleri birleştirir (sonraki adım).

İş yarıda kalırsa (süre sınırı, düğüm hatası) aynı komutu tekrar gönderin: biten örnekler atlanır, kalanlar
üretilir. Config'i değiştirip eski işe devam edilemez; bu bilinçli bir korumadır.

## İzleme

```bash
squeue -u $USER                                   # bekleyen/çalışan işler; REASON sütunu nedenini söyler
tail -f logs/llmr-member-<jobid>_0.out             # bir işin logu
grep '^\[generate\]' logs/llmr-member-<jobid>_0.out  # üretim ilerlemesi: "60/60 samples"
```

`squeue`'da sık görülen nedenler: `Resources` (uygun GPU boş değil, bekler), `Dependency` (önceki iş bitmedi),
`DependencyNeverSatisfied` (önceki iş başarısız oldu; logunu okuyun, kalan işleri `scancel` edin, düzeltip
yeniden gönderin), `AssocGrpCpuLimit` (grubun o partition'daki CPU kotası dolu; `cuda` partition'ını kullanın).

## Örnek: R1-Distill pilotu (15 Ekim 2026)

Pilot o gün aynı içerikli `pilot_tosun` adlı bir config ile koştu; repodaki `shared_pilot.yaml` onun kopyası
(yalnız adı ve yolu farklı).

| iş | node / GPU | süre |
|---|---|---|
| generate (60 cevap) | cn18, H200 | ~7 dk (16 GB model indirme ve yükleme dahil) |
| ground_truth | cn07, CPU | saniyeler |
| armorm | cn18, H200 | ~4 dk (ArmoRM indirme dahil) |
| bartscore | cn18, H200 | ~1 dk |
| package | cn07 | saniyeler |

Model cache'i bu pilottan sonra 31 GB idi (R1-Distill 8B, ArmoRM 8B, bart-large, MiniLM). Çıktılar:

```text
~/llmr_data/benchmarks/shared_pilot/
  dataset/                                   donmuş sorular ve split'ler
  contributions/ibrahim/deepseek-r1-distill-llama-8b/shard-00000-of-00001/
    run.json  responses.jsonl  ground_truth.jsonl  armorm.jsonl  bartscore.jsonl
  exports/ibrahim-deepseek-r1-distill-llama-8b-00000-of-00001/   teslim edilecek paket (+ manifest.json)
```

Bu dosyaların nasıl okunduğu ve kontrol edildiği: [5. Sonuçları okuma](05_sonuclari_okuma.md).

## Bilinen sorunlar

| belirti | sebep | çözüm |
|---|---|---|
| `pip install` login node'da `Killed` | login node bellek sınırı | kurulumu `sbatch` ile compute node'da yapın |
| `torch ... sees no GPU` | torch/vLLM CUDA 13 için derlenmiş, sürücü 570 | `setup_env.sh` (cu129 sürümleri) ile yeniden kurun |
| `libtorchcodec_image.so` yüklenemiyor | torchcodec PyPI sürümü CUDA 13 arıyor | `setup_env.sh` CPU torchcodec'i kurar |
| `undefined symbol: sqlite3_deserialize` | `python/3.12.9` modülünün eski sqlite'ı | `config.sh` yeni sqlite'ı önceden yükler; job'larda `llmr_activate` kullanın |
| `Bfloat16 is only supported on GPUs with compute capability ≥ 8.0` | V100/K80'e düştü | bu node'lar `LLMR_EXCLUDE` ile hariç |
| 8B model OOM | 16 GB'lık karta düştü (cn14 A4000) | cn14 varsayılan olarak hariç; `LLMR_EXCLUDE`'u daraltmayın |
| 403 indirme hatası | gated model, lisans kabul edilmemiş veya `hf auth login` yok | lisansı kabul edip giriş yapın |
| `CUDA 12.8 module can only be loaded on GPU-equipped servers` | login node'da uyarı | zararsız; işler GPU node'da modülü yükler |

## İlgili dosyalar

- scriptler: [`cluster/tosun/`](../../cluster/tosun/) (`config.sh`, `setup_env.sh`, `submit_member.sh`, `jobs/shared_member.sbatch`)
- generation ve skorlama: [`collect.py`](../../llm_routing/pipeline/shared/collect.py), [`backends.py`](../../llm_routing/pipeline/shared/backends.py)
- ayrıntı: [Tosun README](../../cluster/tosun/README.md), [ekip rehberi §3](../team_workflow.md#3-üye-sadece-atandığın-modeli-üret)
