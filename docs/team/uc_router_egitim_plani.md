# Hybrid LLM, GraphRouter ve UniRoute: kendi verimizle eğitim planı

**19 Ekim 2026 — araştırma ve uygulama önerisi.** Makaleler, resmi GraphRouter kodu,
SU Lab referans reposu ve bu reponun eğitim akışı incelendi. Bu belge bir implementasyon
veya tamamlanmış eğitim raporu değildir; aşağıdaki yeni config/dosya adları öneridir.
Gerçek eğitim bitince sonuçlar tek bir `docs/walkthrough/02_...md` yazısında,
her router ayrı bölüm olacak şekilde anlatılmalı.

## 1. Sonuç: aynı cevapları kullanabiliriz, deneyleri ayrı tanımlamalıyız

Eğitilecek olan **router**; havuzdaki 12 cevaplayıcı LLM yeniden eğitilmiyor.
Topladığımız cevaplar ve skorlar router'a hangi soruda hangi modelin uygun olduğunu
öğretiyor. Üç yöntem için de yeni cevap üretmek gerekmiyor: tamamlanmış ve doğrulanmış
bir ortak release yeterli. Ek iş; makaleye özgü label, embedding ve eğitimdir.

Yeni ana deneyler için öneri **`shared_v2`**. README ve DECISIONS'a göre v2 hazırlanmış,
gerçek GPU pilotu ve tam koleksiyonu bekliyor. Yerel `data/benchmarks` altında incelenen
manifestler sentetik/smoke verisine ait; burada tamamlanmış gerçek v2 release veya bu
üç router'ın kendi havuzumuzdaki sonuçları bulunmadı. Tosun'daki güncel veri durumu bu
incelemede doğrulanmadı. Mevcut v1 verisiyle de ayrı deney yapılabilir; sürümler birleşmez.

| yöntem | ortak veriden öğrenilecek şey | önerilen sınıf | ana deney |
|---|---|---|---|
| Hybrid LLM, ICLR 2024 | küçük modelin büyük modele yetişme olasılığı | `BinaryRouter` | önceden sabitlenmiş iki model |
| GraphRouter, ICLR 2025 | sorgu–model kenarının seçilmeye uygunluğu | doğrudan `Router` | 12 model, üç ayrı maliyet/kalite senaryosu |
| UniRoute, ICLR 2026 | modelin sorgu kümeleri üzerindeki hata profili | `MultiModelRouter` veya doğrudan `Router` | 6 öğrenme modeli + 6 yeni model |

Hybrid'in iki modeli, GraphRouter'ın 12 modeli ve UniRoute'un yeni altı modeli aynı
oracle'a karşı sıralanamaz. İkili karşılaştırma için diğer router'ları da aynı çifte
daraltan ek deney; dinamik havuz karşılaştırması için aynı yeni model alt kümesini ve
aynı tanıtım verisini kullanan ek deney gerekir. Mevcut leaderboard model listesini
zaten grupluyor; **bilgiye erişim protokolü** de ayrıca kaydedilip gruplamaya eklenmeli.

## 2. İncelenen kaynaklar ve referans kodunun sınırları

Ana yöntem kaynağı kullanıcının verdiği **ICLR konferans PDF'leri**:

- [Hybrid LLM, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/b47d93c99fa22ac0b377578af0a1f63a-Paper-Conference.pdf): §3.1–3.3, Denklem 1–3; §4 ve Ek A.2.
- [GraphRouter, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/41b6674c28a9b93ec8d22a53ca25bc3b-Paper-Conference.pdf): §2.3, Algoritma 1, §3.4.
- [UniRoute, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/file/113ae3a9762ca2168f860a8501d6ae25-Paper-Conference.pdf): §5.1–5.2, Denklem 11–12; Ek F.1–F.3 ve G.2.

Kod incelemesinin sabit noktaları:

| kaynak | incelenen sürüm | nasıl kullanılmalı |
|---|---|---|
| [SU Lab referans repo](https://github.com/SU-Intelligent-systems-Lab/LLM-Routing-Project) | yerel temiz checkout, `e6578414e246469073867d3a8d0c21c5b3e93f92` | `routers/graphrouter/{adapter,gnn,router}.py`, `routers/uniroute/{adapter,core,router}.py`, `docs/routers.md` incelendi; adapter fikirleri alınabilir, yöntem ve sızıntı kontrolleri yeniden yapılmalı |
| [resmi GraphRouter](https://github.com/ulab-uiuc/GraphRouter/tree/0edfb8758d45acd93c8029354867ffb49bb8a97f) | `0edfb8758d45acd93c8029354867ffb49bb8a97f` | ağ ve maskeli eğitim için esas kod referansı |
| Hybrid'in makalede verilen `m365-core/hybrid_llm_routing` adresi | GitHub API kontrolü 404 döndü | erişilebilir resmi implementasyon doğrulanamadı; üçüncü taraf bir MLP'yi orijinal yöntem saymayacağız |
| UniRoute | incelenen makale/yazar sayfalarında erişilebilir resmi implementasyon bulunamadı | SU Lab yeniden implementasyonu yardımcı; ICLR final metni esas |

SU Lab reposu yukarıdaki commit'te yerel bir checkout'tan okundu. Uzaktan erişim 404 verdi; bu, reponun yok olduğunu kanıtlamaz. Referansın
`routers/` dizininde Hybrid LLM implementasyonu yok. Bizdeki önceki Hybrid de
DECISIONS'a göre kaldırılmış; yeni paketi baştan yazmalıyız.

Özellikle şu davranışları taşımamalıyız:

| gözlem | bizim uyarlamamız |
|---|---|
| SU Lab GraphRouter adapter'ı eksik effect'i `0` yapıyor | eksiklik yanlış cevap değildir; gözlem maskesi/complete-case cohort kullanılmalı |
| resmi GraphRouter `split_data()` eşit büyüklükte ardışık task blokları varsayıyor | bizim donmuş query-ID split'lerimiz kullanılmalı |
| resmi GraphRouter eğitim döngüsü her epoch test sonucunu hesaplıyor | yalnız val ile checkpoint seçilmeli; test değerlendirmesi eğitimden sonra |
| resmi/SU Lab graph tensörleri test sorgularını da içeriyor; ağ BatchNorm kullanıyor | eğitim grafiğine test düğümü eklenmemeli; etiketsiz test özellikleri dahi eğitim istatistiklerini etkileyebilir |
| SU Lab UniRoute maliyetleri bütün modellerin train kayıtlarından hesaplıyor | yeni modellerin maliyeti de yalnız izin verilen tanıtım split'inden tahmin edilmeli |
| SU Lab UniRoute `predict()` boş val durumunda train'e dönüyor | dinamik deneyde val yoksa açık hata; yeni modelin train etiketlerine dönüş yasak |
| SU Lab UniRoute kNN baseline'ı yeni modellerin train hata etiketlerini kullanıyor | dinamik kNN komşuluk bankası val olmalı; ayrıcalıklı baseline ancak açıkça ayrı raporlanabilir |

Bunlar statik kod incelemesi bulguları; referans eğitimleri bu incelemede çalıştırılmadı.

## 3. Verinin ortak kullanımı

Mevcut `Benchmark` yeterli temel alanlara sahip:

- `queries`, `query_ids`, `split_of`, `sources`;
- `scores[N, M, k]`: seçilmiş **tek** kalite kanalı;
- `costs[N, M]`: sample-0 gerçekleşmiş maliyeti;
- `extras/sample_cost`, `completion_tokens` ve diğer kalite kanalları.

Ana adil karşılaştırma önerisi: üçünü de `ground_truth` görünümünde eğitip aynı
görünümün **sample 0** skoruyla değerlendirmek. Bu, Hybrid'in orijinal BARTScore
deneyinden bir veri/metric uyarlamasıdır. `armorm` ve `bartscore` ayrı deneylerdir;
GT boşluğunu bunlarla doldurmayız. GraphRouter ve UniRoute için ilk sürümde sample-0
supervision kullanılır; k ortalaması veya en iyi sample ayrı deney olur. Hybrid'in
probabilistic hedefi ise k örneği özellikle kullanır. Hiçbiri testte best-of-10 alamaz.

GT'nin beş kaynağı 0/1, Alpaca ve Multi-News sürekli token F1. Ana sonuçla birlikte
kaynak bazında eğri ve eşit kaynak ağırlıklı macro kalite verilmeli. UniRoute için
beş 0/1 kaynakta ayrıca accuracy sonucu; tüm kaynaklarda `loss = 1 - GT` sürekli
uyarlama sonucu raporlanmalı. F1'ı keyfi bir eşikle doğruluk etiketine çevirmeyelim.

Release değişmez. Router label'ları ve embedding cache'leri `runs/` altında;
benchmark fingerprint'i, query-ID listesi, model sırası, encoder revision'ı,
pool seed, eğitim seed'i ve label politikasının hash'iyle saklanır. Üç yönteme ortak
encoder dayatılmaz. Sabit encoder çıktıları kendi router'ı içinde tekrar kullanılabilir;
test cache'i ancak eğitim bittikten sonra üretilir.

## 4. Hybrid LLM: önce tamamlanması gereken en küçük router

### Yöntemi koruyan uygulama

`llm_routing/routers/hybrid_llm/labels.py` üç varyantı üretmeli:

```text
Δ_j(x) = q_small,j(x) - q_large,j(x)
r_det:   y(x) = 1[Δ_0(x) >= 0]
r_prob:  y(x) = mean_j 1[Δ_j(x) >= 0]
r_trans: y(x;t) = mean_j 1[Δ_j(x) >= -t]
```

İlk örnekleme politikası önerisi: iki modelde aynı sample indeksindeki sonlu çiftleri
kullanmak. Makale §3.2 on örnekle ortalama tarif ediyor; bütün 10×10 karşılaştırmaları
zorunlu kılmıyor. Çapraz çiftlerin tamamını kullanmak istenirse ayrı, belgelenmiş
estimator seçeneği olmalı. NaN çiftler paydaya girmez; hiç geçerli çifti olmayan
soru eğitimden çıkarılır ve sayısı kaydedilir. §3.3'te `>`/`>=` yazımı tutarsız olduğu
için sınırda `H >= -t` tanımını izleme tercihi açıkça belgelenmeli; özellikle 0/1 GT'de
eşitlik sık görülür.

`t*`, Denklem 3'ün train üzerindeki ortalama ikili label uzaklığını büyüten değeri.
Bunu yalnız sınıf oranını %50'ye yaklaştırmakla değiştirmeyelim. Aday t değerleri
train farklarından türetilip kaydedilebilir; eşit optimumda küçük t seçilir.

**0/1 GT'de `r_trans`, `r_prob`'a indirgenir.** Beş 0/1 kaynakta `Δ_j ∈ {-1, 0, 1}`
olur. `0 <= t < 1` için `Δ_j >= -t` ile `Δ_j >= 0` aynı label'ı verir. `t >= 1`
olunca bütün label'lar 1 olur ve Denklem 3'ün amacı 0'a düşer. Yani bu kaynaklarda
`r_trans` label'ı `r_prob` label'ıyla aynıdır. `t*`'ı ve iki varyant arasındaki farkı
yalnız sürekli token F1 olan Alpaca ve Multi-News belirler. Makale `r_trans`'ı büyük
kalite farkında dengesiz BARTScore label'ları için önerdi; GT ana deneyimiz bu durumu
çoğunlukla sınamıyor. Bu yüzden `r_prob` de ana sette koşulmalı. Kaynak bazında label
dağılımı ve `r_trans`/`r_prob` label'larının aynı olduğu soru oranı raporlanmalı.
Aynı sebeple sınırdaki `>=`/`>` seçimi önemlidir: `>=` ile iki modelin de doğru
(ya da ikisinin de yanlış) olduğu eşitlikler "kolay" (1) sayılır, `>` ile 0 sayılır.
Bu seçim 0/1 kaynaklarda label'ları t'den çok daha fazla değiştirir; DECISIONS'a yazılmalı.
Hedefler aynı BCE loss'la öğrenilir. DeBERTa-v3-large, 5 epoch ve val checkpoint
seçimi makale tarifidir. CPU TF-IDF varyantı yalnız entegrasyon kontrolü olarak
ayrı isimle bulunmalı. Mevcut `core/classifiers.py` yeniden kullanılabilir, ancak
`TransformerClassifier.fit()` şu anda **val/checkpoint seçmiyor**; önce bu eksik tamamlanmalı.

Başlangıç çifti önerisi `qwen2.5-7b` / `qwen2.5-32b`: aynı aile, farklı ölçek.
Bu bir sonuç iddiası değil. Train kalite ve maliyetleriyle çiftin anlamlı olduğu
kontrol edilir; değişiklik gerekirse test açılmadan sabitlenir. Diğer router'ların
aynı çiftteki karşılaştırma koşuları da bu seçimi kullanır.

### Eşik ve rapor

Skor büyüdükçe küçük model seçilir. Dağıtım eşiği val üzerinde istenen kalite kaybına
göre seçilip dondurulmalı. Mevcut `BinaryRouter.sweep()` test **tahminlerinin** quantile'larını
kullanarak betimsel eğri çiziyor; bu test etiketiyle eğitim değildir, fakat önceden
kalibre edilmiş bir dağıtım eşiği de değildir. `binary_metrics()` test eğrisinden
`drop<=1%` noktasını seçiyor; bu alanı val'de seçilmiş eşik garantisi diye sunamayız.

Rapor: küçük model trafik payı, all-large'a göre kalite farkı ve gerçek token-proxy
maliyeti. Küçük modele %40 trafik göndermek, token maliyetinde %40 tasarruf anlamına
gelmez. BARTScore gibi negatif ölçeklerde göreli yüzde kaybı yanında mutlak fark;
all-large GT sıfırsa göreli kayıp için tanımsız değer raporlanmalı.

### BARTScore ayrımı

Bizim `shared_v1/v2` kanalı SU Lab tanımıdır: `facebook/bart-large` altında
`sum log p(response | prompt)`. Referans cevap kullanmaz, uzunluğa duyarlıdır ve
512-token kesimi vardır. Hybrid'in MixInstruct referanslı BARTScore deneyinin birebir
tekrarı sayılamaz. İlk ana eğitim GT'de; mevcut BARTScore'da ek deney yapılırsa adı
bu uyarlamayı açıkça taşımalı. Orijinal skoru eklemek gerekirse mevcut release'in
skorlarını değiştirmeden yeni, sürümlü bir türetilmiş artifact gerekir. Legacy
scorer'daki referanslı tanım da doğrulanmadan makaleyle birebir kabul edilmemeli.

## 5. GraphRouter: senaryo başına ayrı öğrenme

`llm_routing/routers/graphrouter/{labels,gnn,router}.py` önerilir.

**Açık karar: makale mi, resmi kod mu?** ICLR metni ile resmi kod (`0edfb87`) ağ ve
eğitim ayarlarında birbirini tutmuyor:

| konu | makale (§2.3, §3.4–3.5) | resmi kod / SU Lab portu |
|---|---|---|
| graph | task, query ve LLM düğümlü heterogeneous graph | ayrı task düğümü yok; task embedding'i query özelliğine eklenir |
| GNN katmanı | iki katmanlı graph attention network (GAT) | FeatureAlign + iki `GeneralConv` (attention yok) |
| gizli boyut | 32 | 16 (task 8 + query 8) |
| optimizer / lr | Adam, lr `1e-3`'ten LambdaLR ile 0'a | AdamW, sabit lr `3e-4`, weight decay `1e-4` |
| normalizasyon | performans ve maliyet ikisi de normalize edilir | maliyet task başına min-max; effect ham |

AGENTS.md "makalenin yöntemini izle" diyor; ancak makaledeki sayılar resmi kodla
üretilmiş olabilir ve makale metni GAT'ın tam ayarlarını vermiyor. Hangisinin ana
varyant olacağı **ekip kararıdır** ve implementasyondan önce DECISIONS'a yazılmalı.
Seçilmeyen taraf explainer'da belgelenmiş sapma olarak yer alır. Bu belgenin
aşağıdaki ağ ve eğitim tarifi resmi koddan alınmıştır; bu bir öneri, karar değildir.

```text
effect = sample-0 GT
cost_hat(model, source) = o model/kaynağın train token-proxy maliyet ortalaması
cost_norm = train'den öğrenilmiş, kaynak başına min-max(cost_hat)
reward = alpha * effect - beta * cost_norm
label = one_hot(argmax_model reward)
```

Başlangıç maliyet tahminini kaynak başına sabit tutmak küçük ve anlaşılırdır.
Sonradan prompt uzunluğuna bağlı tahmin eklemek ayrı bir iyileştirme olur. Normalizasyon
parametreleri yalnız train'de hesaplanır; sıfır aralık açıkça ele alınır. Val/test gerçek
completion uzunluğu karar özelliklerine girmez. Gerçek test maliyeti sadece seçimden
sonra evaluator tarafından okunur.

| ayrı eğitim senaryosu | alpha | beta |
|---|---:|---:|
| Performance First | 1.0 | 0.0 |
| Balance | 0.5 | 0.5 |
| Cost First | 0.2 | 0.8 |

Her senaryoda kenar özellikleri `[reward, cost_norm, effect]`; hedef en yüksek reward
modelinin kenarında 1. Resmi kodun eğitimi (makaleden farkı yukarıda): BCE, 1000 optimizer adımı, adım başına 32 maskeleme
çekimi, maske oranı 0.5, AdamW lr `3e-4`, weight decay `1e-4`; checkpoint val macro-F1.
Maske çekimleri sıradan 32-sorguluk minibatch değildir. Eşit reward'da model sırasına
göre ilk modeli seçme davranışı özellikle GT'de raporlanmalı ve test edilmeli.

Çıkarım `argmax(edge_score)` yapar. Bu skor beklenen kalite değildir.
`MultiModelRouter.predict_quality()` üzerinden yeniden `score - λ*cost` yapmak
maliyeti ikinci defa ekler. `sweep()` bir senaryo için tek `Decision` döndürebilir;
üç senaryo ayrı kayıt/çalışma noktasıdır. Mevcut `aiq()` tek noktayı maliyet aralığının
iki ucuna da sabit kaliteyle uzatıyor; bu, o kalitenin daha düşük bütçelerde erişilebilir
olduğunu göstermez. Tek noktalı native deneyde AIQ karşılaştırması yerine
reward/kalite/maliyet tablosu kullanılmalı; ilave eğitim ağırlıklarıyla eğri üretmek
sonradan açıkça tanımlanabilir.

İlk karşılaştırmada yalnız train kenarlarını çıkarım grafiğinde görünür tutma önerisi
var. Resmi kodun testte train∪val kenarlarını açması ek etiket bilgisi sağlar; bunu
uygulamak istenirse aynı adaptasyon verisi diğer yöntemlere de açılıp farklı protokol
olarak adlandırılmalı. Yeni test sorguları eğitim sonrası eklenir; bu sorguların
hiçbir effect/reward kenarı görünür değildir. Sorguları ayrı ayrı veya farklı
batch'lerde değerlendirmek seçimi değiştirmemeli.

Görev bilgisi `Benchmark.sources` olabilir: yedi kaynak ilk task tanımıdır; BBH'nin
27 alt görevi bu görünümde ayrı kaynak olarak taşınmıyor. Kaynağın kullanıcı isteğiyle
bilindiği **source-known** deney koşulu açıkça kaydedilmeli; bu metadata bütün router'lara
izin verilen girdi olmalı. Sadece serbest sorgu metniyle çalışma istenirse train'de
öğrenilmiş görev tahmincisi gerekir ve bu ayrı bir varyanttır. Test query ID'sinden
görev tahmin etmek gizli bir servis varsayımı olmamalı.

MiniLM-L6-v2 ve mean pooling resmi koda uyumlu başlangıçtır. Yedi görev ve 12 model
için açıklamalar kamuya açık model kartları/görev tanımlarından sabit dosyada hazırlanır;
test performansından model yeteneği paragrafı yazılmaz, ücretli GPT-4o çağrısı yapılmaz.
Bu açıklama üretimindeki uyarlama da belgelenir. PyG isteğe bağlı bağımlılık olmalı;
diğer router'ları listelemek için zorunlu GPU/PyG import'u yapılmamalı.

## 6. UniRoute: model ayrımı, sorgu ayrımı kadar önemli

`llm_routing/routers/uniroute/{labels,embedding,router}.py` önerilir. Önce K-means,
ardından LearnedMap varyantı. Açık encoder adayı SU Lab'deki `BAAI/bge-base-en-v1.5`;
Gecko ile aynı encoder veya aynı büyüklük değildir, 768 boyutlu açık bir **ikame**dir.
Revision, CLS pooling, L2 normalizasyonu ve token kesimi kaydedilir.

Önerilen ana protokol: model listesi 6 `H_train` ve 6 `H_new` olarak test skorları
açılmadan sabit `pool_seed` ile ayrılır. Sorguların donmuş 60/10/30 split'i değişmez.

| hangi bilgi? | train sorguları | val sorguları | test sorguları |
|---|---|---|---|
| `H_train` kalite/maliyet | router öğrenimi | K/ayar seçimi ve izin verilen profil hesabı | ana dinamik değerlendirmede kullanılmaz |
| `H_new` kalite/maliyet | **erişim yok** | yeni model profili ve maliyet tahmini | yalnız evaluator |

K-means train sorgu embedding'lerinde öğrenilir. Final PDF Denklem 12:
`Psi[model, cluster] = val kümesindeki ortalama hata`;
`gamma(x, model) = Phi(x) @ Psi[model]` ve seçim `argmin(gamma + λ*cost_hat)`.
0/1 kaynaklarda hata `1 - correctness`; sürekli GT için `1 - token_F1` kullanmak
bizim açıkça raporlanan uyarlamamızdır. Ham negatif BARTScore/ArmoRM'yi bu aralığa
hiç işlem yapmadan koymayalım.

K seçimi yeni modellere bakmaz. Ek G.2'deki adaylar `3..floor(N_val/50)`:
her adayda `H_train` profilini train'den üretip val üzerinde deferral alanıyla seçmek.
Val küçükse aday aralığının boş olması için smoke config'inde açık sabit K gerekir.
LearnedMap için aday K değerlendirmesi **öğrenilmiş haritayı da eğiterek** yapılmalı;
SU Lab'deki `select_k()` yalnız hard map alanıyla K seçiyor.

Final metnin §5.2'si haritayı eğitirken sabit `Psi(H_train)` için val diyor;
Ek G.2'nin seçim tarifi ise train profilini kullanıyor. SU Lab çekirdeği train profilini
seçiyor. Bu ayrımı gizlemeyelim: önerilen uygulama K seçimi için Ek G.2'yi, seçilen K
ile son LearnedMap fit'i için §5.2'yi izler; kullanılan profil split'i manifestte ayrı
alan olur. Yeni modellerin profili her durumda yalnız val'den gelir.

LearnedMap'in final Ek F.3 mimarisi: donmuş embedding, BatchNorm, iki ayrı
128-boyutlu FC/BN/ReLU blok, FC(K), softmax. Adam, 5 epoch, batch 64, lr `0.005`.
Hedef hata olasılığının log loss'u; zaten `[0,1]` olan `gamma`ya ikinci sigmoid
eklenmemeli. SU Lab'nin arXiv-v2 denklem numaralarını final PDF'ye taşımayalım:
finalde FC(K) açıkça yazıyor; eski `FC(M)` typo notu burada gerekmiyor.

Boş val kümesindeki profil hücresi için öneri: aynı modelin **val genel hata
ortalaması**; bu bir uygulama tercihidir ve boş-küme sayısı raporlanır. Modelin val'de
hiç etiketi yoksa train'e veya başka kalite kanalına dönmek yerine deney açıkça durmalı.
Yeni model için train maliyet ortalaması da kullanılmaz; `cost_hat(H_new)` val'den.

`evaluation_models()` ile `H_new` döndürmek mevcut runner'da var, fakat tek başına
yeterli değil: runner'ın `complete_cohort()` adımı fit öncesinde bütün model sütunlarına
bakıyor ve baseline `.fit(train, val)` hâlâ yeni modellerin train etiketlerini alabiliyor.
UniRoute protokolü cohort filtresinden **önce** uygulanmalı: öğrenim girdisi sadece
`H_train`, tanıtım girdisi val, test havuzu `H_new`. Baseline'lar da aynı bilgi sınırına
uymalı. Gerekirse küçük bir eğitim-modeli/protokol hook'u eklenir; her router'a yeni
bir eğitim altyapısı kurulmaz.

Adil dinamik baseline'lar: val profiline dayalı sabit model/Pareto-random ve **val
bankalı kNN**. Quality regression eklenirse `H_new` için sadece tanıtım verisiyle eğitilen
`val-only` baseline olarak adlandırılır; aynı val üzerinde tuning gerekiyorsa bu veri
kendi içinde ayrılır. Bütün modellerin train skorlarıyla eğitilmiş model yalnız
`clairvoyant` ayrıcalıklı referans olarak ayrıca raporlanabilir.

İlk raporda sabit model ayrımı ve 3 eğitim seed'i kullanılır; böylece leaderboard
farklı havuzları birleştirmez. Sonra önceden sabitlenmiş birkaç pool seed için ayrı
sonuçlar verilir. Makalenin 400 tekrarında sorgu split'leri de değişiyor; bizim donmuş
sorgu split'ini korumamız bilinçli sapma. Pool varyansı eğitim seed varyansından ayrı
raporlanmalı. Yeni modeli tanıtmak için gereken val cevap üretimi/puanlama maliyeti
tek seferlik onboarding gideri olarak ayrıca sayılır; release'te önceden toplanmış
olması bu giderin sıfır olduğu anlamına gelmez.

## 7. Mevcut altyapıda gereken sınırlı değişiklikler

| yer | yapılacak iş |
|---|---|
| `core/classifiers.py` | Hybrid için val loss/checkpoint seçimi; CPU API'sini koru |
| `registry.py` | üç paketi kaydet; ağır bağımlılıkları yalnız ilgili router'da yükle |
| `runner.py` | UniRoute bilgi sınırlarını cohort/baseline fit öncesinde uygula; val'de sabitlenen operating point'i ayrı kaydet |
| `core/evaluation.py` ve sonuç kaydı | GraphRouter tek noktasını destekle; kaynak bazında/macro sonuç ve cohort hash'i ekle |
| `leaderboard.py` | mevcut fingerprint+model gruplamasına protokol/cohort kimliğini ekle; native GraphRouter senaryolarını ayrı göster |
| `configs/routers/` ve `configs/experiments/` | CPU smoke ile gerçek encoder config'lerini ayır; metric/model ayrımı isimde ve manifestte açık olsun |
| `docs/papers/` | şablondan üç explainer; yöntem, native ölçüt, veri uyarlaması ve sonuç |
| `docs/walkthrough/` | eğitim adımı tamamlanınca tek Türkçe yazı, gerçek Tosun komut/sayıları |

Genel `fit(train, val)` ve `Decision` sözleşmesi korunur. Test gerçek maliyetiyle
`Decision.cost` yazmak muhasebedir; aynı maliyetin seçimden önce argmax girdisi olması
yasaktır. Genel metrikleri yeniden yazmak yerine paper-native alanlar eklenmeli:
Hybrid trafik payı/kalite kaybı; GraphRouter reward/kalite/maliyet ve macro-F1;
UniRoute QNC/Area. Genel AIQ'nun UniRoute Area ile aynı olduğu varsayılmaz.

## 8. Uygulama sırası ve eğitimden önce geçmesi gereken kontroller

1. **Release:** v2 GPU pilotu, tam koleksiyon, bundle kontrolleri, immutable assembly.
   Ana deney metric/cost raporlama tercihlerini DECISIONS'a geçir. Bu belge cost proxy
   ve mixed-quality açık kararlarını kendiliğinden kapatmıyor.
2. **Hybrid LLM:** üç label varyantı + CPU smoke; ardından DeBERTa eğitim/val seçimi.
   Eşik raporunu doğrula. Qwen çiftini test açılmadan sabitle.
3. **UniRoute K-means:** dinamik model protokolünü ve baseline sınırlarını tamamla;
   sonra LearnedMap. Böylece graph'a geçmeden cohort/protokol raporu hazır olur.
4. **GraphRouter:** resmi ağla üç senaryoyu uygula; kaynak açıklamaları ve train-only
   graph. Sonra istenirse GraphRouter için de aynı tanıtım bütçeli yeni-model deneyi.
5. **Karşılaştırma:** aynı model havuzu/cohort/metric/protokolde ≥3 seed; ardından
   açıklayıcı notebook'lar ve tek eğitim walkthrough'u. Notebook modülü import eder.

Gerekli yöntem testleri:

- Hybrid: elle hesaplanabilen hard/soft hedef, NaN paydayı değiştirmez, eşitlik ve t*
  kontrolü; val checkpoint ve eşik seçimi test skorları değişince aynı kalır.
- GraphRouter: maskelenen kenar kendi gerçek effect/reward'ını göremez; train/val/test
  graph ayrımı, model indeksi ve scenario label'ı; inference'te test quality/cost
  değişince seçim değişmez. Resmi ağla aynı küçük graph üzerinde sayısal kontrol.
- UniRoute: `H_new×train` kalite/maliyet/eksiklikleri değişince router ve adil baseline
  değişmez; K seçimi `H_new×val` etiketlerini kullanmaz; val boşsa hata; test etiketleri
  ve gerçek maliyetleri yalnız evaluator'a ulaşır. Hard/learned haritada şekil ve
  olasılık toplamı; seçilmiş havuzda bütün baseline'lar aynı cohort'u kullanır.
- Ortak: her router geçerli `Decision`, sample-0 muhasebesi ve aynı veri fingerprint'i;
  kayıt–yükleme sonrası aynı tahmin. CPU varyantı gerçek mimariyle aynı isimde raporlanmaz.

## 9. Tosun'da nasıl eğiteceğiz?

Gerçek LLM cevaplarını toplama/puanlama ile router eğitimini ayrı işler olarak
çalıştıracağız. Mevcut `submit_train.sh` zaten seed array'i, config override'ı ve
tek GPU isteğini destekliyor. UniRoute'un donmuş embedding sonrası K-means/MLP kısmı
CPU'da yapılabilir; GraphRouter'ın graph ağı küçük, Hybrid'in encoder fine-tuning'i
daha ağırdır. GPU belleği ve süre için pilot ölçümü olmadan sayı vaat edilmez.
Hybrid için batch/sequence pilotu ve gerekirse uygun GPU override'ı yapılmalı.

**Aşağıdaki router config'leri henüz yok; komutlar implementasyon sonrasının hedef
arayüzüdür ve bu incelemede çalıştırılmadı.** V2 release'i hazırken önce mevcut
`llmr validate-data configs/benchmarks/shared_v2.yaml` kontrolü geçmeli.

```bash
# Planlanan gerçek eğitim config'leri, üç seed (0, 1, 2):
bash cluster/tosun/submit_train.sh configs/experiments/hybrid_llm_trans__shared_v2_gt.yaml 3
bash cluster/tosun/submit_train.sh configs/experiments/hybrid_llm_prob__shared_v2_gt.yaml 3
bash cluster/tosun/submit_train.sh configs/experiments/uniroute_kmeans__shared_v2_dynamic_gt.yaml 3
bash cluster/tosun/submit_train.sh configs/experiments/uniroute_learned__shared_v2_dynamic_gt.yaml 3
bash cluster/tosun/submit_train.sh configs/experiments/graphrouter_performance__shared_v2_gt.yaml 3
bash cluster/tosun/submit_train.sh configs/experiments/graphrouter_balance__shared_v2_gt.yaml 3
bash cluster/tosun/submit_train.sh configs/experiments/graphrouter_cost__shared_v2_gt.yaml 3
```

Bu yedi config ilk ana seti oluşturur (`r_prob`, §4'teki 0/1 GT indirgenmesi
yüzünden `r_trans`'ın yanında); Hybrid `det`, aynı çiftte GraphRouter
karşılaştırması, statik UniRoute ve diğer kalite kanalları ayrı ek deneylerdir.
Çıktılar mevcut `runs/<benchmark>/<experiment>/...` düzeninde config, checkpoint,
label/protokol manifesti, kalite–maliyet noktaları ve eğitim süresini içermeli.
Sonuçlar oluşmadan leaderboard'a tahminî performans satırı eklenmez.
