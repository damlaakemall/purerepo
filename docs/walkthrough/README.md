# Nasıl yaptık: projenin adımları

Bu klasör projenin her büyük adımını **tek bir yazıda, son haliyle** anlatır: ne yaptık, neden böyle seçtik,
Tosun'da nasıl çalıştırdık, sonucu nasıl aldık ve kontrol ettik. Amaç yalnız işi yapmış olmak değil, ekipteki
herkesin nasıl çalıştığını anlaması. Deneme-yanılma günlüğü ya da her model için ayrı yazı değildir; adımın
genel işleyişini gerçek bir koşuyla gösterir.

| adım | yazı | durum |
|---|---|---|
| 1 | [Router'lar için eğitim verisi üretmek](01_egitim_verisi.md): sorular, modeller, Tosun'da cevap üretme, puanlama, kontrol | pilotla doğrulandı (15 Ekim 2026), tam koleksiyon bekliyor |
| 2 | Router'ları eğitmek ve karşılaştırmak (her makale bir bölüm) | sırada |

Önce [routing temelleri](../routing_fundamentals.md) okunursa bu adımların neden gerektiği kendiliğinden
anlaşılır. Ayrıntılı kurallar ve dosya biçimleri: [teknik protokol](../shared_protocol.md),
[ekip rehberi](../team_workflow.md), [Tosun README](../../cluster/tosun/README.md).

## Yeni yazı ne zaman eklenir

Bir adım bitince ya da önemli ölçüde değişince o adımın yazısı eklenir veya güncellenir. Yeni yazı için
[şablon](_sablon.md) kullanılır. Kurallar:

- **Bir adım, bir yazı.** Adımın parçaları (ör. soru seçimi, puanlama) aynı yazının bölümleridir.
- **Son hali anlatın.** "Önce X'i denedik, olmadı" yerine "X'i şöyle yapıyoruz, çünkü...". Bir başkasının da
  karşılaşacağı ve çözülmüş bir sorun "bilinen sorunlar" tablosuna girer.
- **Neden'i yazın.** Bir seçimin gerekçesi (referans, makale, donanım, maliyet, adillik kuralı) kodda yazmaz.
- **Gerçek bir örnek gösterin.** Çalıştırılmış komutlar, gerçek sayılar. Yazdığınız her komutu bir kez çalıştırın.
- **Ne çok uzun ne çok kısa.** Ayrıntı protokolde; yazı akışı ve gerekçeyi verir, bağlantıyla ayrıntıya yollar.
- Türkçe yazılır; kod, komut ve dosya adları olduğu gibi kalır.
