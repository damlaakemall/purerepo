# Nasıl yaptık: adım adım anlatım

Bu klasör projenin her adımını **son haliyle** anlatır: ne yaptık, neden böyle seçtik, Tosun'da nasıl
çalıştırdık ve sonucu nasıl aldık, kontrol ettik. Amaç yalnız işi yapmış olmak değil, ekipteki herkesin
nasıl çalıştığını anlaması. Deneme-yanılma günlüğü değildir; her model için ayrı yazı da değildir. Her yazı
bir adımın genel işleyişini anlatır ve gerçek bir örnekle (çoğunlukla 15 Ekim 2026 Tosun pilotu) gösterir.

Ayrıntılı kurallar ve dosya biçimleri başka yerde: [teknik veri protokolü](../shared_protocol.md),
[ekip rehberi](../team_workflow.md), [Tosun README](../../cluster/tosun/README.md). Bu yazılar onların
"neden"ini ve akışını anlatır, tekrar etmez.

## Okuma sırası

| # | yazı | soru |
|---|---|---|
| 1 | [Soru seti](01_soru_seti.md) | Hangi testleri kullanıyoruz, her biri neyi ölçüyor, neden bunlar? |
| 2 | [Model havuzu](02_model_havuzu.md) | Hangi 12 modeli neden seçtik, maliyeti nasıl sayıyoruz? |
| 3 | [Skorlama](03_skorlama.md) | Bir cevabın doğru/iyi olduğuna nasıl karar veriyoruz? Referansla aynı olduğunu nasıl gösterdik? |
| 4 | [Tosun'da çalıştırma](04_tosunda_calistirma.md) | Hesaptan sonuca: bir modeli Tosun'da nasıl ürettik ve skorladık? |
| 5 | [Sonuçları okuma ve kontrol](05_sonuclari_okuma.md) | Çıkan dosyalara nasıl bakıyoruz, sonucun doğru olduğundan nasıl emin oluyoruz? |

Önce [routing temelleri](../routing_fundamentals.md) okunursa "neden bu veriyi topluyoruz" sorusu
kendiliğinden cevaplanır: router'ları eğitmek ve karşılaştırmak için her sorunun her modelde ne kalite ve
ne maliyetle cevaplandığını bilmemiz gerekiyor.

## Yeni yazı ne zaman eklenir

Bir adım bitince (ör. tam `shared_v1` koleksiyonu, assemble, ilk router) o adımın yazısı eklenir veya
güncellenir. [Şablon](_sablon.md) ile başlayın. Kurallar:

- **Son hali anlatın.** "Önce X'i denedik, olmadı" yerine "X'i şöyle yapıyoruz, çünkü...". Karşılaşılan
  ve çözülen bir sorun bir başkasının da başına gelecekse "bilinen sorunlar" tablosuna girer.
- **Neden'i yazın.** Bir seçimin gerekçesi (referansla aynılık, maliyet, makale, donanım) kodda yoktur.
- **Gerçek bir örnek gösterin.** Çalıştırılmış komutlar, gerçek çıktı sayıları. Yazdığınız her komutu
  bir kez çalıştırın.
- **Bağlantı verin.** Kodun ilgili dosyasına ve ayrıntılı protokol bölümüne.
- Türkçe yazılır (ekip dokümanları gibi); kod, komut ve dosya adları olduğu gibi kalır.

Planlanan sonraki yazılar: tam koleksiyon ve birleştirme (assemble / validate / evaluate-data), her router
için "makaleden koda" anlatımı (label, model, loss, eğitim, karar), router karşılaştırması.
