# 1 Milyon Sayının Matris ile Sıralanması — Radix Sort Görselleştirici

`1.000 x 1.000`'lik bir piksel matrisine yerleştirilmiş 1.000.000 sayının,
**LSD Radix Sort** algoritmasıyla canlı olarak sıralanışını gösteren bir
pygame uygulaması.

Bu proje, aşağıdaki gibi saf Python döngüleriyle yazılmış ve bu yüzden
yavaş çalışan bir ilk sürümün geliştirilmiş hâlidir:

- Piksel piksel `PixelArray` ataması yerine NumPy + `pygame.surfarray`
  ile tek seferde vektörize çizim.
- Sayma / kümülatif toplama / yerleştirme adımları NumPy'nin stabil
  sıralaması (`np.argsort(..., kind="stable")`) ile vektörize edilmiş
  radix sort.
- Sıralama sırasında da olay döngüsünün çalışması (pencere donmaz,
  her an kapatılabilir).
- Gökkuşağı renk paleti, duraklatma, yeniden karıştırma, hız ayarı,
  ekran görüntüsü alma ve fare ile hücre değeri okuma gibi
  etkileşimli özellikler.

## Kurulum

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Çalıştırma

```bash
python radix_sort_visualizer.py
```

> Not: Bu betik bir ekran (display) gerektirir. Sunucu/headless bir
> ortamda denemek isterseniz `SDL_VIDEODRIVER=dummy` ortam değişkenini
> kullanabilirsiniz (görüntü olmadan sadece mantığı test eder).

## Kontroller

| Tuş         | İşlev                                              |
|-------------|-----------------------------------------------------|
| `SPACE`     | Duraklat / devam et                                 |
| `R`         | Sayıları yeniden karıştır ve en baştan başlat        |
| `↑` / `↓`   | Animasyon hızını artır / azalt                      |
| `C`         | Renk modunu değiştir (Gökkuşağı ↔ Gri tonlama)      |
| `S`         | Anlık ekran görüntüsünü PNG olarak kaydet            |
| `ESC`       | Çıkış                                               |

## Ayarları değiştirmek

`radix_sort_visualizer.py` dosyasının en üstündeki sabitlerle oynayarak
davranışı değiştirebilirsiniz:

- `N`, `SIZE`: Sayı adedi ve matris boyutu (`SIZE * SIZE == N` olmalı).
- `RADIX_BASE`: Radix tabanı. Daha küçük bir taban (örn. `2` veya `4`)
  daha fazla (ve daha yavaş gözle izlenebilir) geçiş üretir; `10` ise
  klasik ondalık basamaklarla 6 geçişte tamamlar.
- `DEFAULT_STEP_DELAY_MS`: Geçişler arası varsayılan bekleme süresi.
