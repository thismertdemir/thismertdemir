"""
1.000.000 Sayının Matris Üzerinde Radix Sort ile Görselleştirilmesi
====================================================================

Orijinal fikir korunmuştur: N adet sayı, SIZE x SIZE'lık bir piksel
matrisine yerleştirilir ve LSD (Least Significant Digit) Radix Sort
algoritması, her basamak geçişinden sonra ekrana çizilerek canlı olarak
görselleştirilir.

Bu sürümde orijinal koda göre şu geliştirmeler yapılmıştır:

  * NumPy + pygame.surfarray ile piksel çizimi vektörize edildi.
    (Saf Python ile PixelArray üzerinde 1.000.000 piksel tek tek
    boyanınca çok yavaş kalıyordu; artık tüm kare milisaniyeler
    içinde çiziliyor.)
  * Radix sort'un sayma/kümülatif toplama/yerleştirme adımları da
    NumPy ile vektörize edildi (stabil sıralama ile birebir aynı
    sonucu üretir), böylece 1 milyon elemanlı diziler saniyenin
    çok altında bir sürede bir basamak geçişini tamamlayabiliyor.
  * Pencere artık sıralama sırasında da olay (event) döngüsünü
    işliyor; bu sayede işletim sistemi pencereyi "yanıt vermiyor"
    olarak işaretlemiyor ve her an kapatılabiliyor.
  * Gri tonlama yerine, değere göre renk üreten bir HSV gökkuşağı
    paleti eklendi (isteğe bağlı olarak gri tonlamaya geçilebilir).
  * Duraklatma, yeniden karıştırma, hız ayarı, ekran görüntüsü alma
    ve fare ile hücre değeri okuma gibi etkileşimli özellikler
    eklendi.
  * Üst bilgi (HUD) panelinde geçiş sayısı, geçen süre, taban (radix
    tabanı), basamak değeri ve kontrol ipuçları gösteriliyor.
  * Taban (RADIX_BASE) parametrik hale getirildi; taban küçültülüp
    büyütülerek animasyonun kaç adımda tamamlanacağı ayarlanabiliyor.
  * Kod, kolayca okunup değiştirilebilmesi için fonksiyonlara ve
    küçük bir durum makinesine (state machine) bölündü.

Kontroller
----------
    SPACE       : Duraklat / devam et
    R           : Sayıları yeniden karıştır ve en baştan başlat
    UP / DOWN   : Animasyon hızını azalt / artır (geçişler arası bekleme)
    C           : Renk modunu değiştir (Gökkuşağı <-> Gri tonlama)
    S           : Anlık ekran görüntüsünü PNG olarak kaydet
    ESC / Kapat : Çıkış
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterator, Tuple

import numpy as np
import pygame

# ==========================================
# AYARLAR
# ==========================================

N = 1_000_000          # Sıralanacak sayı adedi
SIZE = 1_000            # Matris boyutu (SIZE x SIZE = N olmalı)
WIDTH, HEIGHT = SIZE, SIZE

HUD_HEIGHT = 86          # Üst bilgi paneli yüksekliği (piksel)
WINDOW_WIDTH = WIDTH
WINDOW_HEIGHT = HEIGHT + HUD_HEIGHT

RADIX_BASE = 10          # Radix sort tabanı (10 -> klasik ondalık basamaklar)

TARGET_FPS = 60
DEFAULT_STEP_DELAY_MS = 350     # İki basamak geçişi arasındaki varsayılan bekleme
MIN_STEP_DELAY_MS = 0
MAX_STEP_DELAY_MS = 2000
STEP_DELAY_INCREMENT = 50

INITIAL_DISPLAY_SECONDS = 1.2   # Karıştırılmış hâli göstermek için bekleme

BG_COLOR = (18, 18, 22)
HUD_TEXT_COLOR = (235, 235, 240)
HUD_ACCENT_COLOR = (90, 200, 250)
HUD_DONE_COLOR = (120, 230, 140)

assert SIZE * SIZE == N, "SIZE * SIZE, N'e eşit olmalıdır."


class ColorMode(Enum):
    RAINBOW = auto()
    GRAYSCALE = auto()


class AppState(Enum):
    SHOWING_SHUFFLED = auto()
    SORTING = auto()
    DONE = auto()


# ==========================================
# SAYI -> RENK (VEKTÖRİZE)
# ==========================================

def values_to_rainbow(values: np.ndarray, max_value: int) -> np.ndarray:
    """Değerleri (0..max_value) HSV tabanlı bir gökkuşağı paletine eşler.

    Doygunluk ve parlaklık sabit (1.0) tutularak hue (renk tonu) değere
    göre 0..1 aralığında değiştirilir. Tamamen NumPy ile vektörize
    edilmiştir; Python döngüsü içermez.
    """

    hue = values.astype(np.float64) / max(max_value, 1)
    h6 = hue * 6.0
    i = np.floor(h6).astype(np.int64) % 6
    f = h6 - np.floor(h6)

    q = 1.0 - f
    t = f

    r = np.empty_like(hue)
    g = np.empty_like(hue)
    b = np.empty_like(hue)

    masks = [i == k for k in range(6)]

    r[masks[0]] = 1.0; g[masks[0]] = t[masks[0]]; b[masks[0]] = 0.0
    r[masks[1]] = q[masks[1]]; g[masks[1]] = 1.0; b[masks[1]] = 0.0
    r[masks[2]] = 0.0; g[masks[2]] = 1.0; b[masks[2]] = t[masks[2]]
    r[masks[3]] = 0.0; g[masks[3]] = q[masks[3]]; b[masks[3]] = 1.0
    r[masks[4]] = t[masks[4]]; g[masks[4]] = 0.0; b[masks[4]] = 1.0
    r[masks[5]] = 1.0; g[masks[5]] = 0.0; b[masks[5]] = q[masks[5]]

    rgb = np.stack([r, g, b], axis=-1) * 255.0
    return rgb.astype(np.uint8)


def values_to_grayscale(values: np.ndarray, max_value: int) -> np.ndarray:
    """Küçük değerler koyu, büyük değerler açık gri tonlar üretir."""

    scaled = (values.astype(np.float64) / max(max_value, 1) * 255.0).astype(np.uint8)
    return np.stack([scaled, scaled, scaled], axis=-1)


def values_to_colors(values: np.ndarray, max_value: int, mode: ColorMode) -> np.ndarray:
    if mode is ColorMode.RAINBOW:
        return values_to_rainbow(values, max_value)
    return values_to_grayscale(values, max_value)


# ==========================================
# RADIX SORT (VEKTÖRİZE, ADIM ADIM ÜRETEN GENERATOR)
# ==========================================

@dataclass
class SortStep:
    numbers: np.ndarray
    pass_index: int
    total_passes: int
    exp: int
    digit_place_value: int


def count_passes(max_value: int, base: int) -> int:
    if max_value <= 0:
        return 1
    passes = 0
    value = max_value
    while value > 0:
        passes += 1
        value //= base
    return passes


def radix_sort_steps(numbers: np.ndarray, base: int = 10) -> Iterator[SortStep]:
    """LSD Radix Sort'u her basamak geçişinden sonra `yield` eden üreteç.

    Her adımda, o basamağa göre NumPy'nin stabil sıralaması (`kind="stable"`)
    kullanılır. Bu, elle yazılmış bir sayma sıralamasıyla (counting sort)
    birebir aynı sonucu üretir; çünkü LSD radix sort zaten her basamak için
    *stabil* bir sıralama gerektirir. Burada yapılan tek fark, sayma /
    kümülatif toplama / yerleştirme adımlarının saf Python döngüleri yerine
    NumPy'nin optimize edilmiş C koduyla çalıştırılmasıdır.
    """

    current = numbers.copy()
    max_value = int(current.max()) if current.size else 0
    total_passes = count_passes(max_value, base)

    exp = 1
    for pass_index in range(1, total_passes + 1):
        digit = (current // exp) % base
        order = np.argsort(digit, kind="stable")
        current = current[order]
        yield SortStep(
            numbers=current,
            pass_index=pass_index,
            total_passes=total_passes,
            exp=exp,
            digit_place_value=exp,
        )
        exp *= base


# ==========================================
# ÇİZİM YARDIMCILARI
# ==========================================

def render_matrix(surface: pygame.Surface, numbers: np.ndarray, max_value: int, mode: ColorMode) -> None:
    """1 milyon sayıyı tek seferde (vektörize) ekrana çizer."""

    colors = values_to_colors(numbers, max_value, mode)
    grid = colors.reshape(HEIGHT, SIZE, 3)
    # surfarray, (genişlik, yükseklik, 3) şeklinde bir dizi bekler.
    pixel_array = grid.transpose(1, 0, 2)
    pygame.surfarray.blit_array(surface, pixel_array)


def format_duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes:02d}:{secs:05.2f}"


def draw_hud(
    screen: pygame.Surface,
    font: pygame.font.Font,
    small_font: pygame.font.Font,
    state: AppState,
    pass_index: int,
    total_passes: int,
    exp: int,
    elapsed: float,
    step_delay_ms: int,
    color_mode: ColorMode,
    cursor_info: str,
) -> None:
    hud_rect = pygame.Rect(0, HEIGHT, WINDOW_WIDTH, HUD_HEIGHT)
    pygame.draw.rect(screen, (28, 28, 34), hud_rect)
    pygame.draw.line(screen, (60, 60, 70), (0, HEIGHT), (WINDOW_WIDTH, HEIGHT), 2)

    if state is AppState.SHOWING_SHUFFLED:
        status_text = "Karıştırıldı — sıralama başlıyor..."
        status_color = HUD_ACCENT_COLOR
    elif state is AppState.SORTING:
        status_text = f"Sıralanıyor  |  Geçiş {pass_index}/{total_passes}  |  Basamak değeri: {exp}"
        status_color = HUD_ACCENT_COLOR
    else:
        status_text = "Sıralama tamamlandı!"
        status_color = HUD_DONE_COLOR

    mode_text = "Gökkuşağı" if color_mode is ColorMode.RAINBOW else "Gri Tonlama"

    line1 = font.render(status_text, True, status_color)
    line2 = small_font.render(
        f"N = {N:,}".replace(",", ".") +
        f"   |  Taban: {RADIX_BASE}  |  Renk modu: {mode_text}  |  "
        f"Hız (gecikme): {step_delay_ms} ms  |  Süre: {format_duration(elapsed)}",
        True,
        HUD_TEXT_COLOR,
    )
    line3 = small_font.render(
        "SPACE: Duraklat/Devam   R: Yeniden Karıştır   ↑/↓: Hız   "
        "C: Renk Modu   S: Ekran Görüntüsü   ESC: Çıkış",
        True,
        (170, 170, 180),
    )
    line4 = small_font.render(cursor_info, True, (150, 200, 150))

    screen.blit(line1, (12, HEIGHT + 6))
    screen.blit(line2, (12, HEIGHT + 30))
    screen.blit(line3, (12, HEIGHT + 50))
    screen.blit(line4, (WINDOW_WIDTH - line4.get_width() - 12, HEIGHT + 6))


def cursor_value_text(numbers: np.ndarray, mouse_pos: Tuple[int, int]) -> str:
    x, y = mouse_pos
    if 0 <= x < WIDTH and 0 <= y < HEIGHT:
        index = y * SIZE + x
        value = int(numbers[index])
        return f"İmleç -> x:{x} y:{y}  değer: {value:,}".replace(",", ".")
    return "İmleç matris üzerinde değil"


def save_screenshot(screen: pygame.Surface) -> str:
    filename = f"radix_sort_{int(time.time())}.png"
    pygame.image.save(screen, filename)
    return filename


# ==========================================
# ANA PROGRAM
# ==========================================

def make_shuffled_numbers() -> np.ndarray:
    numbers = np.arange(N, dtype=np.int64)
    rng = np.random.default_rng()
    rng.shuffle(numbers)
    return numbers


def main() -> None:
    pygame.init()
    pygame.display.set_caption("1 Milyon Sayının Matris ile Sıralanması — Radix Sort")

    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    matrix_surface = pygame.Surface((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    pygame.font.init()
    font = pygame.font.SysFont("consolas,menlo,monospace", 20, bold=True)
    small_font = pygame.font.SysFont("consolas,menlo,monospace", 15)

    print("1.000.000 sayı oluşturuluyor ve karıştırılıyor...")
    numbers = make_shuffled_numbers()
    print("Sayılar hazır. Görselleştirme başlıyor.")

    max_value = N - 1
    color_mode = ColorMode.RAINBOW
    step_delay_ms = DEFAULT_STEP_DELAY_MS

    state = AppState.SHOWING_SHUFFLED
    state_entered_at = pygame.time.get_ticks()
    last_step_time = pygame.time.get_ticks()
    paused = False

    sort_gen: Iterator[SortStep] = radix_sort_steps(numbers, RADIX_BASE)
    pass_index = 0
    total_passes = count_passes(max_value, RADIX_BASE)
    current_exp = 1

    sort_start_time = None
    elapsed = 0.0

    running = True
    while running:
        now = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    print("Yeniden karıştırılıyor...")
                    numbers = make_shuffled_numbers()
                    sort_gen = radix_sort_steps(numbers, RADIX_BASE)
                    pass_index = 0
                    current_exp = 1
                    total_passes = count_passes(max_value, RADIX_BASE)
                    state = AppState.SHOWING_SHUFFLED
                    state_entered_at = now
                    sort_start_time = None
                    elapsed = 0.0
                    paused = False
                elif event.key == pygame.K_UP:
                    step_delay_ms = max(MIN_STEP_DELAY_MS, step_delay_ms - STEP_DELAY_INCREMENT)
                elif event.key == pygame.K_DOWN:
                    step_delay_ms = min(MAX_STEP_DELAY_MS, step_delay_ms + STEP_DELAY_INCREMENT)
                elif event.key == pygame.K_c:
                    color_mode = (
                        ColorMode.GRAYSCALE if color_mode is ColorMode.RAINBOW else ColorMode.RAINBOW
                    )
                elif event.key == pygame.K_s:
                    name = save_screenshot(screen)
                    print(f"Ekran görüntüsü kaydedildi: {name}")

        # --- Durum makinesi ---
        if state is AppState.SHOWING_SHUFFLED:
            if now - state_entered_at >= INITIAL_DISPLAY_SECONDS * 1000:
                state = AppState.SORTING
                sort_start_time = now
                last_step_time = now

        elif state is AppState.SORTING:
            if not paused and now - last_step_time >= step_delay_ms:
                try:
                    step = next(sort_gen)
                    numbers = step.numbers
                    pass_index = step.pass_index
                    total_passes = step.total_passes
                    current_exp = step.exp
                    last_step_time = now
                    print(f"Sıralama aşaması: {step.exp} basamağı tamamlandı "
                          f"({pass_index}/{total_passes}).")
                except StopIteration:
                    state = AppState.DONE
                    print("Sıralama tamamlandı!")

            if sort_start_time is not None and not paused:
                elapsed = (now - sort_start_time) / 1000.0

        # --- Çizim ---
        screen.fill(BG_COLOR)
        render_matrix(matrix_surface, numbers, max_value, color_mode)
        screen.blit(matrix_surface, (0, 0))

        mouse_pos = pygame.mouse.get_pos()
        cursor_info = cursor_value_text(numbers, mouse_pos)

        draw_hud(
            screen,
            font,
            small_font,
            state,
            pass_index,
            total_passes,
            current_exp,
            elapsed,
            step_delay_ms,
            color_mode,
            cursor_info,
        )

        pygame.display.flip()
        clock.tick(TARGET_FPS)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
