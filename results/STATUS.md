# CBA-SHADE avtonom eksperiment — holat

Avtonom rejim har uyg'onishda shu faylni o'qiydi va yangilaydi.
Holat faqat git'dan o'qiladi. To'liq ko'rsatma: `docs/AUTONOMOUS_PROMPT.md`.

| Faza | Holat | Izoh |
|---|---|---|
| FAZA_A — parametrlarni muzlatish | **tugadi** | 5 bosqich sozlash; natijalar `results/tuning/` da |
| FAZA_B — GitHub Actions to'liq eksperiment | **tugadi** | 19:32 UTC da natijalar branchga yozildi |
| FAZA_C — `docs/RESULTS.md` yakuniy tahlil | **tugadi** | 9 bo'lim, barcha raqamlar bilan |

**Oxirgi tekshiruv:** 2026-09-23 20:10 UTC

## Muzlatilgan parametrlar

`POP_FACTOR=6`, `MEM_INIT="ensemble"`, `TAIL_FRAC=0.15`, `TAIL_DIV=1e-2`,
`N_MIN=4`, `H_SIZE=6`, `ARC_RATE=2.6`, `K_RSP=3.0`, `P_MAX=0.25`,
`P_MIN_RATE=0.125`, `EIG_LR=0.2`. **Bundan keyin o'zgartirilmaydi.**

## Jurnal

- Uyg'otish mexanizmi empirik sinovdan o'tdi (16:15 UTC da rejalashtirilgan
  test aynan vaqtida yetib keldi).
- GitHub Actions quvuri ikki marta `success`; natijalarni branchga qaytarib
  commit qilish ham tasdiqlandi (`github-actions[bot]` commiti).
- Sozlash xom ma'lumotlari `results/tuning/` da saqlandi.
- FAZA_B ishga tushirildi (run 35887728952).
- 17:05 UTC: 50D bo'lagi `success` (40.6 daqiqa).
- 17:08 UTC: 30D bo'lagi `success` (50.3 daqiqa).
- 18:05 UTC: 100D 106 daqiqadan beri ishlamoqda; chegara 350 daqiqa
  (~22:08 UTC gacha), shuning uchun zaxira yetarli.
- `docs/RESULTS.md` ning protokol bo'limi yozildi (raqamlarga bog'liq emas).
- 19:06 UTC: 100D 166 daqiqadan beri ishlamoqda (chegara 350, ~22:08 UTC gacha).
- `scripts/weaknesses.py` yozildi va sinovdan o'tkazildi: 8-bo'lim uchun
  mag'lubiyat va teng natijalarni jadvallardan avtomatik ajratadi. Teng
  natija Holm p qiymati bilan aniqlanadi, o'rtachalar taqqoslanishi bilan
  emas - aks holda ahamiyatsiz farq g'alaba deb sanalardi.

## Asosiy natija

Friedman: chi2 = 122.90, p = 4.0e-24. CBA-SHADE eng yaxshi o'rtacha rank
(2.50), lekin **post-hoc testda zamonaviy DE variantlarining hech biridan
statistik jihatdan ustun emas** (Holm p = 0.305 barcha to'rttasi uchun).
Ahamiyatli farq faqat DE va CMA-ES ga nisbatan.

Eng yaxshi raqib bilan taqqoslaganda: 36 instansiyadan 8 tasida g'alaba,
18 tasida teng, 10 tasida mag'lubiyat.

Tizimli zaifliklar: shovqinli funksiyalar (uchala NoisySphere + 100D
NoisyRastrigin), RotatedElliptic (30D va 50D), 100D da Levy va Zakharov.
Batafsil: `docs/RESULTS.md` 8-bo'lim.
