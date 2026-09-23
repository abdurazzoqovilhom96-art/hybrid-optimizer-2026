# CBA-SHADE avtonom eksperiment — holat

Avtonom rejim har uyg'onishda shu faylni o'qiydi va yangilaydi.
Holat faqat git'dan o'qiladi. To'liq ko'rsatma: `docs/AUTONOMOUS_PROMPT.md`.

| Faza | Holat | Izoh |
|---|---|---|
| FAZA_A — parametrlarni muzlatish | **tugadi** | 5 bosqich sozlash; natijalar `results/tuning/` da |
| FAZA_B — GitHub Actions to'liq eksperiment | **ishlayapti** | 30D va 50D tugadi; faqat 100D qoldi |
| FAZA_C — `docs/RESULTS.md` yakuniy tahlil | qisman | protokol bo'limi yozildi; raqamlar kutilmoqda |

**Oxirgi tekshiruv:** 2026-09-23 19:06 UTC

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

## Kutilayotgan zaif tomon

`RotatedElliptic`: CBA-SHADE ~1440, LSHADE-cnEpSin ~0.359. Yakuniy
hisobotda ochiq ko'rsatilishi shart, yashirilmasligi kerak.
