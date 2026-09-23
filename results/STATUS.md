# CBA-SHADE avtonom eksperiment — holat

Avtonom rejim har uyg'onishda shu faylni o'qiydi va yangilaydi.
Holat faqat git'dan o'qiladi. To'liq ko'rsatma: `docs/AUTONOMOUS_PROMPT.md`.

| Faza | Holat | Izoh |
|---|---|---|
| FAZA_A — parametrlarni muzlatish | **tugadi** | 5 bosqich sozlash; natijalar `results/tuning/` da |
| FAZA_B — GitHub Actions to'liq eksperiment | **ishlayapti** | 50D tugadi (40 daq); 30D va 100D davom etmoqda |
| FAZA_C — `docs/RESULTS.md` yakuniy tahlil | boshlanmagan | FAZA_B tugagach |

**Oxirgi tekshiruv:** 2026-09-23 17:05 UTC

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
- 17:05 UTC: 50D bo'lagi `success` (40.6 daqiqa). 30D va 100D hali ishlamoqda.
  100D uchun taxminiy tugash ~19:00 UTC; ish chegarasi 350 daqiqa.

## Kutilayotgan zaif tomon

`RotatedElliptic`: CBA-SHADE ~1440, LSHADE-cnEpSin ~0.359. Yakuniy
hisobotda ochiq ko'rsatilishi shart, yashirilmasligi kerak.
