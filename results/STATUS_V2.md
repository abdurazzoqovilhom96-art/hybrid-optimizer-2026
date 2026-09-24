# ACE-SHADE — holat

Ko'rsatma: `docs/ACE_SHADE_PROMPT.md` (yakuniy, o'zgarmaydi).

| Bosqich | Holat | Actions daq |
|---|---|---|
| B0 — kod + 8 birlik testi | boshlanmagan | — |
| B1 — CEC-2017 da sozlash (`r_N`, `eta`) | boshlanmagan | 43 |
| B2 — parametrlarni muzlatish | boshlanmagan | — |
| B3 — CEC-2022 (12F x 2D x 30run x 8alg) | boshlanmagan | 823 |
| B4 — ablatsiya (4 variant) | boshlanmagan | 411 |
| B5 — `docs/RESULTS_V2.md` + kategoriya tahlili | boshlanmagan | — |

**Oxirgi tekshiruv:** 2026-09-24

## Yopilgan ko'lam

Maydon: **CEC-2022** (12 F, D = 10/20, 30 run). Boshqa benchmark yo'q.
Komponentlar: **N1** (to'plamli kovariatsiya) va **N2** (parametr
rejimi ansambli). N3 (shovqin damping) **olib tashlandi** — CEC-2022 da
shovqinli funksiya yo'q, ablatsiya bilan asoslab bo'lmaydi.
Raqobatchilar (7): NL-SHADE-RSP, LSHADE-SPACMA, LSHADE-cnEpSin,
LSHADE-RSP, jSO, L-SHADE, CMA-ES.

Tashqarida: bbob-noisy, RL ilovasi, AGSK, UH-CMA-ES, DE.

## Muvaffaqiyat mezoni (oldindan belgilangan)

- S1: CEC-2022 da eng yaxshi o'rtacha rank
- S2: Holm post-hoc `p < 0.05` — L-SHADE, jSO, LSHADE-cnEpSin ga nisbatan
- S3: ablatsiyada N1 va N2 ning har biri hissa qo'shadi

NL-SHADE-RSP ga nisbatan S2 bajarilmasa — muvaffaqiyatsizlik emas,
shunday yoziladi.

## To'xtash sharti

B3 tugagach natija qanday bo'lsa shunday yoziladi. Qayta sozlash,
maydon almashtirish, komponent qo'shish yo'q.

## Jurnal

- Ko'lam yopildi; promt yakuniy. Kod hali boshlanmagan.
