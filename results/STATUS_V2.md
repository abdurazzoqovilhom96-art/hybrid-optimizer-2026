# ACE-SHADE — holat

Ko'rsatma: `docs/ACE_SHADE_PROMPT.md`. Asos: `docs/REDESIGN_PLAN.md`.
Holat faqat git'dan o'qiladi.

| Bosqich | Holat | Izoh |
|---|---|---|
| B0 — kod + 9 birlik testi | boshlanmagan | `ace_shade.py` |
| B1 — sozlash (CEC-2017 + sintetik shovqin) | boshlanmagan | |
| B2 — parametrlarni muzlatish | boshlanmagan | |
| B3 — **asosiy**: bbob-noisy | boshlanmagan | 30 F x 3 D x 15 inst x 10 alg |
| B4 — ablatsiya N1/N2/N3 | boshlanmagan | |
| B5 — Koshi shovqini tekshiruvi | boshlanmagan | |
| B6 — ikkilamchi: CEC-2022 | boshlanmagan | 12 F x 2 D x 30 run x 10 alg |
| B7 — `docs/RESULTS_V2.md` + kategoriya tahlili | boshlanmagan | |

**Oxirgi tekshiruv:** 2026-09-24

## Qарорlar

**Asosiy maydon almashtirildi:** statik CEC o'rniga **shovqinli
optimallashtirish** (`bbob-noisy`, 30 funksiya). Sabab: o'lchangan
kuchimiz (`NoisyRastrigin` 19.5x, `DynamicSphere` 70x) noaniq relyefga
tegishli; statik CEC da esa 2014–2017 bazaviy usullaridan ajralib
chiqa olmadik (Holm p = 0.305). N3 endi markaziy hissa.

CEC-2022 ikkilamchi maydon sifatida saqlanadi.

Sozlash: CEC-2017 (umumiy parametrlar) + o'z sintetik shovqinli
to'plamimiz (shovqin parametrlari). Uchala to'plam kesishmaydi.

Raqobatchilar: **UH-CMA-ES** (majburiy), NL-SHADE-RSP, AGSK,
LSHADE-SPACMA, LSHADE-cnEpSin, LSHADE-RSP, jSO, L-SHADE, CMA-ES, DE.

## Tekshirilgan faktlar

- `opfunu`: CEC-2022 (12 F) va CEC-2017 (29 F) — optimumda rasmiy bias
- `coco-experiment`: bbob 2160, bbob-noisy 2700 masala; shovqin ishlaydi
- COCO natijalar arxivi (`numbbo.github.io`) muhit siyosati bilan
  bloklangan (403) — ruxsat berilsa muallif natijalari bilan
  taqqoslash ochiladi

## Jurnal

- Maydon almashtirildi, promt v2 yozildi. Kod hali boshlanmagan.
