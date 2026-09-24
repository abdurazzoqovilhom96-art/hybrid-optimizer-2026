# ACE-SHADE — holat

Butun eksperiment bitta faylda (`ace_shade.py`) va bitta ish oqimida
(`.github/workflows/ace_shade.yml`). Bosqichma-bosqich to'xtash yo'q:
ishga tushirilgach test -> sozlash -> muzlatish -> baholash -> ablatsiya
-> tahlil -> hisobot ketma-ketligi avtomatik o'tadi.

## Quvur

| Ish | Mazmun | Shard |
|---|---|---|
| `selftest` | 9 birlik testi — darvoza | 1 |
| `tune` | CEC-2017, baholash byudjetida | 12 |
| `freeze` | eng yaxshi konfiguratsiya qotiriladi | 1 |
| `run` | CEC-2022, 12F x 2D x 30run x 8alg | 48 |
| `ablate` | N1 o'chiq / N2 o'chiq / cmu manbai | 24 |
| `report` | jadvallar + `docs/RESULTS_V2.md` | 1 |

## Chiqish

```
frozen_params.json              muzlatilgan parametrlar (audit uchun)
results/tuning_v2/              sozlash ranklari
results/v2/tables/              15 jadval + kategoriya tahlili
results/v2/figures/             24 yaqinlashish grafigi
results/ablation_v2/            ablatsiya
docs/RESULTS_V2.md              avtomatik hisobot
```

## Muvaffaqiyat mezoni (oldindan belgilangan)

- S1: CEC-2022 da eng yaxshi o'rtacha rank
- S2: Holm post-hoc `p < 0.05` — **asosiy guruhga** nisbatan
  (LSHADE-cnEpSin, jSO, L-SHADE, CMA-ES)
- S3: ablatsiyada N1 va N2 ning har biri hissa qo'shadi

Ikkilamchi guruh (NL-SHADE-RSP, LSHADE-SPACMA, LSHADE-RSP) qayta
amalga oshirilgan, shuning uchun ularga nisbatan g'alaba kuchsiz dalil
va hisobotda shunday belgilanadi.

## Jurnal

- Yagona fayl yozildi (1869 qator), paket tarqatildi.
- 9/9 birlik testi o'tdi; quvur tez rejimda boshidan oxirigacha sinaldi.
