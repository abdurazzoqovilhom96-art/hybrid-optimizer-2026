# hybrid-optimizer-2026

**CBA-SHADE** (Covariance-Basis Adaptive SHADE) — differensial evolyutsiya
oilasiga mansub gibrid metaevristik optimizatsiya algoritmi va uning to'liq
eksperimental baholash quvuri.

## Tarkibi

`hybrid 2026.py` — yagona skript, to'rtta bo'limdan iborat:

1. **Test funksiyalari** — 12 ta benchmark (siljitilgan, aylantirilgan,
   shovqinli va dinamik variantlari bilan) va yagona `Tracker` FES hisoblagichi.
2. **Algoritm va raqobatchilar** — `CBA_SHADE` hamda zamonaviy DE/ES oilasi
   (L-SHADE, jSO, LSHADE-cnEpSin, SHADE, CMA-ES, klassik DE); `INCLUDE_CLASSIC=1`
   bilan metafora asosidagi eski to'plam (GWO, WOA, PSO, SCA, HHO) ham qo'shiladi.
3. **Parallel eksperiment** — `joblib` orqali barcha (algoritm, funksiya,
   o'lcham, run) kombinatsiyalari.
4. **Eksport** — xom natijalar, o'rtacha/std jadvallari, LaTeX jadval,
   Friedman testi va **post-hoc tahlili**, Mann–Whitney U + Holm bo'yicha
   yutuq/tenglik/yutqazish jadvali, konvergensiya grafiklari.

## Ishga tushirish

```bash
pip install numpy pandas scipy joblib matplotlib
python "hybrid 2026.py"                 # to'liq protokol: 3 o'lcham x 30 run
TEMOA_QUICK=1 python "hybrid 2026.py"   # tezkor smoke-test: 10D x 3 run
INCLUDE_CLASSIC=1 python "hybrid 2026.py"   # klassik algoritmlar ham qo'shiladi
```

Natijalar `cba_shade_outputs/` (yoki `cba_shade_quicktest/`) papkasiga yoziladi.

Algoritm nomi skriptning boshidagi bitta `ALGO_NAME` doimiysi bilan
belgilanadi; chiqish papkasi nomi ham shundan olinadi.

## Hujjatlar

- `docs/ANALYSIS.md` — oldingi eksperiment natijalarining batafsil statistik tahlili.
- `docs/ABLATION.md` — komponentlar bo'yicha ablatsiya tajribalari va xulosalar.
