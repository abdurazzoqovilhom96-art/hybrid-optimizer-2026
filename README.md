# hybrid-optimizer-2026

**CBA-SHADE** (Covariance-Basis Adaptive SHADE) — differensial evolyutsiya
oilasiga mansub gibrid metaevristik optimizatsiya algoritmi va uning to'liq
eksperimental baholash quvuri.

Butun loyiha bitta mustaqil faylda: **`cba_shade.py`**. Tashqi ma'lumot fayli
yoki yordamchi modul talab qilinmaydi.

## Tarkibi

1. **Test funksiyalari** — 12 ta benchmark (siljitilgan, aylantirilgan,
   shovqinli va dinamik variantlari) va yagona `Tracker` FES hisoblagichi.
2. **Algoritm va raqobatchilar** — `CBA_SHADE` hamda zamonaviy DE/ES oilasi:
   L-SHADE, jSO, LSHADE-cnEpSin, SHADE, CMA-ES va klassik DE.
   `INCLUDE_CLASSIC=1` bilan metafora asosidagi eski to'plam (GWO, WOA, PSO,
   SCA, HHO) ham qo'shiladi.
3. **Parallel eksperiment** — `joblib` orqali barcha (algoritm, funksiya,
   o'lcham, run) kombinatsiyalari.
4. **Eksport** — xom natijalar va konvergensiya egri chiziqlari,
   o'rtacha/std jadvallari, LaTeX jadval, Friedman testi va **post-hoc
   tahlili**, Mann–Whitney U + Holm bo'yicha yutuq/tenglik/yutqazish jadvali,
   konvergensiya grafiklari.

## Ishga tushirish

```bash
pip install numpy pandas scipy joblib matplotlib
python cba_shade.py              # to'liq protokol: 12 funksiya x {30,50,100}D x 30 run
QUICK=1 python cba_shade.py      # tezkor smoke-test: 10D x 3 run (~1 daqiqa)
```

### Google Colab

To'liq protokol Colab'ning 2 yadrosida bir necha soat oladi va sessiya uzilib
qolishi mumkin. Shuning uchun tajribani o'lchamlar bo'yicha bo'laklab ishlating,
keyin `MERGE` bilan birlashtiring — natija bitta sessiyada ishlatilgan bilan
bir xil bo'ladi.

```python
!git clone https://github.com/abdurazzoqovilhom96-art/hybrid-optimizer-2026.git
%cd hybrid-optimizer-2026

# Natijalar Drive'da saqlansin (sessiya uzilsa ham yo'qolmaydi)
from google.colab import drive; drive.mount('/content/drive')
%env OUT_BASE=/content/drive/MyDrive/cba

# 1-bo'lak
!DIMS=30 RUNS=30 OUT_DIR=$OUT_BASE/d30 python cba_shade.py
# 2-bo'lak (yangi sessiyada bo'lsa ham bo'ladi)
!DIMS=50 RUNS=30 OUT_DIR=$OUT_BASE/d50 python cba_shade.py
# 3-bo'lak + hammasini birlashtirish
!DIMS=100 RUNS=30 OUT_DIR=$OUT_BASE/full MERGE=$OUT_BASE/d30,$OUT_BASE/d50 python cba_shade.py
```

Oxirgi buyruq `full/` papkasida uchala o'lcham bo'yicha yagona jadvallar,
statistika va grafiklarni beradi.

### Muhit o'zgaruvchilari

| O'zgaruvchi | Ma'nosi |
|---|---|
| `QUICK=1` | 10D x 3 run smoke-test |
| `DIMS=30,50` | qaysi o'lchamlar ishlatilsin |
| `RUNS=30` | mustaqil run soni |
| `OUT_DIR=nom` | chiqish papkasi |
| `MERGE=d1,d2` | avvalgi bo'laklarni qo'shib tahlil qilish |
| `INCLUDE_CLASSIC=1` | klassik/metafora algoritmlarni ham qo'shish |

Algoritm nomi skriptning boshidagi bitta `ALGO_NAME` doimiysi bilan
belgilanadi; standart chiqish papkasi nomi ham shundan olinadi.

## Hujjatlar

- `docs/ANALYSIS.md` — oldingi eksperiment natijalarining batafsil statistik tahlili.
- `docs/ABLATION.md` — komponentlar bo'yicha ablatsiya tajribalari va xulosalar.
