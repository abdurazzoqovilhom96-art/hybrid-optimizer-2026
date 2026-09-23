# Sozlash tajribalarining xom ma'lumotlari

Bu fayllar `docs/ABLATION.md` dagi barcha jadvallarning manbasi. Ular git'da
saqlanadi, chunki hisoblash muhiti vaqtinchalik va maqola uchun raqamlar
qayta tekshirilishi mumkin bo'lishi kerak.

Barchasi 30D da, `FES = 3000·D` byudjeti va `42 + 1000·D + run` seed sxemasi
bilan olingan. Ustunlar: `Alg, Function, Dim, Run, Best`.

| Fayl | Nima tekshirilgan | Run |
|---|---|---|
| `ablation_raw.csv` | Eski TEMOA_V10 komponentlari, yutqazayotgan masalalarda | 11 |
| `round2b_raw.csv` | Populyatsiya hajmi × CMA quyrug'i rejimi | 15 |
| `round3_raw.csv` | Xotira boshlang'ich holati, jSO F cheklovlari, p-rate | 15 |
| `round4_raw.csv` | Ikki parametr rejimi ansambli vs qat'iy rejimlar | 15 |
| `round5_raw.csv` | Quyruqni ishga tushirish chegarasi (`TAIL_DIV`) | 15 |

Ranklarni qayta hisoblashda CEC konventsiyasini qo'llang (`Best < 1e-8 → 0`),
aks holda mashina aniqligidagi ma'nosiz farqlar natijani buzadi.
