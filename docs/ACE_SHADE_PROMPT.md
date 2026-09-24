# ACE-SHADE: bajarish promti (v2 — shovqinli optimallashtirish maydoni)

> O'zini o'zi yetarli hujjat. Kontekst yo'qolsa shu fayldan davom
> ettiriladi. Holat: `results/STATUS_V2.md`. Asos: `docs/REDESIGN_PLAN.md`.

## 0. Mandat va qat'iy qoidalar

**Mandat.** "Buni qilib bo'lmaydi" — natija emas. Har bir o'lchangan
zaiflik uchun matematik asoslangan yechim quriladi va sinaladi. To'xtash
sharti faqat o'lchov natijasi.

**Buzilmaydigan qoidalar:**

1. Sozlash to'plami baholash to'plamlari bilan **hech qachon**
   kesishmaydi (1.2-bo'limga qarang).
2. Parametrlar B2 da muzlatiladi va keyin o'zgartirilmaydi.
3. Har bir yangi komponent ablatsiya bilan asoslanadi.
4. Barcha algoritmlar bir xil: seed, chegara ishlovi, baholash
   hisoblagichi, to'xtash sharti, tavsiya qilingan yechim ta'rifi.
5. Natijalar bezalmaydi. Mag'lubiyat raqami bilan yoziladi.
6. Faqat `claude/youthful-pascal-sjxqnn` branchiga push.
7. Har qadamdan keyin darhol commit + push.

## 1. Maydon va to'plamlar

### 1.1 Nega shovqinli optimallashtirish asosiy maydon

O'lchangan kuchli tomonlarimiz — `NoisyRastrigin` 19.5x, `DynamicSphere`
70x, `Schwefel` 75 060x — noaniq va nostatsionar relyefga tegishli.
Statik CEC to'plamlarida bunday funksiya yo'q va u yerda biz 2014–2017
yillardagi bazaviy usullardan ham ajralib chiqa olmadik
(`docs/RESULTS.md`, Holm p = 0.305).

Shu sababli N3 (shovqin bilan boshqariladigan damping) **markaziy
hissa** bo'ladi, yamoq emas.

### 1.2 To'plamlar (tekshirilgan, mavjud)

| Rol | To'plam | Hajm | Holat |
|---|---|---|---|
| **Asosiy baholash** | `bbob-noisy` (COCO) | 30 funksiya (f101–f130), D = 5/10/20, instansiya 1–15 | ✅ tekshirildi |
| **Ikkilamchi baholash** | CEC-2022 (`opfunu`) | 12 funksiya, D = 10/20, 30 run | ✅ optimumda rasmiy bias |
| **Sozlash** | CEC-2017 (`opfunu`) | 29 funksiya, D = 10/30 | ✅ tekshirildi |
| **Shovqin parametrlarini sozlash** | o'z sintetik shovqinli to'plamimiz (eski ish) | 12 funksiya | repozitoriyda |

**Kesishmaslik dalili:** CEC-2017 va CEC-2022 turli funksiya
to'plamlari; BBOB COCO oilasidan, ikkalasidan ham mustaqil. Shovqin
parametrlari BBOB shovqin modellarida emas, o'z sintetik to'plamimizda
sozlanadi — bu BBOB ning Gauss/tekis/Koshi modellariga moslab olishning
oldini oladi.

O'rnatish: `pip install coco-experiment opfunu`

### 1.3 Raqobatchilar

| Algoritm | Nega kerak |
|---|---|
| **UH-CMA-ES** (Hansen va b.) | **Majburiy** — shovqinli optimallashtirishning etakchi usuli. Bo'lmasa retsenzent birinchi bo'lib shuni so'raydi |
| NL-SHADE-RSP | CEC-2021 g'olibi |
| AGSK | zamonaviy, boshqa oiladan |
| LSHADE-SPACMA | eng yaqin gibrid raqib (DE + CMA) |
| LSHADE-cnEpSin | eigen-crossover manbai |
| LSHADE-RSP, jSO, L-SHADE | oila asoslari |
| CMA-ES, DE | nazorat |

**Qayta amalga oshirish xavfi.** Har bir raqib maqoladagi psevdokodga
qat'iy amal qilib yoziladi; har birining sanity-testi bo'ladi (CEC-2017
da e'lon qilingan tartib bilan solishtirish). Nomuvofiqlik topilsa
`docs/RESULTS_V2.md` da ochiq yoziladi.

## 2. Shovqinli baholashning metodologiyasi (B0 da hal qilinadi)

Shovqinli masalada **kuzatilgan eng yaxshi qiymat noto'g'ri o'lchov**:
u shovqinli namunalar minimumi, shuning uchun optimistik qiya. To'g'ri
o'lchov — algoritm **tavsiya qilgan** nuqtadagi shovqinsiz qiymat.

Tartib:
1. Har bir algoritm byudjet oxirida bitta `x_rec` tavsiya qiladi
   (odatda populyatsiya markazi yoki eng yaxshi nuqta — barcha
   algoritmlar uchun bir xil ta'rif).
2. `cocoex` shovqinsiz qiymatga kirish berishini tekshirish.
3. Bermasa: `x_rec` ni `n_eval = 100` marta baholab o'rtacha olinadi;
   bu baholashlar byudjetga **kirmaydi** va barcha algoritmlar uchun
   bir xil qo'llanadi.

Bu qaror `docs/RESULTS_V2.md` da alohida bo'lim sifatida yoziladi.

## 3. ACE-SHADE: to'liq matematik spetsifikatsiya

### 3.1 Holat

```
P (hajmi N_t), A (arxiv)
M_F[b], M_CR[b]   b = 0 (jSO), 1 (L-SHADE), H = 6 katak
p_bank, p_eig, p_cma  va ularning kreditlari c_*
C (to'plamli kovariatsiya), p_c (evolyutsiya yo'li)
B, d:  C = B diag(d^2) B^T   (dangasa yangilanish)
nu (shovqin bahosi)
```

### 3.2 Boshlang'ich qiymatlar

```
N_init = round(r_N * D),  N_min = 4,  H = 6,  A_r = 2.6
bank 0 (jSO):     M_F = 0.3,  M_CR = 0.8,  terminal katak (0.9, 0.9)
bank 1 (L-SHADE): M_F = 0.5,  M_CR = 0.5
p_bank = p_eig = 0.5,   p_cma = 0.1,   C = I,  p_c = 0,  nu = 0
```

Sozlanadigan: `r_N, kappa, lambda_nu, eta, g_nu, alpha`.

### 3.3 To'plamli kovariatsiya (N1 — asosiy yangilik)

Joriy kod har avlodda `np.cov(pop[:N/2])` oladi: `n = 3D` namunadan `D`
o'lchamda, spektral xato `O(sqrt(D/n)) ~ 58%`. O'rniga:

```
mu     = floor(N_t / 2)
w_k    = ln(mu + 0.5) - ln(k),  k = 1..mu,  normallashtirilgan
mu_eff = 1 / sum(w_k^2)
c_c    = (4 + mu_eff/D) / (D + 4 + 2 mu_eff/D)
c_1    = 2 / ((D + 1.3)^2 + mu_eff)
c_mu   = min(1 - c_1, 2 (mu_eff - 2 + 1/mu_eff) / ((D + 2)^2 + mu_eff))
```

Har avlodda **qabul qilingan** avlodlarning eng yaxshi `mu` tasi bo'yicha:

```
m_old <- m;   m <- sum_k w_k x_(k)
y_w    = (m - m_old) / sigma
p_c   <- (1 - c_c) p_c + sqrt(c_c (2 - c_c) mu_eff) y_w
y_k    = (x_(k) - m_old) / sigma
C     <- (1 - c_1' - c_mu') C + c_1' p_c p_c^T + c_mu' sum_k w_k y_k y_k^T
c_1'   = c_1 / (1 + nu),   c_mu' = c_mu / (1 + nu)
```

**Qadam hajmi CSA bilan emas, populyatsiya tarqalishidan:**
`sigma = mean_j( std_i( P[i,j] ) )`.
*Sabab:* CSA Gauss namunasini faraz qiladi; bizda avlodlarning bir qismi
DE mutatsiyasidan keladi. Populyatsiya tarqalishi taqsimotdan mustaqil
va qo'shimcha parametr talab qilmaydi.

**Dangasa xos yoyilma:** `fes - eigeneval > 1/(10 D (c_1 + c_mu))` da
qayta hisoblanadi; `C <- (C + C^T)/2`; xos qiymatlar `>= 1e-20 * max`
gacha qirqiladi. Narx `O(D^3)` → amortizatsiya `O(D^2)`.

**Ochiq metodologik izoh:** `y_k` barcha qabul qilingan avlodlardan
olinadi, faqat CMA tarmog'idan emas. Bu darslikdagi CMA baholagichi
emas — bu *muvaffaqiyatli qadamlarning kovariatsiyasi*, va crossover
bazisi uchun aynan shu kerak. B4 da faqat-CMA-tarmoq varianti bilan
taqqoslanadi.

### 3.4 Bitta model, ikki iste'molchi

**(a) Eigen-bazisda crossover** (`p_eig` ehtimollik bilan):
`U_i = ( binom_cross( V_i @ B, x_i @ B, CR_i ) ) @ B^T`
(satr vektorlar uchun `x @ B` = `B^T x` — joriy koddagi shakl to'g'ri.)

**(b) CMA taqsimotidan namuna** (`p_cma_eff` ulush bilan):
`z ~ N(0,I);  U_i = m + sigma (B @ (d * z))`,
`p_cma_eff = p_cma / (1 + lambda_nu * nu)`

Alohida quyruq fazasi **yo'q**: `TAIL`, `TAIL_FRAC`, `TAIL_DIV` olib
tashlanadi.

### 3.5 DE yadrosi va rejim ansambli (saqlanadi)

```
p_num  = max(2, round((P_max - (P_max - P_min) t) N_t))
r1: RSP vaznlari w_k = K (N-k)/N + 1,  r1 != i
r2 ~ P u A,  r2 != i, r1
V_i = x_i + Fw_i (x_pbest - x_i) + F_i (x_r1 - x_r2)

bank_i ~ Bernoulli(1 - p_bank)
CR_i = clip(N(mu_CR, 0.1), 0, 1);  mu_CR < 0 -> 0
F_i  ~ Cauchy(mu_F, 0.1) -> (0, 1]
bank 0, t < 0.6:  F_i <- min(F_i, 0.7)
bank 0:  Fw_i = F_i * (0.7 | 0.8 | 1.2)  agar t < 0.2 | t < 0.4 | aks holda
bank 1:  Fw_i = F_i
```

Chegara: midpoint-target, ikkala tarmoq uchun bir xil.
`rho` prior "issiq start" olib tashlanadi.

### 3.6 Shovqin bahosi va uch ta'siri (N3 — markaziy hissa)

Har `g_nu` avlodda joriy eng yaxshi nuqta qayta baholanadi:

```
s  = IQR(fitness) + eps
nu <- (1 - alpha) nu + alpha * |f_1(x*) - f_2(x*)| / s
```

| Qayerda | Formula | Nega |
|---|---|---|
| Tanlov | qabul: `f(u_i) <= f(x_i) - kappa * nu * s` | Shovqin ichidagi "yaxshilanish" populyatsiyani suradi |
| Kovariatsiya | `c_1/(1+nu)`, `c_mu/(1+nu)` | Shovqinli reyting modelni buzadi |
| CMA tarmog'i | `p_cma/(1 + lambda_nu * nu)` | Shovqinda taqsimot modeli foydasiz |

`nu = 0` da tanlov sharti aynan `f(u) <= f(x)` ga qaytadi.

**Koshi shovqini uchun ehtiyot:** `f_1 - f_2` ayirmasi og'ir dumli
taqsimotda portlashi mumkin. Shuning uchun ayirma `IQR` ga
normallashtiriladi va `nu` yuqoridan `nu_max` bilan chegaralanadi
(`nu_max` sozlanadi). Bu B5 da alohida tekshiriladi.

### 3.7 Birlashgan kredit arbitraji (N2)

Uchala ulush (`p_bank`, `p_eig`, `p_cma`) bir xil qoida bilan:

```
gain_i = max(f(x_i) - f(u_i), 0)
har guruh g uchun:
  share_g = |g| / N_t
  FIR_g   = (sum_{i in g} gain_i / sum gain) / share_g
  c_g    <- (1 - eta) c_g + eta * FIR_g
  p_g     = clip(c_g / sum c, 0.02, 0.98)
```

### 3.8 LPSR

`N_{t+1} = max(N_min, round(N_init + (N_min - N_init) fes / max_fes))`

## 4. Chiqish: avvalgi struktura + yangi kategoriya tahlili

### 4.1 Avvalgi struktura o'zgarishsiz

Eksperiment tugagach **avtomatik** hosil bo'ladi:

```
tables/summary_mean_std.csv       tables/friedman_test.txt
tables/summary_results.csv        tables/friedman_posthoc.csv
tables/summary_table.tex          tables/wilcoxon_pvalues.csv
tables/overall_average_ranks.csv  tables/win_tie_loss.csv
tables/complexity.csv             figures/Conv_<F>_<D>.png
raw_data/full_raw_results.csv     raw_data/curves.npz
```

`PRECISION_FLOOR = 1e-8`, xom qiymatlar saqlanadi. Bo'lak/birlashtirish
mexanizmi (`MERGE`, `OUT_DIR`, `DIMS`, `RUNS`) va Actions ish oqimi ham
saqlanadi.

### 4.2 YANGI: COCO me'yoriy ko'rsatkichlari

BBOB jamoasi yakuniy xatoni emas, **maqsadga yetish vaqtini**
kutadi. Shuning uchun qo'shimcha:

```
tables/ert.csv          ERT (expected running time) maqsad darajalari bo'yicha
tables/ecdf.csv         byudjet ichida yechilgan (funksiya, maqsad) juftliklari ulushi
figures/ecdf_<D>.png    ECDF grafiklari
```

Maqsad darajalari: `10^2, 10^1, 10^0, 10^-1, ..., 10^-8`.

### 4.3 YANGI: kategoriya tahlili (majburiy)

```
tables/category_ranks.csv        tables/category_wtl_<rival>.csv
tables/category_summary.md       tables/dimension_category.csv
```

**Asosiy maydon (bbob-noisy) uchun uchta o'q:**

| O'q | Kategoriyalar |
|---|---|
| A — shovqin darajasi | mo'tadil (f101–f106) / kuchli (f107–f121) / kuchli + ko'p ekstremumli (f122–f130) |
| B — shovqin modeli | Gauss / tekis / Koshi (f101–f130 da 3 davr bilan almashadi — **B0 da dasturiy tasdiqlansin**) |
| C — o'lcham | 5 / 10 / 20 |

**Ikkilamchi maydon (CEC-2022) uchun:** bir ekstremumli (F1) / asosiy
ko'p ekstremumli (F2–F5) / gibrid (F6–F8) / kompozitsiya (F9–F12).

`category_summary.md` har kategoriya x har raqib uchun:

| Kategoriya | Raqib | + | = | − | Bizning rank | Uning ranki | Holat |
|---|---|---|---|---|---|---|---|

**Holat qoidasi:** `YUTDIK` faqat Holm tuzatilgan `p < 0.05` bo'lganda.
O'rtacha kichikroq bo'lishi yetarli emas.

Har kategoriya uchun matnda: qaysi funksiyada yutdik va qanchaga
(raqam bilan), qaysida yutqazdik va qanchaga, sabab bo'yicha taxmin
(qaysi komponent ishladi yoki ishlamadi).

## 5. Bosqichlar

| Bosqich | Mazmun | Chiqish |
|---|---|---|
| **B0** | `ace_shade.py`: ACE-SHADE + 10 raqobatchi (UH-CMA-ES bilan) + BBOB/CEC yuklagichlari + barcha jadval generatorlari + birlik testlari. Shovqinsiz baholash masalasi (2-bo'lim) hal qilinadi. B o'qi guruhlashi tasdiqlanadi | kod, testlar yashil |
| **B1** | CEC-2017 da `r_N, eta` sozlash; o'z sintetik shovqinli to'plamimizda `kappa, lambda_nu, g_nu, alpha, nu_max` sozlash | `results/tuning_v2/` |
| **B2** | Parametrlarni muzlatish va e'lon qilish | `docs/ABLATION_V2.md` |
| **B3** | **Asosiy:** bbob-noisy, 30 F x 3 D x 15 instansiya x 10 alg, byudjet `10^4 * D` | `results/noisy/` |
| **B4** | Ablatsiya: N1, N2, N3 alohida o'chirilgan + faqat-CMA-tarmoq varianti | `results/ablation_v2/` |
| **B5** | Koshi shovqini bo'yicha maxsus tekshiruv (3.6 dagi `nu_max`) | `results/cauchy/` |
| **B6** | **Ikkilamchi:** CEC-2022, 12 F x 2 D x 30 run x 10 alg | `results/cec2022/` |
| **B7** | `docs/RESULTS_V2.md` + kategoriya tahlili (4.3) | hisobot |

**Shardlash:** B3 funksiya bo'yicha 30 ta parallel ish; B6 funksiya
bo'yicha 12 ta ish (D = 20 da `MaxFES = 10^6` — eng og'ir bo'lak).

### B0 birlik testlari (o'tmasa keyingi bosqichga o'tilmaydi)

1. `C` simmetrik va musbat yarim aniq qoladi (1000 avlod)
2. `B diag(d^2) B^T ~ C`; dangasa davr to'g'ri
3. `nu = 0` da tanlov sharti aynan `f(u) <= f(x)`
4. Baholash hisoblagichi `MaxFES` dan oshmaydi (barcha algoritmlar)
5. Chegara ishlovi barcha nuqtalarni `[lb, ub]` ichida saqlaydi
6. Bir xil seed → bir xil natija
7. CEC-2022 va CEC-2017 funksiyalari optimumda rasmiy bias beradi
8. `bbob-noisy` da bir xil `x` uchun ketma-ket baholashlar farq qiladi
9. Tavsiya qilingan yechim baholashi byudjetga kirmaydi

## 6. Xavflar

| Xavf | Javob |
|---|---|
| UH-CMA-ES ni to'g'ri amalga oshirish | Maqoladagi psevdokod; sanity-test; nomuvofiqlik ochiq yoziladi |
| Raqiblarning qayta amalga oshirilishi e'lon qilingan natijalardan past | CEC-2017 da tartib bilan solishtirish; farq hisobotda ko'rsatiladi |
| COCO natijalar arxivi bloklangan (403) | Muhit sozlamalarida `numbbo.github.io` ga ruxsat berilsa, muallif natijalari bilan taqqoslash ochiladi — bu qayta amalga oshirish xavfini yo'q qiladi |
| Koshi shovqinida `nu` portlashi | `nu_max` chegarasi; B5 |
| `y_k` ni DE avlodlaridan olish | B4 ablatsiyasi |
| 20D da CEC-2022 byudjeti `10^6` | Funksiya bo'yicha shardlash |
