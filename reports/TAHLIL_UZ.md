# TEMOA_V10_HYBRID — batafsil tahlil, diagnostika va tuzatish

**Sana:** 2026-09-25 · **Obyekt:** `hybrid 2026.py` (635 qator) · **Protokol:** `run_all.py`

Bu hujjatdagi har bir raqam o'lchangan, taxmin qilinmagan. Barcha o'lchovlar
`results_prelim/` (30D, rotated suite, 12 ta mustaqil yurish) va matnda
ko'rsatilgan maxsus diagnostik yurishlardan olingan. To'liq protokol (30/50/100D,
30 yurish) sizning mashinangizda bajariladi; uni ishga tushirish `README.md` da.

---

## 1. Qisqacha xulosa

Uchta natija, muhimlik tartibida:

**1.1. Gibrid o'zining yadrosidan yomonroq ishlaydi.** Kod izohida TEMOA'ning
o'zagi L-SHADE/jSO ekani ochiq aytilgan. Ammo taqqoslashda na L-SHADE, na jSO
bor. Men ikkalasini ham yozib qo'shganimda, asl TEMOA_V10 ularga ko'pchilik
funksiyalarda yutqazadi, Schwefel'da esa ~6 tartibga (30D, median error):

| 30D, median error | Schwefel | RotatedElliptic | NoisyRastrigin | BentCigar |
|---|---|---|---|---|
| **TEMOA_V10** | 2.05e+03 | 2.01e+04 | 3.58e+01 | 2.43e+00 |
| jSO | 3.43e+00 | 6.47e+02 | 1.88e+01 | 5.96e+01 |
| L-SHADE | 9.99e-02 | 1.38e+03 | 2.16e+01 | 3.18e+01 |

Bu farq populyatsiya hajmi bilan tushuntirilmaydi — populyatsiya tenglashtirilgan
nazoratda ham saqlanadi (§3.4).

**1.2. Taqqoslash adolatsiz tuzilgan.** Ikkita baseline nashr etilgan shaklidan
zaiflashtirilgan, zamonaviy DE esa umuman yo'q (§2.1). Bu holatda "TEMOA hammasidan
ustun" degan xulosa tekshirib bo'lmaydigan da'vo bo'lib qoladi.

**1.3. Qulashning sababi aniqlandi va tuzatildi.** Ayb — WOA spirali (operator 2):
u populyatsiyani `x_pbest` ga yig'ib, **darhol katta foyda beradi**, lekin
kelajakdagi diversity'ni yo'q qiladi. Adaptiv operator tanlash buni ko'ra olmaydi,
chunki zarar kechikkan (§3.3). Tuzatilgan **TEMOA_V11** 30D da Schwefel'da
2.05e+03 → 1.25e-01 (16 000×) va RotatedElliptic'da 2.01e+04 → 2.77e+01 (725×)
beradi, va jSO'ni 8 funksiyadan 7 tasida yutadi (§4.2).

> **Eng muhim tavsiya:** maqolani hozirgi holatida "V10 hammasidan ustun" deb
> yubormang. Ma'lumot bu da'voni qo'llab-quvvatlamaydi. Qo'llab-quvvatlanadigan,
> va aslida qiziqarliroq bo'lgan da'vo §6 da.

---

## 2. Metodologik tanqid

### 2.1. Baseline'lar nashr etilgan shaklidan zaif

| Baseline | Kodda nima bor | Nashr etilgan shakli | Oqibati |
|---|---|---|---|
| `baseline_PSO` | c₁=c₂=2.0, w:0.9→0.4, Vmax yo'q, constriction yo'q | Clerc–Kennedy: χ=0.7298, c=2.05, Vmax bilan | Poli/Trelea turg'unlik sohasidan tashqarida → **divergent**. O'lchandi: NoisySphere 10D da asl konfiguratsiya 6.79e+02, constriction bilan 1.18e-02 — **57 000× farq** |
| `baseline_HHO` | Levy "rapid dive" bosqichi **butunlay yo'q** (`levy` so'zi funksiyada uchramaydi) | Heidari va b. (2019): 4 ta exploitation tarmog'idan 2 tasi Levy dive ishlatadi | TEMOA aynan shu komponentni HHO'dan qarz olgan va kredit bergan. Uni baseline'dan olib tashlash — o'z foydasiga taqqoslash |
| Zamonaviy DE | **Yo'q** | L-SHADE (2014), jSO (2017) | TEMOA'ning o'z yadrosi taqqoslashda yo'q → gibridlashtirish hissasi o'lchanmagan |

Qo'shimcha: barcha baseline'lar `NP=30` da qotirilgan, TEMOA esa `NP=6·D`
(30D→180, 100D→500, ya'ni 6×–16.7×). Bu o'z-o'zidan ulkan ustunlik. O'lchandi:
TEMOA'ni `NP=30` ga tushirganda u baseline'lar darajasiga tushadi
(NoisyRastrigin 30D: NP=180 da 31.9, NP=30 da 74.6).

### 2.2. Benchmark to'plami separabellikka og'ishgan

12 ta masaladan **faqat 1 tasi aylantirilgan** (`RotatedElliptic`). Qolganlari
faqat siljitilgan, ya'ni separabel yoki deyarli separabel bo'lib qoladi. Bu
koordinata bo'yicha ishlaydigan binomial crossover'ni — ya'ni DE oilasini —
sun'iy ravishda ustun qiladi. CEC to'plamlarida aylantirish standart.
`--suite rotated` (yangi standart) buni tuzatadi, `--suite shift` esa asl
to'plamni qayta ishlab chiqaradi.

Yana: `composition` funksiyasi aslida CEC composition emas — u shunchaki
0.5·Ackley + 0.5·Sphere, aylantirish, bias va komponent vaznlari yo'q. Nomi
chalg'ituvchi.

### 2.3. Hisobot qilinadigan qiymat xato

Kod `f(x)` ni chiqaradi, `f(x) − f*` ni emas. Schwefel uchun `f* ≠ 0`:

| dim | 30 | 50 | 100 |
|---|---|---|---|
| Schwefel `f*` | 3.818e-04 | 6.364e-04 | 1.273e-03 |

Ya'ni masalani **mukammal** yechgan algoritm ham log o'qida 1e-03 da "to'xtagan"
ko'rinadi. Yangi kodda barcha qiymatlar error sifatida beriladi.

### 2.4. Dinamik va shovqinli masalalar noto'g'ri o'lchanadi

- **DynamicSphere**: `best-so-far` harakatlanuvchi optimum uchun umuman
  o'lchov emas — u faqat "optimum yonidan o'tganda tasodifan yaqin nuqta
  tanlanganmi" degan savolga javob beradi va hech qachon yomonlasha olmaydi.
  To'g'ri o'lchov — **offline error** (Branke). Yangi kodda shu.
- **Noisy**: `best_x` shovqinli qiymat bo'yicha tanlanib, keyin shovqinsiz
  baholanadi. Error shovqin amplitudasidan (0.5) pastga tushgach, tanlov
  butunlay shovqin qur'asi bilan hal bo'ladi — bu **ekstremal-qiymat
  yonbosishi**: optimum atrofida ko'proq nuqta olgan algoritm omad bilan yutadi.
  Yangi kodda tavsiya qoidasi: eng yaxshi 12 nomzod 25 martadan qayta baholanadi
  va o'rtachasi bo'yicha eng yaxshisi tavsiya etiladi.

### 2.5. Statistik tahlil

| Muammo | Kodda | To'g'risi |
|---|---|---|
| Friedman bloklari | 36 blok = 12 funksiya × 3 o'lcham bitta testda | Bir funksiyaning 30/50/100D varianti mustaqil blok emas (bir xil landshaft, shift, rotation) → p **anti-konservativ**. Har o'lcham uchun alohida test |
| Post-hoc | **Yo'q** | Friedman faqat "farq bor" deydi. Holm post-hoc qo'shildi + Nemenyi critical difference |
| Juftlanganlik | `mannwhitneyu` (juftlanmagan) | Yurishlar bir xil masala nusxasida, mos indeks bilan → **Wilcoxon signed-rank** kuchliroq |
| Ko'plik nazorati | Holm har katakda 6 ta test ustida (36 ta alohida oila) | Aslida 216 ta test bajariladi. Endi ham katak ichida, ham butun oila bo'yicha Holm |
| Effekt hajmi | **Yo'q** | Vargha–Delaney Â₁₂ + Cliff's delta qo'shildi |
| Deskriptiv | faqat mean/std | best/worst/mean/std/**median**/IQR |
| Iman-Davenport | yo'q | Friedman χ² kam bloklarda konservativ — F statistikasi ham beriladi |

### 2.6. Takrorlanuvchanlik

Global `np.random.seed()` ishlatiladi (`Generator` emas), `requirements.txt`
yo'q, urug'lar va muhit qayd etilmaydi, vaqt o'lchanmaydi. Yangi kodda har
yurishga alohida `default_rng([42, dim, run, alg_index])`, shovqin oqimi
algoritmlar bo'yicha moslashtirilgan, `manifest.json` da versiyalar/urug'lar/
platforma, har yurishda wall-clock.

### 2.7. Kod darajasidagi nuqsonlar

- **Eigen bazis rank-deficient bo'lib qoladi.** `np.cov(pop[:N/2])` dan
  quriladi, LPSR esa `N` ni `D` dan ancha pastga tushiradi. 100D da byudjetning
  90% ida 100×100 kovariatsiya **25 ta nuqtadan** baholanadi — bazis asosan
  shovqin. (V11 da `n_samples > dim` sharti qo'yildi.)
- **Latent cheksiz sikl.** `r2` tanlashdagi `while np.any(clash)` sikli
  `pop_size == 2` bo'lsa hech qachon tugamaydi. Hozir `N_min=4` himoya qiladi,
  lekin bu tasodifiy himoya. (V11 da guard qo'yildi.)
- `np.cov` eigenvalue'lari tashlab yuboriladi (`_, B = eigh(C)`) — bu to'g'ri,
  lekin `B` ning ustunlari tartibi eigenvalue bo'yicha, shuning uchun eng
  informativ yo'nalishlar oxirida qoladi; kodda bunga bog'liqlik yo'q, ammo
  hujjatlashtirilmagan.

---

## 3. Diagnostika: nima ishlaydi, nima ishlamaydi

### 3.1. Operator ablatsiyasi (30D, rotated, 8 yurish, median error)

Har bir konfiguratsiya — **bir xil kod yo'li**, faqat `OPS` bayrog'i o'zgargan,
shuning uchun farq faqat o'sha operatorga tegishli.

| Konfiguratsiya | Schwefel | NoisyRastrigin | Ackley | RotatedElliptic | Rosenbrock |
|---|---|---|---|---|---|
| FULL (0,1,2,3) | 1.86e+03 | 3.53e+01 | 7.11e-15 | 1.49e+04 | 1.22e+01 |
| op0 only (pbest-DE) | 5.92e+01 | **1.51e+01** | 1.42e-14 | **2.19e+02** | 1.33e+01 |
| op0+op1 (leader/GWO) | 1.18e+03 | 2.04e+01 | 1.07e-14 | 1.32e+04 | **8.55e+00** |
| op0+op2 (spiral/WOA) | 2.38e+03 | 4.18e+01 | 1.42e-14 | 9.97e+02 | 1.85e+01 |
| op0+op3 (Levy/HHO) | **2.79e-09** | 1.98e+01 | 7.11e-15 | 1.39e+03 | 1.34e+01 |
| FULL, ES dumisiz | 1.86e+03 | 3.53e+01 | 1.07e-14 | 1.48e+04 | 1.21e+01 |
| jSO (mos yadro) | 4.51e+00 | 2.29e+01 | 3.70e-13 | 7.32e+02 | 1.86e+01 |

Xulosalar:

- **Operator 2 (WOA spirali) dominatsiya qilingan** — uni op0 ga qo'shish
  sinalgan **har bir funksiyada** op0 ning o'zidan yomonroq. Bu shunchaki
  foydasiz emas, zararli.
- **Operator 1 (leader/GWO) faqat Rosenbrock'da yordam beradi** (8.55 vs 13.3),
  qolgan joylarda zarar: RotatedElliptic'da 219 → 1.32e+04 (60×).
- **Operator 3 (Levy/HHO) Schwefel'da hal qiluvchi**: op0+op3 = 2.79e-09, ya'ni
  masala amalda yechilgan. Boshqa joylarda neytral yoki biroz zarar.
- **Operator 0 (jSO pbest-DE) — asosiy ishchi kuch.**
- **(1+1)-ES dumi hech narsa bermaydi.** FULL va FULL-ES'siz ikkita funksiyada
  bit-ma-bit bir xil, qolganlarida shovqin doirasida. Byudjetning 5% i (30D da
  4500 FES) behuda ketadi.

### 3.2. Eigen-crossover — yagona o'zini oqlagan qo'shimcha

RotatedElliptic, 30D, op0 only: eigen bilan **2.19e+02**, eigensiz **1.35e+04**.
**62× farq.** Bu nazariy jihatdan ham kutilgan: aylantirilgan, yomon
shartlangan masalada koordinata bo'yicha crossover ishlamaydi, kovariatsiya
bazisida esa ishlaydi. Boshqa (separabel) masalalarda foyda yo'q yoki manfiy,
va "adaptiv" `P_EIG` uni kerak bo'lmaganda o'chira olmaydi.

### 3.3. Nega adaptiv operator tanlash qutqarmaydi — va mening gipotezam qanday rad etildi

Portfelda Schwefel'ni 2.79e-09 gacha yechadigan kombinatsiya (op0+op3) **bor**,
lekin to'liq portfel 1.86e+03 beradi — 12 tartib yomonroq. Nega AOS to'g'ri
operatorni topa olmaydi?

Men avval **CR_FLOOR** ayb deb o'yladim (CR ≥ 0.7 majburlash separabel
multimodal masala uchun noto'g'ri). **Sinab ko'rdim — gipoteza rad etildi:**
`CR_FLOOR=False` da Schwefel 2685 → 2350, ya'ni sabab bu emas.

Keyin **kredit taqsimoti** ayb deb o'yladim (binar "muvaffaqiyat ulushi" kam
uchraydigan lekin katta foydali Levy'ni jazolaydi) va uni ΔF-ga asoslangan
kreditga almashtirmoqchi bo'ldim. V10 ni instrumentlashtirib o'lchadim
(Schwefel, 30D, bitta yurish):

| | op0 pbest-DE | op1 leader | op2 spiral | op3 Levy |
|---|---|---|---|---|
| o'rtacha muvaffaqiyat ulushi (AOS shuni mukofotlaydi) | 0.103 | 0.127 | 0.120 | 0.060 |
| **umumiy fitness foydasining ulushi** | 11.1% | 14.2% | **66.1%** | 8.6% |
| ajratilgan ehtimollik (t=0.75 da) | 0.175 | 0.086 | **0.644** | 0.095 |

**Bu mening ikkinchi gipotezamni ham rad etadi.** Spiral fitness yaxshilanishining
66% ini beradi — ΔF-ga asoslangan kredit unga **yanada ko'proq** ehtimollik
bergan bo'lardi. Ya'ni "yaxshiroq kredit formulasi" yechim emas.

Haqiqiy mexanizm shu: **spiral qarz olib ishlaydi.** U populyatsiyani `x_pbest`
atrofiga yig'ib, tez pasayadi — bu har qanday qisqa muddatli kredit uchun
a'lo ko'rinadi. To'lov esa kechikkan va taqsimlangan: yo'qolgan diversity
Levy'ning sakrashlarini foydasiz qiladi. **Darhol yaxshilanishga asoslangan
hech qanday kredit sxemasi buni ko'ra olmaydi.** Shuning uchun V11 da javob —
formulani o'zgartirish emas, balki kredit ko'ra olmaydigan narsani
(populyatsiya diversity'sini) **bevosita o'lchaydigan qo'riqchi**.

### 3.4. Populyatsiya hajmi — hisobot qilinmagan ulkan sezgirlik

30D, median error, `POP_FACTOR` o'zgarishi:

| | NoisyRastrigin | RotatedElliptic | Rosenbrock | Schwefel |
|---|---|---|---|---|
| TEMOA_V10, NP=6·D=180 | 3.19e+01 | 1.45e+04 | 5.59e-01 | 2.68e+03 |
| TEMOA_V10, NP=18·D=500 | 2.29e+01 | **2.19e+00** | 1.18e+01 | 1.54e+03 |
| nisbat | 1.4× | **6600× yaxshi** | 21× yomon | 1.7× |

`POP_FACTOR=6` 10D da sozlangan va 30D+ uchun noto'g'ri kalibrlangan. Bu
parametr sezgirligi maqolada umuman ko'rsatilmagan, holbuki u ba'zi
funksiyalarda gibridlashtirishdan ko'ra kattaroq ta'sir beradi.

Populyatsiya tenglashtirilgan nazorat (§1.1 dagi da'voni tekshiradi) —
TEMOA/L-SHADE nisbati, >1 = TEMOA yomonroq:

| NP teng | NoisyRastrigin | RotatedElliptic | Rosenbrock | Schwefel | Zakharov |
|---|---|---|---|---|---|
| NP=180 | 463× yomon | 0.93× (teng) | 6.5× yomon | **7.0e+06× yomon** | 3571× **yaxshi** |
| NP=500 | 22× yomon | **234× yaxshi** | 1.1× (teng) | **4.5e+04× yomon** | 16× **yaxshi** |

Ya'ni: farq populyatsiya hajmidan emas. Va e'tibor bering — TEMOA Zakharov va
(katta NP da) RotatedElliptic'da **haqiqatan yutadi**. Bu §6 uchun muhim.

---

## 4. TEMOA_V11 — o'lchovga asoslangan tuzatish

### 4.1. Nima o'zgardi va nega

| # | O'zgarish | Asos (o'lchangan) |
|---|---|---|
| a | `OPS = (0,1,3)` — spiral olib tashlandi | §3.1: op2 sinalgan har bir funksiyada dominatsiya qilingan |
| b | `LS_FRACTION = 0` — ES dumi yo'q | §3.1: hissasi o'lchov doirasidan past; 5% byudjet qaytariladi |
| c | eigen-crossover `n_samples > dim` sharti bilan | §2.7: 100D da 90% byudjetda bazis 25 nuqtadan quriladi |
| d | `P_MIN = 0.02` (0.05 o'rniga) | 0.05 floor har bir operatorga butun yurish davomida 5% kafolatlaydi — dominatsiya qilingan operatorni o'chirib bo'lmaydi |
| e | **diversity guard**: normallashgan diversity `DIV_THRESH` dan pastga tushsa, operator taqsimoti Levy foydasiga majburan o'zgaradi | §3.3: kredit populyatsiya qulashini ko'ra olmaydi; bu ko'radi |

**Sinab ko'rilgan va tashlab yuborilgan:** ΔF-ga asoslangan kredit (§3.3 — u
zararli operatorni ko'proq mukofotlagan bo'lardi) va `CR_FLOOR` ni o'chirish
(§3.3 — Schwefel'ga ta'sir qilmadi).

**Parametrlarni sozlash.** `POP_FACTOR`, `DIV_THRESH`, `DIV_BOOST` **10D da**,
uchta funksiyada (Ackley, Schwefel, RotatedElliptic), 3×3 gridda, 10 yurishda
tanlandi; mezon — `log10(median error)` ning o'rtachasi. G'olib:
`POP_FACTOR=12, DIV_THRESH=1e-4, DIV_BOOST=0.5`. Shundan keyin **muzlatildi**.
Hisobot qilinadigan 30/50/100D natijalariga qarab hech bir parametr
tanlanmagan.

### 4.2. V11 natijalari (30D, rotated, 10 yurish, median error)

| | Schwefel | NoisyRastrigin | Ackley | Griewank | RotatedElliptic | Rosenbrock | Zakharov | BentCigar |
|---|---|---|---|---|---|---|---|---|
| TEMOA_V10 | 2.05e+03 | 3.58e+01 | **7.11e-15** | 0.00e+00 | 2.01e+04 | **1.24e+01** | **1.44e-20** | **2.43e+00** |
| **TEMOA_V11** | **1.25e-01** | **1.92e+01** | 1.42e-14 | 0.00e+00 | **2.77e+01** | 1.39e+01 | 1.04e-14 | 8.58e+00 |
| jSO | 3.43e+00 | 1.88e+01 | 3.70e-13 | 1.11e-16 | 6.47e+02 | 1.85e+01 | 1.93e-10 | 5.96e+01 |
| L-SHADE | 9.99e-02 | 2.16e+01 | 2.09e-12 | 5.55e-17 | 1.38e+03 | 1.86e+01 | 1.07e-10 | 3.18e+01 |

**V11 yutgan joylar:** Schwefel V10 dan 16 000×, RotatedElliptic 725×,
NoisyRastrigin 1.9×. jSO ga nisbatan 8 funksiyadan 7 tasida yaxshiroq,
bittasida teng (NoisyRastrigin, 1.02×).

**V11 yutqazgan joylar — bularni yashirmaslik kerak:**
- **BentCigar**: 8.58 vs V10 ning 2.43 (3.5× yomon). V11 baribir jSO (59.6) va
  L-SHADE (31.8) dan yaxshi, lekin V10 dan yomon.
- **Zakharov**: 1.04e-14 vs 1.44e-20. Ikkalasi ham amalda nol, lekin farq real.
- **Rosenbrock**: 13.9 vs 12.4 (1.12× yomon). Bu kutilgan edi — §3.1 ga ko'ra
  Rosenbrock aynan op1 (leader) dominant bo'lganda yaxshi ishlaydi, V11 esa
  `P_MIN` ni pasaytirib va diversity qo'riqchisini qo'shib, uning
  dominantligini kamaytiradi.
- **Ackley**: 1.42e-14 vs 7.11e-15 — ikkalasi ham mashina aniqligida, farq
  ahamiyatsiz.

Ya'ni V11 ham universal ustun emas. Bu — kutilgan va to'g'ri natija (§6).

---

## 5. Hisoblash xarajati

10D, bitta yurishga o'rtacha wall-clock (4 yadroli konteyner, 12 ta yurish):

| algoritm | sekund | | algoritm | sekund |
|---|---|---|---|---|
| jSO | 0.19 | | PSO | 0.30 |
| L-SHADE | 0.19 | | TEMOA_V11 | 0.33 |
| HHO | 0.26 | | TEMOA_V10 | 0.42 |
| WOA | 0.25 | | SCA | 0.44 |
| GWO | 0.46 | | DE | 0.59 |

TEMOA_V10 jSO dan ~2.2× qimmat (eigendekompozitsiya har avlodda `O(D³)`).
V11 arzonroq, chunki eigen-gate rank-deficient holatda eigendekompozitsiyani
o'tkazib yuboradi. `O(D³)` 100D da hali ham arzon (funksiya baholashga
nisbatan), lekin 1000D da hukmron bo'ladi — **masshtablanish chegarasi shu
yerda**, va maqolada aytilishi kerak.

Xotira: `O(N·D)` populyatsiya + arxiv, `O(D²)` kovariatsiya. 100D, N=500 da
ahamiyatsiz.

---

## 6. No-Free-Lunch: TEMOA qaysi masala sinfiga mos?

Ma'lumot "universal ustunlik" ni qo'llab-quvvatlamaydi va hech qachon
qo'llab-quvvatlamaydi ham — NFL bo'yicha bu kutilgan. Ammo ma'lumot **aniq,
tor va himoya qilinadigan** da'voni qo'llab-quvvatlaydi:

> TEMOA'ning kovariatsiya-bazisli crossover'i **yomon shartlangan, aylantirilgan,
> separabel bo'lmagan** masalalarda haqiqiy ustunlik beradi: RotatedElliptic'da
> op0-only konfiguratsiyasi eigen bilan 2.19e+02, eigensiz 1.35e+04 (62×), va
> V11 shu masalada jSO dan 23×, L-SHADE dan 50× yaxshi. Zakharov'da ham TEMOA
> oilasi L-SHADE'dan barqaror ustun (§3.4).

Bu — nashrga arzigulik natija. U "12 funksiyada hammasini yutdik" dan ko'ra
ancha ishonchliroq, chunki mexanizmi tushuntirilgan va ablatsiya bilan
izolyatsiya qilingan.

Aksincha, **deceptive multimodal** masalalar (Schwefel) TEMOA_V10 uchun aniq
zaif sinf, va buning sababi ham aniqlangan (§3.3).

---

## 7. Nashr uchun aniq tavsiyalar

1. **Da'voni qayta joylashtiring.** "Universal ustunlik" o'rniga: "eigen-bazisli
   adaptiv crossover + diversity-qo'riqchi bilan boshqariladigan operator
   portfeli yomon shartlangan aylantirilgan masalalarda jSO/L-SHADE ni ortda
   qoldiradi". Yutqazgan joylarni jadvalda ko'rsating.
2. **L-SHADE va jSO ni taqqoslashga qo'shing.** Ularsiz maqola qabul
   qilinmaydi — reviewer birinchi savoli shu bo'ladi.
3. **Baseline'larni nashr etilgan shakliga qaytaring** (PSO constriction,
   HHO Levy dive bilan). Kodda ikkalasi ham bor.
4. **Ablatsiya jadvalini kiriting** (§3.1). Bu maqolaning eng kuchli qismi:
   har bir komponentning hissasi izolyatsiya qilingan.
5. **CEC'2017/2022 to'plamini qo'shing.** Hozirgi 12 ta klassik funksiya
   yetarli emas; hech bo'lmasa CEC'2017 ning 30 ta funksiyasini qo'shing.
6. **Parametr sezgirligini ko'rsating** (§3.4) — `POP_FACTOR` ta'siri ba'zi
   funksiyalarda gibridlashtirishdan katta.
7. **Statistikani to'ldiring**: median/IQR, har o'lcham uchun Friedman,
   Holm post-hoc, Â₁₂ effekt hajmi. Barchasi `analyze.py` da tayyor.
8. **Qo'shimcha domenlar.** Hozir faqat cheklanmagan uzluksiz optimizatsiya
   bor. Sizning tadqiqot yo'nalishingiz (kiberxavfsizlik, AI) uchun: feature
   selection (binar variant), hyperparameter tuning, cheklovli muhandislik
   masalalari. Bularsiz "amaliy ahamiyat" da'vosi asossiz.

---

## 8. Takrorlash

```bash
pip install -r requirements.txt
python tests/test_all.py                                    # 16/16 o'tishi kerak
python run_all.py --dims 30 50 100 --runs 30 --jobs 20
python analyze.py --out results --control TEMOA_V11
python experiments/ablation.py    --dim 30 --runs 15 --jobs 20
python experiments/sensitivity.py --dim 30 --runs 15 --jobs 20
```

Windows uchun: `run_windows.bat` (hammasini ketma-ket bajaradi, uzilsa davom
ettiradi).

Sizning mashinangizda (i7-13700F, 16 yadro / 24 oqim) to'liq protokol ~2 soat.

### Bu hisobotdagi raqamlarning manbai

| Bo'lim | Manba | Yurishlar |
|---|---|---|
| §1.1, §4.2 | 30D, rotated suite | 10 |
| §3.1 | operator ablatsiyasi, 30D | 8 |
| §3.3 (instrumentatsiya) | `logger=` bilan bitta yurish | 1 |
| §3.4 | POP_FACTOR sweep, 30D | 5 |
| §5 | 10D smoke | 12 |

Bular **dastlabki** o'lchovlar: statistik testlar uchun yurishlar soni kam
(Wilcoxon signed-rank n=10 da minimal p ≈ 0.002, n=8 da ≈ 0.008). Yakuniy
maqola raqamlari uchun 30 yurishli to'liq protokolni o'z mashinangizda ishga
tushiring — barcha xulosalar yo'nalishi shu ma'lumotda allaqachon aniq, lekin
p-qiymatlar va effekt hajmlari to'liq yurishdan olinishi kerak.
