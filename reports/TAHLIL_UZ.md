# TEMOA_V10_HYBRID — batafsil tahlil, diagnostika va tuzatish

**Sana:** 2026-09-25 · **Obyekt:** `hybrid 2026.py` (635 qator) · **Protokol:** `run_all.py`

Bu hujjatdagi har bir raqam o'lchangan, taxmin qilinmagan. Barcha o'lchovlar
`results_prelim/` (30D, rotated suite, 12 ta mustaqil yurish) va matnda
ko'rsatilgan maxsus diagnostik yurishlardan olingan. To'liq protokol (30/50/100D,
30 yurish) sizning mashinangizda bajariladi; uni ishga tushirish `README.md` da.

---

## 1. Qisqacha xulosa

To'rtta natija, muhimlik tartibida. **Ular 30D, rotated suite, 12 ta funksiya,
12 ta mustaqil yurishdagi to'liq o'lchovga asoslangan** (`results_prelim/`).

**1.1. Taqqoslash adolatsiz tuzilgan.** Ikkita baseline nashr etilgan shaklidan
zaiflashtirilgan, zamonaviy DE (L-SHADE, jSO) esa umuman yo'q — holbuki kod
izohida TEMOA'ning o'zagi aynan shular ekani aytilgan (§2.1). Bu holatda
"gibridlashtirish foyda berdi" degan da'voni tekshirib bo'lmaydi: siz faqat
"zamonaviy DE 1995-yilgi swarm usullaridan yaxshi" ekanini ko'rsatgan bo'lasiz.

**1.2. TEMOA_V10 da ikkita jiddiy, aniq lokalizatsiya qilingan nuqson bor:**

| 30D, median error | TEMOA_V10 | jSO | L-SHADE | V10 qancha yomon |
|---|---|---|---|---|
| **Schwefel** | 2.40e+03 | 5.43e+00 | **3.08e-02** | L-SHADE dan **78 000×** |
| **RotatedElliptic** | 2.35e+04 | **5.47e+02** | 1.84e+03 | jSO dan **43×** |

Ikkalasining ham sababi aniqlandi (§3.1, §3.3) va tuzatildi (§4).

**1.3. LEKIN — V10 umuman olganda L-SHADE/jSO dan yomon emas.** Bu mening
dastlabki xulosamning tuzatilishi. Men avval bir necha funksiya bo'yicha
o'lchab, "V10 o'z yadrosidan ko'pchilik funksiyada yutqazadi" degan edim.
**To'liq 12 funksiyali to'plamda bu noto'g'ri:**

| | TEMOA_V11 | TEMOA_V10 | jSO | L-SHADE | DE |
|---|---|---|---|---|---|
| o'rtacha rank (12 funksiya) | **2.08** | 2.46 | 3.17 | 3.50 | 3.79 |
| V10 nechta funksiyada yutadi | — | — | 7/12 | 8/12 | — |

Mening oldingi xulosam men **tanlagan qism to'plamga** asoslangan edi va u
tasodifan V10 ning zaif holatlarini o'z ichiga olgan. Bu — aynan men asl
tadqiqotni tanqid qilgan xato (cherry-picking), shuning uchun uni ochiq
tuzatyapman. To'g'ri ifoda: **V10 o'rtacha yaxshi, lekin ikkita funksiyada
halokatli darajada yomon**, va aynan shu ikki holat tuzatishga arziydi.

**1.4. Hech bir DE-oilasi algoritmi orasidagi farq statistik ahamiyatli emas.**
Friedman juda ahamiyatli (χ²=114.4, p=2.4e-19; Iman–Davenport F=71.6,
p=1.4e-47), lekin bu faqat "swarm usullari DE oilasidan orqada" degani. Holm
post-hoc'da TEMOA_V11 ning GWO/WOA/HHO/PSO/SCA dan ustunligi ahamiyatli
(p_holm 3.6e-09 … 5.1e-03), ammo **V10, jSO, L-SHADE va hatto oddiy DE bilan
farqi ahamiyatsiz** (p_holm > 0.80). Nemenyi critical difference = 4.81,
yuqoridagi 5 ta algoritm orasidagi rank farqi esa 1.7 dan kichik.

> **Bu 12 funksiya DE oilasi ichida hech qanday ustunlik da'vosini
> qo'llab-quvvatlay olmaydi** — qaysi algoritm g'olib bo'lishidan qat'i nazar.
> Bu asl tadqiqotning markaziy da'vosiga ham to'liq tegishli.

Batafsil §4.3 da, shu jumladan nima uchun 12 yurish printsipial jihatdan
yetarli emasligi va 30 yurish nega yetarli bo'lishi.

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

> **Ogohlantirish.** Bu bo'limdagi o'lchovlar **tanlangan qism to'plamlarda**
> (4–8 funksiya) — ular sababni aniqlash uchun mo'ljallangan, umumiy ustunlik
> xulosasi uchun emas. Umumiy xulosalar faqat §4.2 dagi to'liq 12 funksiyali
> to'plamdan chiqariladi. §1.3 qism to'plamdan noto'g'ri umumlashtirish qanday
> xatoga olib kelganini ko'rsatadi.

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
| f | `p` jadvali jSO ning nashr etilgan 0.25→0.125 shakliga keltirildi | V10 da 0.25→0.05, ya'ni pbest hovuzi yurish oxirida deyarli yo'qoladi. **Bu o'zgarish alohida ablatsiya qilinmagan** — shuning uchun uni hissa sifatida da'vo qilmayman, faqat qayd etaman |

**Sinab ko'rilgan va tashlab yuborilgan:** ΔF-ga asoslangan kredit (§3.3 — u
zararli operatorni ko'proq mukofotlagan bo'lardi) va `CR_FLOOR` ni o'chirish
(§3.3 — Schwefel'ga ta'sir qilmadi).

**Parametrlarni sozlash.** `POP_FACTOR`, `DIV_THRESH`, `DIV_BOOST` **10D da**,
uchta funksiyada (Ackley, Schwefel, RotatedElliptic), 3×3 gridda, 10 yurishda
tanlandi; mezon — `log10(median error)` ning o'rtachasi. G'olib:
`POP_FACTOR=12, DIV_THRESH=1e-4, DIV_BOOST=0.5`. Shundan keyin **muzlatildi**.
Hisobot qilinadigan 30/50/100D natijalariga qarab hech bir parametr
tanlanmagan.

### 4.2. V11 natijalari — to'liq 12 funksiyali to'plam

30D, rotated suite, 12 yurish, **median error** (`results_prelim/tables/`):

| Function | TEMOA_V11 | TEMOA_V10 | jSO | L-SHADE | DE |
|---|---|---|---|---|---|
| Ackley | 1.42e-14 | **7.11e-15** | 4.69e-13 | 3.80e-12 | **7.11e-15** |
| BentCigar | 3.70e+01 | **2.46e+00** | 7.99e+01 | 3.05e+01 | 1.49e+03 |
| Composition | 7.11e-15 | **0.00e+00** | 3.08e-13 | 2.82e-12 | 2.67e-15 |
| DynamicSphere | 1.14e+03 | 6.85e+02 | 2.48e+03 | 2.92e+03 | **4.81e+02** |
| Griewank | **0.00e+00** | **0.00e+00** | 1.11e-16 | 5.55e-17 | **0.00e+00** |
| Levy | **1.97e-22** | 4.48e-02 | 3.91e-21 | 3.97e-21 | 8.95e-02 |
| NoisyRastrigin | **2.03e+01** | 3.14e+01 | 2.18e+01 | 2.18e+01 | 1.83e+02 |
| NoisySphere | **1.06e-02** | 1.11e-02 | 1.07e-02 | 1.44e-02 | 1.71e-02 |
| Rosenbrock | 1.41e+01 | **1.30e+01** | 1.79e+01 | 1.83e+01 | 2.83e+01 |
| RotatedElliptic | **2.68e+01** | 2.35e+04 | 5.47e+02 | 1.84e+03 | 3.36e+06 |
| Schwefel | 1.18e+02 | 2.40e+03 | 5.43e+00 | **3.08e-02** | 1.93e+03 |
| Zakharov | 6.73e-15 | **2.28e-20** | 1.23e-10 | 2.66e-10 | 2.73e-02 |
| **o'rtacha rank** | **2.08** | 2.46 | 3.17 | 3.50 | 3.79 |

**V11 hal qilgan narsalar (katta effekt hajmi, Â₁₂ = 1.00):**
- **RotatedElliptic**: 2.35e+04 → 2.68e+01, ya'ni **877×**. V11 bu yerda jSO dan
  20×, L-SHADE dan 69× yaxshi.
- **Schwefel**: 2.40e+03 → 1.18e+02, ya'ni **20×**. Muhim: bu **to'liq tuzatish
  emas** — L-SHADE (3.08e-02) va jSO (5.43) hali ham ancha oldinda. Diversity
  qo'riqchisi qulashni yumshatadi, lekin bartaraf etmaydi.
- **NoisyRastrigin**: 3.14e+01 → 2.03e+01 (Â₁₂ = 0.91).
- **Levy**: 4.48e-02 → 1.97e-22.

**V11 yomonlashtirgan narsalar — bularni yashirmaslik kerak:**
- **DynamicSphere**: 6.85e+02 → 1.14e+03 (Â₁₂ = 0.00, ya'ni har bir yurishda
  yomonroq). Bu tizimli: diversity qo'riqchisi harakatlanuvchi optimumni
  kuzatishga xalaqit beradi. DE (4.81e+02) ikkalasidan ham yaxshi.
- **BentCigar**: 2.46e+00 → 3.70e+01 (15×).
- **Zakharov**: 2.28e-20 → 6.73e-15.
- **Composition, Ackley**: mashina aniqligida, farq ahamiyatsiz.
- **Rosenbrock**: 1.30e+01 → 1.41e+01, kutilgan (§3.1).

Ya'ni V11 universal yaxshilanish emas: u ikkita og'ir nuqsonni tuzatadi va
o'rtacha rankni 2.46 → 2.08 ga keltiradi, lekin buning evaziga
DynamicSphere va BentCigar'da haq to'laydi.

### 4.3. Statistik quvvat — nega bu raqamlar hali yetarli emas

Holm tuzatishi katak ichida (11 ta raqobatchi) qo'llanganda V11 ning
GWO/WOA/HHO/PSO/SCA ustidan ustunligi ahamiyatli. Ammo **butun oila bo'yicha
(132 ta test) hech bir katak omon qolmaydi — 0/132.**

Bu "effekt yo'q" degani emas, bu **quvvat yetishmasligi**, va sababi sof
matematik:

| yurishlar (n) | Wilcoxon signed-rank minimal p = 2/2ⁿ | Holm chegarasi (132 test) | mumkinmi? |
|---|---|---|---|
| 12 (hozirgi) | 4.88e-04 | 3.79e-04 | **yo'q** |
| 20 | 1.91e-06 | 3.79e-04 | ha |
| **30 (to'liq protokol)** | **1.86e-09** | 2.31e-04 (216 test) | **ha, keng zaxira bilan** |

Ya'ni 12 yurishda **hech qanday natija** oila darajasida ahamiyatli bo'la
olmaydi, qancha katta effekt bo'lishidan qat'i nazar. 30 yurishda esa quvvat
mo'l-ko'l. **Shuning uchun maqola raqamlari uchun to'liq 30 yurishli protokol
majburiy** — bu shunchaki "yaxshiroq" emas, balki xulosa chiqarish uchun zarur
shart.

Ikkinchi cheklov — **funksiyalar soni**. Nemenyi critical difference 12 blok va
12 algoritm uchun **4.81**, yuqoridagi 5 ta DE algoritmi orasidagi rank farqi
esa 1.71. Yurishlar sonini oshirish bunga yordam bermaydi — bu bloklar
(funksiyalar) soniga bog'liq. CD ni 2.0 ga tushirish uchun **~70 ta funksiya**
kerak. Amaliy yechim: CEC'2017 (30 funksiya) + CEC'2022 (12 funksiya) ni
qo'shish, yoki taqqoslanadigan algoritmlar sonini 5–6 taga qisqartirish
(CD k ga ham bog'liq).

---

## 5. Hisoblash xarajati

30D, bitta yurishga o'rtacha wall-clock (4 yadroli konteyner, 12 yurish,
FES = 90 000, `results_prelim/tables/runtime_seconds.csv`):

| algoritm | sek | | algoritm | sek |
|---|---|---|---|---|
| jSO | 1.89 | | HHO | 2.73 |
| L-SHADE | 1.94 | | **TEMOA_V10** | **2.98** |
| HHO_orig | 2.30 | | PSO | 3.36 |
| **TEMOA_V11** | **2.43** | | SCA | 4.79 |
| WOA | 2.58 | | GWO | 5.07 |
| PSO_orig | 2.73 | | DE | 6.54 |

- TEMOA_V10 jSO dan **1.58× qimmat** — har avlodda `O(D³)` eigendekompozitsiya.
- TEMOA_V11 V10 dan **19% arzon**, chunki eigen-gate rank-deficient holatda
  dekompozitsiyani o'tkazib yuboradi (§4.1.c) va ES dumi yo'q.
- DE va GWO sekinligi algoritm murakkabligidan emas, **Python sikllaridan**:
  ular har individ uchun alohida iteratsiya qiladi, DE/jSO oilasi esa
  vektorlashtirilgan. Ya'ni bu jadval algoritmik xarajatni emas,
  implementatsiya sifatini ham aks ettiradi — maqolada shuni aytish kerak.

**Masshtablanish chegarasi.** `O(D³)` 100D da funksiya baholashga nisbatan
arzon, lekin 500–1000D da hukmron bo'ladi. Katta o'lchamli optimizatsiya
da'vosi qilinmoqchi bo'lsa, eigen-crossover'ni cheklangan xotirali variantga
(masalan, faqat yuqori k ta asosiy komponenta) almashtirish kerak.

Xotira: `O(N·D)` populyatsiya + arxiv, `O(D²)` kovariatsiya. 100D, N=500 da
ahamiyatsiz.

## 6. No-Free-Lunch: TEMOA qaysi masala sinfiga mos?

To'liq to'plamdagi ma'lumot "universal ustunlik" ni qo'llab-quvvatlamaydi
(§1.4: DE oilasi ichidagi farqlar ahamiyatsiz). Ammo u **aniq, tor va
mexanizmi tushuntirilgan** da'voni qo'llab-quvvatlaydi:

> **Kovariatsiya-bazisli (eigen) crossover yomon shartlangan, aylantirilgan
> masalalarda hal qiluvchi ustunlik beradi.** RotatedElliptic, 30D: op0-only
> konfiguratsiyasi eigen bilan 2.19e+02, eigensiz 1.35e+04 — **62×** (§3.2).
> To'liq V11 bu funksiyada 2.68e+01, ya'ni jSO dan 20×, L-SHADE dan 69×
> yaxshi, Â₁₂ = 1.00 (har bir yurishda). Zakharov'da ham TEMOA oilasi
> L-SHADE/jSO dan 4–5 tartib oldinda.

Bu da'vo kuchli, chunki (a) mexanizmi nazariy jihatdan tushunarli — aylantirilgan
masalada koordinata bo'yicha crossover ishlamaydi, kovariatsiya bazisida
ishlaydi; (b) ablatsiya bilan izolyatsiya qilingan; (c) effekt hajmi maksimal.

**Zaif sinflar — ochiq aytilishi kerak:**
- **Deceptive multimodal (Schwefel).** V10 halokatli (2.40e+03), V11 yaxshilaydi
  (1.18e+02) lekin **hali ham L-SHADE dan 3800× orqada**. Bu ochiq muammo.
- **Dinamik masalalar.** V11 V10 dan yomon (1.14e+03 vs 6.85e+02), oddiy DE
  esa ikkalasidan yaxshi (4.81e+02). Diversity qo'riqchisi harakatlanuvchi
  optimumni kuzatishga xalaqit beradi.
- **Juda yomon shartlangan alohida yo'nalish (BentCigar).** V11 V10 dan 15×
  yomon.

Ya'ni to'g'ri xulosa NFL ga mos: **bu oila aylantirilgan/ill-conditioned
masalalar uchun, dinamik va deceptive-multimodal masalalar uchun emas.**

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

> Eksperiment qamrovi — qaysi sohalar, nechta kategoriya, qaysi raqobatchilar
> va dizayn sonlari — alohida hujjatda: **`reports/QAMROV_UZ.md`**.


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

### Port to'g'riligi — bu hisobotdagi barcha tanqid shunga tayanadi

Butun tahlil `temoa/` paketidagi ko'chirilgan V10 asl `hybrid 2026.py` dagi V10
bilan bir xil ishlashiga tayanadi. Bu tekshirildi (`tests/test_port_fidelity.py`):

| Tekshiruv | Natija |
|---|---|
| Landshaftlar (`--suite shift` vs asl `make_problem`) | **bit-ma-bit bir xil**, max nisbiy farq = 0.000e+00 |
| TEMOA_V10 natijalari, 30D, 10 yurish, Mann–Whitney | Ackley p=1.000, Schwefel p=0.427, Rosenbrock p=0.970, RotatedElliptic p=0.910 |

Ya'ni port aniqlanadigan farq bermaydi. (Bit-ma-bit bir xillik mumkin emas:
asl kod global RNG dan, port esa in'ektsiya qilingan `Generator` dan oladi.)

Ishga tushirish: `python tests/test_port_fidelity.py`

### Bu hisobotdagi raqamlarning manbai

| Bo'lim | Manba | Funksiyalar | Yurishlar |
|---|---|---|---|
| §1, §4.2, §4.3, §5 | `results_prelim/` — to'liq 30D, rotated suite | **12 (hammasi)** | 12 |
| §3.1 (operator ablatsiyasi) | maxsus diagnostik yurish, 30D | 5 (tanlangan) | 8 |
| §3.2 (eigen) | §3.1 dan | 1 | 8 |
| §3.3 (instrumentatsiya) | `logger=` bilan bitta yurish | 2 | 1 |
| §3.4 (populyatsiya) | POP_FACTOR sweep, 30D | 4–8 (tanlangan) | 5 |
| §4.1 (sozlash) | 10D grid search | 3 (tuning set) | 10 |

**Muhim ogohlantirish.** §3 va §3.4 dagi o'lchovlar **tanlangan qism
to'plamlarda** — ular diagnostika uchun, umumiy xulosa uchun emas. §1.3 aynan
shunday qism to'plamdan noto'g'ri umumlashtirish qanday xatoga olib
kelganini ko'rsatadi. Umumiy xulosalar faqat §1, §4.2, §4.3 dagi to'liq
to'plamdan chiqarilgan.

**Va eng muhimi:** 12 yurishda oila darajasidagi statistik ahamiyatlilikka
erishib bo'lmaydi (§4.3). Maqolaga kiritiladigan har qanday raqam **30
yurishli to'liq protokoldan** olinishi kerak.

