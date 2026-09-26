<!-- Bu fayl rejaning ishchi nusxasi. Asl nusxa Claude ning ichki papkasida
     bo'lgani uchun siz uni ocha olmasdingiz; bu yerda va commit qilingan.
     Har qanday o'zgarish shu faylda bo'ladi. -->

# REJA — L-SHADE-DGR ni Q1 darajaga olib chiqish

## Context — nega bu reja kerak

E1 darvozasi tugadi: CEC'2017, D=10, 29 funksiya, **51 yurish**, 10 algoritm,
100 000 FES. Natija ikki tomonlama:

```
Friedman chi2 = 65.76,  p = 1.03e-10
Iman-Davenport F = 9.43, p = 2.33e-12
Nemenyi CD (alpha=0.05) = 2.5157
```

| # | Algorithm | rank | Holm p (vs bizniki) |
|---|---|---|---|
| **1** | **L-SHADE-DGR** | **3.48** | — |
| 2 | jSO | 4.22 | 0.596 **ahamiyatsiz** |
| 3 | TEMOA_V12 | 4.31 | 0.596 ahamiyatsiz |
| 4 | TEMOA_V11 | 4.55 | 0.536 ahamiyatsiz |
| 5 | LSHADE | 5.00 | 0.225 ahamiyatsiz |
| 6 | BIPOP_CMAES | 5.36 | 0.090 ahamiyatsiz |
| 7 | IPOP_CMAES | 5.60 | **0.046 ahamiyatli** |
| 8 | CMAES | 7.16 | **2.8e-05 ahamiyatli** |
| 9 | TEMOA_V10 | 7.17 | **2.8e-05 ahamiyatli** |
| 10 | sepCMAES | 8.14 | **4.3e-08 ahamiyatli** |

**Biz 1-o'rindamiz, lekin adaptiv DE nasabidan ustunligimiz isbotlanmagan.**
jSO ga qarshi win/tie/loss = **4–21–4**. Rank farqi 0.74, Nemenyi CD esa 2.52.

Bu reja beshta talabingizni bajaradi va shu holatdan Q1 darajaga chiqish
yo'lini beradi.

---

## §1 — Raqiblar ro'yxati: TEMOA_V10/V11/V12 olib tashlanadi

### Nega olib tashlanadi

Ular **raqib emas** — bizning o'z algoritmimizning eski versiyalari. Ularni
taqqoslash jadvalida qoldirish ikki xato tug'diradi:

1. **Friedman rankini buzadi.** TEMOA_V10 (rank 7.17) kabi zaif a'zolar
   boshqa hammaning o'rtacha rankini sun'iy ravishda yaxshilaydi. Nemenyi CD
   algoritm soniga bog'liq (o'lchangan, N=29):

   | k (algoritm soni) | Nemenyi CD |
   |---|---|
   | 10 (hozirgi) | 2.5157 |
   | 9 (taklif) | 2.2309 |
   | 7 (faqat haqiqiy raqiblar) | 1.6730 |

   Ya'ni ro'yxat qisqarsa test **sezgirroq** bo'ladi.
2. **Taqrizchiga o'zimizni o'zimiz bilan taqqoslayotgandek ko'rinadi.**

> **Muhim va yoqimsiz haqiqat — yashirmayman.** Bu o'zgarish bizning
> ahvolimizni yaxshilamaydi. k=7 da ham CD = 1.673, jSO bilan rank farqimiz
> esa atigi **0.38**. Ya'ni raqiblar ro'yxatini tozalash **metodologik
> to'g'rilik**, "ustunlikni ahamiyatli qilish" usuli emas. jSO dan ustunlik
> da'vosi baribir o'tmaydi.

Ular repozitoriyda **ablatsiya mos yozuvi** sifatida qoladi (`OURS` lug'atida),
lekin `RIVALS` ga kirmaydi va asosiy jadvalda ko'rinmaydi.

### Yakuniy 9 ta raqib

| # | Raqib | Yil | Maqomi | Holat |
|---|---|---|---|---|
| 1 | CMA-ES (restart) | 2001 | uzluksiz BBO etaloni (Hansen) | ✅ bor |
| 2 | IPOP-CMA-ES | 2005 | standart restart CMA-ES (Auger & Hansen) | ✅ bor |
| 3 | sep-CMA-ES | 2008 | diagonal nazorat — aylanish izolyatsiyasi | ✅ bor |
| 4 | BIPOP-CMA-ES | 2009 | eng kuchli umumiy uzluksiz etalon | ✅ bor |
| 5 | L-SHADE | 2014 | CEC'2014 **g'olibi** | ✅ bor |
| 6 | jSO | 2017 | CEC'2017 yetakchisi; bizning yadromiz | ✅ bor |
| 7 | **LSHADE-cnEpSin** | 2017 | CEC'2017 **3-o'rin**; eigen-crossover ajdodimiz | ❌ **qo'shiladi** |
| 8 | **L-SHADE-RSP** | 2018 | CEC'2018 **g'olibi**; jSO ning vorisi | ❌ **qo'shiladi** |
| 9 | **NL-SHADE-RSP** | 2021 | CEC'2021 **g'olibi**; zamonaviy chegara | ❌ **qo'shiladi** |

### Nega aynan shu uchtasi

Hozirgi ro'yxatning eng yangi a'zosi **jSO — 2017**. To'qqiz yillik. 2026-yilgi
maqolada bu aynan biz tanqid qilgan **zaif-baseline muammosi**.

- **LSHADE-cnEpSin** — eigen-crossoverni biz undan olganmiz. Undan ustun
  kelmasak, "eigen-crossoverli portfel" degan gapning ma'nosi yo'q. Bu
  **intellektual halollik masalasi**, nafaqat kuch masalasi.
- **L-SHADE-RSP** — jSO ning bevosita vorisi (rank-based selective pressure).
  jSO bilan tengmiz; uning vorisi bilan ham tengmizmi — shuni bilish kerak.
- **NL-SHADE-RSP** — 2021, zamonaviy chegara.

### EA4eig va L-SRTDE nega YO'Q

Tekshirdim: ularning nashr etilgan jadvallari **CEC'2021/2022 va CEC'2024**
to'plamlarida. CEC'2017 D=10, 51 yurishlik jadvali topilmadi. Umumiy asossiz
taqqoslash — soxta taqqoslash. Ular **faqat iqtibos qilinadigan prior art**
bo'lib qoladi (eigen-crossover egasi sifatida), raqib sifatida emas.

### Implementatsiya xavfi va uni yopish

> Noto'g'ri implementatsiya qilingan raqib **bizning foydamizga** xato qiladi —
> bu loyiha aynan shu aybni tuzatish uchun boshlangan.

Har bir yangi raqib uchun **majburiy validatsiya**:

| Tekshiruv | Mezon |
|---|---|
| Nashr etilgan CEC'2017 jadvali bilan solishtirish | median xato, D=10 va D=30, log10 farqi < 0.5 |
| Byudjet | `overrun == 0`, `fes/max_fes > 0.98` |
| Takrorlanuvchanlik | bir urug'dan bit-identical |
| Xarakterli xulq | L-SHADE-RSP rank-based tanlov **haqiqatan** ishlayotgani (RSP o'chirilsa natija yomonlashishi) |

Validatsiyadan o'tmagan implementatsiya **jadvalga kiritilmaydi** — "taxminan
to'g'ri" deb qo'yilmaydi.

---

## §2 — Qayerda, kimdan, qaysi parametrda yutqazdik

Manba: `reports/data_gate_E1_10D/`, 51 yurish, family-wide Holm tuzatishi.

### 2A. Sinf bo'yicha o'rtacha rank (7 ta haqiqiy raqib, TEMOA'lar chiqarilgan)

| Algorithm | unimodal (2) | simple multimodal (7) | **hybrid (10)** | composition (10) | ALL 29 |
|---|---|---|---|---|---|
| **L-SHADE-DGR** | 3.75 | **3.07** | **1.80** ← 1-o'rin | **3.10** | **2.69** |
| jSO | 3.75 | 3.64 | 2.10 | 3.50 | 3.07 |
| LSHADE | 3.75 | 4.29 | 2.95 | 3.65 | 3.57 |
| BIPOP_CMAES | 3.75 | 3.50 | 4.20 | 3.40 | 3.72 |
| **IPOP_CMAES** | 3.75 | **2.93** ← **bizdan yaxshi** | 4.50 | 3.95 | 3.88 |
| CMAES | 3.75 | 5.21 | 5.65 | 4.95 | 5.17 |
| sepCMAES | 5.50 | 5.36 | 6.80 | 5.45 | 5.90 |

**O'qish:**
- **hybrid (F11–F20): 1.80 — aniq 1-o'rin.** Bu bizning haqiqiy zonamiz.
- **composition (F21–F30): 3.10 — 1-o'rin** (restart qo'shilgandan keyin
  yaxshilandi; oldingi darvozada V11 4-o'rinda edi).
- **simple multimodal (F4–F10): 3.07 — 2-o'rin.** IPOP-CMA-ES (2.93) bizdan
  yaxshi. **Yagona sinf, unda yutqazamiz.**
- unimodal: hamma 1e-8 poliga tushadi, sinf hech narsani ajratmaydi.

### 2B. Raqib bo'yicha natija (29 funksiya, Holm family-wide)

| Raqib | biz yutdik (+) | teng (=) | **biz yutqazdik (−)** |
|---|---|---|---|
| sepCMAES | 18 | 9 | 2 |
| CMAES | 15 | 11 | 3 |
| IPOP_CMAES | 13 | 12 | 4 |
| BIPOP_CMAES | 12 | 14 | 3 |
| LSHADE | 8 | 19 | 2 |
| **jSO** | **4** | **21** | **4** |

### 2C. Aniq yutqazgan funksiyalarimiz — eng og'ridan eng yengiliga

| F | Funksiya (CEC'2017) | Bizniki | Eng yaxshi | Kim | Â₁₂ | p_holm | Og'irligi |
|---|---|---|---|---|---|---|---|
| **F4** | Shifted Rotated **Rosenbrock** | 3.99 | **0.00** | CMAES | **0.186** | 3.4e-06 | **halokatli** |
| **F5** | Shifted Rotated **Rastrigin** | 1.99 | **0.00** | IPOP_CMAES | **0.181** | 7.2e-05 | **halokatli** |
| F24 | Composition 4 | 200.0 | 100.0 | BIPOP/sepCMAES | 0.154 | 9.8e-07 | katta (2×) |
| F21 | Composition 1 | 100.0 | 100.0 | jSO | 0.206 | 1.4e-05 | dispersiya |
| F30 | Composition 10 | 1377.49 | 1377.24 | jSO | 0.121 | 3.5e-04 | dispersiya |
| F25 | Composition 5 | 479.19 | 475.69 | BIPOP_CMAES | 0.169 | 2.5e-06 | kichik (1.01×) |
| F7 | Lunacek Bi-Rastrigin | 11.53 | 11.37 | IPOP_CMAES | 0.337 | 6.3e-02 | kichik |
| F18 | Hybrid 8 | 0.468 | 0.243 | LSHADE | 0.376 | 0.88 | ahamiyatsiz |
| F27 | Composition 7 | 390.52 | 390.52 | LSHADE | 0.334 | 4.8e-02 | dispersiya |

**Â₁₂ o'qilishi:** 0.5 = teng. 0.186 degani — tasodifiy tanlangan bizning
yurishimiz tasodifiy tanlangan CMA-ES yurishidan faqat **18.6%** hollarda
yaxshi. Bu **large** effekt, bizga qarshi.

**Naqsh aniq:** F4 va F5 da CMA-ES **aniq nolga** tushadi, biz esa umuman
tushmaymiz. Qolgan yutqazishlar 1–2× darajasida — ular muhim emas.
**Butun muammo F4 va F5 da.**

---

## §3 — Qaysi qism ishlamayapti: diagnoz va yechim

### 3A. Matematik diagnoz — to'rtta invariantlik tasdig'i

**Belgilash.** `T(x) = Ax + b`, `A` teskarilanuvchi. Transformatsiyalangan
masala `f_T(y) = f(T(y))`. Algoritm `T` ga **invariant** deyiladi, agar u `f`
va `f_T` da (mos boshlang'ich holatda) bir xil traektoriya chizsa.

**1-tasdiq — DE diagonal affin transformatsiyaga invariant.**
`A = diag(a₁,…,a_D)`, `a_i ≠ 0`. DE/rand/1/bin uchun:
```
mutatsiya:  v = x_{r1} + F(x_{r2} − x_{r3})
            T(v) = T(x_{r1}) + F(T(x_{r2}) − T(x_{r3}))    (A chiziqli)
binomial:   koordinata bo'yicha tanlov; A diagonal ⇒ koordinatalar aralashmaydi
```
Ikkala operator `T` bilan kommutatsiyalanadi ⇒ **invariant**.
*Oqibati:* o'zgaruvchilarni normallashtirish DE uchun **hech narsa bermaydi**.
Guruh-normallashtirishni hissa deb da'vo qilib bo'lmaydi — u standart
tayyorgarlik.

**2-tasdiq — DE aylanishga invariant EMAS.**
`A = R`, `RᵀR = I`, `R` diagonal emas:
```
binomial crossover:  u_i = v_i  agar  rand_i < CR,  aks holda  x_i
```
crossover **berilgan bazisda koordinata tanlaydi**; `R` koordinatalarni
aralashtiradi ⇒ `crossover ∘ R ≠ R ∘ crossover` ⇒ **invariant emas**.

**3-tasdiq — CMA-ES to'liq affin invariant.**
Kovariatsiya `C` ni moslashtirib, CMA-ES har qanday to'la rangli `A` ga
invariant bo'ladi (Hansen).

**4-tasdiq — eigen-crossover aylanish invariantligini taqriban tiklaydi.**
Populyatsiya kovariatsiyasi `Ĉ` ning xos vektorlari `B` bazisida crossover
qilinsa, `B → R` yaqinlashganda crossover aylanishga invariant bo'ladi.
**Bu ma'lum natija** (LSHADE-cnEpSin, EA4eig) va **bizning hissamiz emas**.

### 3A-bis. Nazariya F4/F5 ni tushuntiradi

F4 = Shifted **Rotated** Rosenbrock, F5 = Shifted **Rotated** Rastrigin —
ikkalasi ham `f(R(x − o))` shaklida. Rosenbrock vodiysi aylantirilgan, egri
va yomon shartlangan; uni kuzatish uchun to'la kovariatsiya kerak.
2-tasdiqqa ko'ra DE uni kuzata olmaydi, 3-tasdiqqa ko'ra CMA-ES kuzatadi.

**Demak F4/F5 dagi yutqazish tasodif emas — u nazariyadan kelib chiqadi**,
va o'lchov buni tasdiqlaydi: CMA-ES aynan nolga tushadi, biz tushmaymiz.

### 3B. Bizda yechim bor edi, lekin ishlamayapti

4-tasdiq: **eigen-crossover** aylanish invariantligini taqriban tiklaydi —
populyatsiya kovariatsiyasi `Ĉ` ning xos bazisi `B` da crossover qilinsa,
`B → R` yaqinlashganda invariantlik tiklanadi.

Bizda bu **bor**: `temoa/algorithms/lshade_dgr.py`, `P_EIG` ehtimoli bilan.
Lekin F4/F5 da ishlamayapti. **Nega — shuni o'lchov aniqlaydi, taxmin emas.**

### 3C. O'tkazilayotgan diagnostika (hozir ishlayapti)

11 ta konfiguratsiya × 6 funksiya (F4, F5, F7, F16, F24, F11) × 15 yurish,
D=10, 100 000 FES:

| Konfiguratsiya | Nimani sinaydi |
|---|---|
| `DGR default` | mos yozuv |
| `P_EIG=0.9 fixed`, `ADAPT_EIG=False` | **adaptatsiya eigen'ni o'chirib qo'yayaptimi?** |
| `P_EIG=0.0` | eigen umuman hissa beryaptimi? |
| `EIGEN_GATE=False` | `n_samples > D` sharti zarar qilyaptimi? |
| `RESTART=False` | restart F4/F5 da yordam beryaptimi yoki zarar? |
| `DIV_GUARD=False` | qo'riqchi Rosenbrock vodiysidan chiqarib yuboryaptimi? |
| `OPS=(0,)` | portfelning o'zi zararmi? |
| `POP_FACTOR=18` | populyatsiya kichikligi sababmi? |
| jSO, BIPOP, IPOP | raqib mos yozuvlari |

> **Bu bo'lim diagnostika natijasi kelgach to'ldiriladi.** Sabab
> o'lchanmaguncha yechim taklif qilinmaydi — aks holda bu taxmin bo'lardi.

**Hisob haqiqati:** 11 × 6 × 15 = 990 yurish × 100 000 FES ≈ **9.6 CPU-soat**.
Cloud konteynerda 4 yadro bor va ularning bir qismini eski darvoza yurishi
band qilgan — shuning uchun bu yerda ~5 soat ketadi. **Sizning mashinangizda
(24 tred) ~25 daqiqa.** Diagnostika 2-bosqich sifatida sizda o'tkaziladi;
konteynerdagisi fon rejimida ishlab turadi va ulgursa, natijasini qo'shamiz.

### 3D. Taklif qilinadigan yechim — hozirgi asosiy nomzod

O'lchov tasdiqlasa, yechim **komponent qo'shish emas, mavjudini to'g'ri
ishlatish** bo'ladi. Uchta nomzod, kuch tartibida:

| # | Yechim | Nima o'zgaradi | Nega ishlashi kerak | Narxi |
|---|---|---|---|---|
| **A** | **Eigen-crossover chastotasini masalaga moslash** — `P_EIG` ni muvaffaqiyat emas, **shartlanganlik soni** `cond(Ĉ)` bo'yicha boshqarish | `P_EIG = clip(1 − 1/log10(cond(Ĉ)), 0.1, 0.9)` | Yomon shartlangan aylantirilgan masalada eigen kerak; yaxshi shartlanganda keraksiz. Hozirgi adaptatsiya **darhol muvaffaqiyat** bo'yicha ishlaydi — bu Rosenbrock vodiysida noto'g'ri signal, chunki vodiy bo'ylab yurish sekin | `O(D³)` allaqachon bor, qo'shimcha narx yo'q |
| **B** | **CMA-ES ni portfelga integratsiya qilish** (LSHADE-SPACMA uslubi) | byudjetning bir qismi CMA-ES ga, adaptiv taqsimlash bilan | F4/F5 da CMA-ES **aniq nolga** tushadi. Bu ochiq zaiflikni to'g'ridan-to'g'ri yopadi | **prior art**: LSHADE-SPACMA aynan shu. Yangilik emas, lekin halol iqtibos bilan ishlatish mumkin |
| **C** | **Muvaffaqiyat o'rniga kechiktirilgan kredit** (AOS topilmasi) | operator krediti darhol ΔF emas, `k` avloddan keyingi hissa bo'yicha | §3.3 o'lchovi: zararli operator fitness foydasining 66% ini oladi. Darhol kredit populyatsiya qulashini ko'ra olmaydi | yangi, sinalmagan — **bu bizning haqiqiy hissamiz bo'lishi mumkin** |

**Tavsiyam: A + C.** B ni tanlamaymiz, chunki u LSHADE-SPACMA ning takrori —
yangilik bermaydi va "CMA-ES qo'shdik" degan gap Q1 hissasi emas.

**Olib tashlanadiganlar (allaqachon o'lchangan):**

| Olib tashlandi | Nega | O'rniga |
|---|---|---|
| WOA spiral operatori | har bir funksiyada zararli; fitness foydasining 66% ini olgan, lekin populyatsiyani qulatgan | diversity guard |
| (1+1)-ES dumi | hissa nol; ikki funksiyada natija bit-identical | byudjet asosiy siklga qaytarildi |
| Shovqinga chidamli kredit (N0–N3) | ifloslanish 8–9× kamaydi, **yakuniy xato o'zgarmadi** | negative result sifatida qoladi |

---

## §4 — Yagona `.py` fayl: `START.py` qayta yoziladi

Bitta buyruq — hammasi. Hozirgi `START.py` faqat CEC darvozasini biladi;
yangi versiya to'rtta bosqichni boshqaradi.

```
python START.py                 # to'liq quvur: check → test → E1 → diagnostika → tahlil
python START.py --check         # faqat tekshiruv va testlar (~6 daq)
python START.py --stage e1      # faqat CEC'2017 darvozasi
python START.py --stage diag    # faqat komponent diagnostikasi
python START.py --stage aos     # faqat AOS kredit tadqiqoti
python START.py --dims 10 30    # o'lchamlar
python START.py --jobs 20       # ishchilar soni
```

**Kafolatlar (har biri testga bog'langan):**

| Kafolat | Qanday ta'minlanadi |
|---|---|
| Testlar yiqilsa **eksperiment boshlanmaydi** | `run_tests()` → `return 1` |
| Uzilsa **yo'qolmaydi** | `--resume`, har funksiyadan keyin CSV yoziladi |
| Qayta ishga tushirish **bit-identical** | urug' nomga bog'liq (`algorithm_seed`), test bilan qulflangan |
| Byudjetdan **oshmaydi** | `Tracker.overrun == 0`, har algoritmda sinaladi |
| To'liqsiz natija **tahlil qilinmaydi** | `merge_shards.py` funksiya va qator sonini tekshiradi |
| Har bosqich **mustaqil** | biri yiqilsa qolganlari saqlanadi |

**Yangi:** bosqichlar oralig'ida progress va ETA, disk joyini tekshirish,
Windows/Linux farqini avtomatik hal qilish, tugagach `reports/` ga
hisobot yozish.

---

## §5 — Matematik aniqlik va halollik qoidalari

Bu bo'lim "yaxshi niyat" emas — har biri **kodga yoki testga bog'langan**.

| Qoida | Ta'minlovchi mexanizm |
|---|---|
| Raqam o'lchanmasa, yozilmaydi | Har bir da'vo `reports/data_*/` dagi faylga havola qiladi |
| p-qiymat quvvatsiz testdan chiqmaydi | 51 yurish; 2 funksiyali sinfda test **chop etilmaydi** (`class_ranks.py` shunday yozilgan) |
| Effekt o'lchami majburiy | Â₁₂ har juftlikda; faqat p-qiymat yetarli emas |
| Ko'p taqqoslash tuzatiladi | Holm — katak ichida **va** butun oila bo'yicha |
| Raqib zaiflashtirilmaydi | `test_competitor_validation.py`: har biri byudjetning >98% ini sarflaydi, xarakterli xulqi tasdiqlanadi |
| Implementatsiya nashr bilan solishtiriladi | Yangi raqiblar CEC'2017 jadvaliga qarab validatsiya qilinadi |
| Yangilik tekshirilmasdan da'vo qilinmaydi | `reports/PRIOR_ART.md`, sana bilan |
| Takrorlanuvchanlik o'lchanadi | `compare_runs.py` — ikki mashinada bit-identical talab qilinadi |
| Salbiy natija yashirilmaydi | Shovqin mexanizmi maqolada "negative results" bo'limida |
| Wall-clock platformaga bog'liq | GitHub runner'lari bir xil ish uchun **2.7× farq qildi** (21.9–59.1 daq). Vaqt **faqat bitta nazorat qilinadigan mashinada** o'lchanadi |

---

## §6 — Mening qo'shimcha taklifim

**Maqolaning markazi algoritm emas, AOS topilmasi bo'lishi kerak.**

Sabab raqamlarda: 51 yurishda ham jSO dan ustunligimiz **p = 0.596**.
Bu da'vo hech qachon o'tmaydi. Lekin bizda o'lchangan boshqa narsa bor:

| | op0 pbest-DE | op1 leader | **op2 spiral** | op3 Levy |
|---|---|---|---|---|
| muvaffaqiyat ulushi (AOS shuni mukofotlaydi) | 0.103 | 0.127 | 0.120 | 0.060 |
| **umumiy fitness foydasi** | 11.1% | 14.2% | **66.1%** | 8.6% |
| ajratilgan ehtimollik | 0.175 | 0.086 | **0.644** | 0.095 |

Eng **zararli** operator (ablatsiya tasdiqlagan) krediting **66%** ini oladi.
Chunki u qarz olib ishlaydi: populyatsiyani yig'ib tez pasayadi, to'lov esa
kechikkan. **Darhol yaxshilanishga asoslangan hech qanday kredit sxemasi buni
ko'ra olmaydi** — probability matching, adaptive pursuit, DMAB, extreme-value,
DE-DDQN, hammasi.

Agar bu umumiy bo'lsa — bu **usullar sinfi** haqidagi topilma, bizning
algoritm haqida emas. Information Sciences / SWEVO darajasi.

**Lekin dalil hozir: bitta funksiya, bitta yurish, bitta algoritm.**
Shuning uchun taklifim — uni jiddiy o'lchash:

- 4–5 AOS sxemasi (probability matching, adaptive pursuit, DMAB, extreme-value)
- × 29 funksiya × 51 yurish
- har operatorning **darhol krediti** va **yakuniy hissasi** alohida yoziladi
- `corr(darhol kredit, yakuniy hissa)` — agar u manfiy yoki nolga yaqin
  bo'lsa, patologiya tasdiqlanadi

Bu tasdiqlanmasa — halol aytamiz va BEEI uchun yaxshi amaliy maqola qoladi.

---

## §7 — Jurnal: BEEI va Q1 kategoriyasiga moslik

### 7A. Qaysi kategoriya Q1 beradi va uni qanday yo'qotamiz

Bulletin of Electrical Engineering and Informatics, Scopus CiteScore bo'yicha:

| ASJC kategoriya | BEEI kvartili | Bizga aloqasi |
|---|---|---|
| **Control and Optimization** | **Q1** (81-persentil, 37/198) | **maqsad** — maqola shu yerga tushishi kerak |
| Control and Systems Engineering | Q2 | — |
| Computer Networks and Communications | Q2 | "IDS usuli" deb yozsak → **Q1 yo'qoladi** |
| Information Systems | Q2 | "ML tizimi" deb yozsak → **Q1 yo'qoladi** |
| Electrical and Electronic Engineering | Q2 | — |

> **Scimago SJR bo'yicha BEEI Q3.** Ikkalasi ham to'g'ri — har xil metrika.
> Muassasangiz qaysi birini hisoblashini bilish kerak. Biz CiteScore
> (Control and Optimization, Q1) bo'yicha maqsad qo'yamiz.

### 7B. Qat'iy loyihalash qoidasi

**Maqola birinchi navbatda OPTIMIZATSIYA maqolasi bo'lishi shart.** ML unda
maqsad funksiya sifatida keladi, mavzu sifatida emas.

| Element | Qoida | Misol shakl |
|---|---|---|
| Sarlavha | optimizatsiya va algoritm **oldinda**, qo'llanma keyin | *"A rotation-aware differential evolution for joint feature weighting and hyperparameter optimization"* |
| Abstrakt | birinchi jumla — **optimizatsiya masalasi**, IDS/ML emas | "This study addressed a heterogeneous continuous optimization problem..." |
| Kalit so'zlar (maks. 7) | birinchi 3 tasi optimizatsiya atamasi | differential evolution; black-box optimization; adaptive operator selection; rotation invariance; feature weighting; hyperparameter optimization; CEC benchmark |
| Introduction | bo'shliq **optimizatsiya bo'shlig'i** sifatida qo'yiladi | "credit assignment in AOS is blind to..." |

Agar maqola "IDS uchun yangi usul" yoki "neyron tarmoq uchun usul" deb
yozilsa, u Computer Networks (Q2) yoki Information Systems (Q2) ga tushadi.
**Q1 yo'qoladi.**

### 7C. BEEI format talablari — buzilsa tashqi taqrizsiz qaytariladi

| Talab | Qiymat | Bizga ta'siri |
|---|---|---|
| Uzunlik | maks. **12 bet**, ~5 000 so'z | tadqiqot hajmini belgilaydi |
| Shrift/format | Times New Roman 10pt, single space, rasmiy shablon | LaTeX/Word shabloni majburiy |
| Abstrakt | 100–200 so'z, **o'tgan zamonda** | "we propose" emas, "this study proposed" |
| Kalit so'zlar | maks. **7** | 7C jadvalidagidek |
| Referenslar | 30–40 | prior art bo'limi shunga sig'ishi kerak |

### 7D. 12 betga moslashtirilgan tuzilish

| Bo'lim | Bet | Mazmun |
|---|---|---|
| 1. Introduction | 1.5 | masala sinfi, **optimizatsiya bo'shlig'i**, hissa ro'yxati |
| 2. Related work | 1.0 | SHADE oilasi, CMA-ES, AOS kredit sxemalari, HPO |
| 3. Proposed method | 3.0 | formulyatsiya, invariantlik tasdiqlari, **psevdokod**, murakkablik |
| 4. Research method | 1.0 | datasetlar, protokol, **9 ta raqib**, statistika |
| 5. Results and discussion | 4.5 | E1 sinf jadvali, AOS tadqiqoti, E2/E3, ablatsiya, **negative results** |
| 6. Conclusion | 0.5 | chegaralangan da'vo + cheklovlar |
| References | 0.5 | 30–40 manba |

**Maqolada:** sinf bo'yicha rank jadvali (29×9 to'liq emas), Friedman/Holm,
CD diagramma, AOS korrelyatsiya grafigi, ablatsiya jadvali.
**Repozitoriyda:** to'liq 29 funksiya × o'lcham jadvallari, konvergensiya
grafiklari, sezgirlik sweep'lari, xom ma'lumot, `manifest.json`.

---

## §8 — Qolgan eksperimentlar (avvalgi rejadan saqlanadi)

### E2 — Qo'llanma: birgalikda vazn + giperparametr

Bu maqolani **Control and Optimization** kategoriyasida ushlab turadigan
qism: aniq optimizatsiya masalasi, ML esa maqsad funksiya.

```
x = (w, θ) ∈ Ω = [0,1]^d × Θ,      Θ = ∏_j [ℓ_j, u_j]  (log-masshtabda)
J(x) = 1 − M_CV(w, θ) + λ‖w‖₁/d
```

| Xossa | Nega shu sinfda | Bizda o'lchangan dalil |
|---|---|---|
| Geterogen (guruhlar har xil masshtabda) | `w ∈ [0,1]`, `θ` log-masshtabda | **hybrid sinfida rank 1.80, 1-o'rin** |
| Separabel emas | xususiyatlar guruh bo'lib ta'sir qiladi | eigen-crossover hissasi |
| Shovqinli | CV bo'linishi har baholashda qayta tasodifiylanadi | shovqin tadqiqoti (negative result) |
| O'rta o'lchamli, arzon baholash | d = 40–78, baholash < 1 s | 10³–10⁴ byudjet → DE hududi |

**Datasetlar:** NSL-KDD (41), UNSW-NB15 (42), CIC-IDS2017 (78) — uchalasi
ochiq, rasmiy train/test bo'linishi, **test to'plami optimizatsiyada
ko'rilmaydi** (kod darajasida tekshiriladi).
**Klassifikatorlar:** SVM (RBF), XGBoost — arzon, 10³–10⁴ byudjet real.

**Majburiy baseline'lar** (bularsiz E2 hech narsa isbotlamaydi):

| Guruh | Nima |
|---|---|
| Optimizatorlar, bir xil formulyatsiyada | L-SHADE-DGR, jSO, L-SHADE, BIPOP-CMA-ES |
| HPO amaliyoti | **TPE (Optuna)**, random search |
| Hukmron formulyatsiya | **binar-maska** kodlash, **bir xil optimizator bilan** — formulyatsiya farqini izolyatsiya qiladi |
| Filtr usullari | ReliefF, mutual information, LASSO, RF-importance |
| Nazorat | vaznsiz (barcha xususiyatlar teng) |

**O'lchamlar:** F1, balanced accuracy, AUC, **TPR@FPR≤1%**, tanlangan
xususiyat soni (`w_i > 0.05`), wall-clock.

### E3 — Byudjet rejimi chegarasi

Bir xil masalada `B ∈ {100, 300, 1000, 3000, 10000}`. O'lchanadi: qaysi `B`
da bizning usul TPE va random search dan o'tadi. Natija — **qaror qoidasi**,
"har doim yaxshi" da'vosi emas. NFL teoremasi hurmat qilinadi.

### E4 — Ablatsiya va sezgirlik

Har bir komponent navbat bilan o'chiriladi (eigen, Levy, diversity guard,
restart, guruh-normallashtirish, §3D yechimi). Sezgirlik: `λ`, `POP_FACTOR`,
`P_EIG` boshqaruvi. Sozlash **faqat alohida validation bo'linishida**,
keyin muzlatiladi.

> **Eslatma:** `experiments/ablation.py` va `sensitivity.py` hozir **V11 ni**
> ablatsiya qiladi, `L-SHADE-DGR` ni emas. E4 dan oldin yangilanishi shart.

---

## §9 — Statistika protokoli

| Element | Qoida |
|---|---|
| Tavsifiy | best / worst / mean / std / median / IQR |
| Omnibus | Friedman + Iman–Davenport, **har o'lcham va har dataset uchun alohida** |
| Post-hoc | Holm step-down, control = L-SHADE-DGR |
| Vizual | Nemenyi CD diagramma (k=9, N=29 → CD = 2.2309) |
| Juftlik | paired Wilcoxon signed-rank, Holm **katak ichida va butun oila bo'yicha** |
| Effekt o'lchami | Vargha–Delaney Â₁₂ + magnitude |

**Quvvat hisobi (o'lchangan):** 29 funksiya × 8 raqobatchi = 232 test,
Holm chegarasi ≈ 2.2e-04. 51 yurishda min p = 8.9e-16 → **yetadi**.
12 yurishda (4.9e-04) **yetmasdi** — shuning uchun 51 yurish majburiy.

---

## §10 — Xavflar va javoblar

| Xavf | Javob |
|---|---|
| jSO dan ustunlik ahamiyatli chiqmaydi | **Kutilgan.** Da'vo qilinmaydi; markaz AOS topilmasiga ko'chiriladi (§6) |
| §3D yechimi F4/F5 ni tuzatmaydi | Halol qayd etiladi; "DE oilasining aylanishga invariant emasligi" cheklov sifatida beriladi |
| TPE/BO kichik byudjetda yutadi | **Kutilgan va o'lchanadi** (E3). Da'vo byudjet bilan chegaralanadi |
| Filter usullari (ReliefF, LASSO) yutadi | Halol qayd etiladi; "wrapper qachon filter'dan ustun" savoli o'z-o'zidan qiymatli |
| Eigen-crossover yangilik emas | **Hech qachon da'vo qilinmaydi**; LSHADE-cnEpSin, EA4eig, L-SRTDE ga iqtibos |
| Yangi raqib noto'g'ri implementatsiya qilinadi | Nashr etilgan CEC'2017 jadvaliga validatsiya; o'tmasa jadvalga kiritilmaydi |
| 12 betga sig'maydi | Tuzilma oldindan belgilangan (§7D); to'liq jadvallar repozitoriyda |
| **Q1 kategoriyasiga tushmaslik** | **Sarlavha/abstrakt/kalit so'zlar optimizatsiya-birinchi** (§7B) |
| AOS topilmasi umumiy emas, Schwefel'ga xos | Shuning uchun 29 funksiya × 51 yurish × 4–5 sxema o'lchanadi |

---

## Bajarilish tartibi

| # | Ish | Fayl | Natija | Qayerda |
|---|---|---|---|---|
| 0 | Rejani `reports/REJA_UZ.md` ga ko'chirish, commit | `reports/` | siz ocha olasiz | cloud |
| 1 | Konteyner yurishini to'xtatish | — | resurs bo'shaydi | cloud |
| 2 | `RIVALS` dan TEMOA'larni chiqarish | `temoa/registry.py` | 9 ta raqib | cloud |
| 3 | **Diagnostika** (11 konfig × 6 funksiya × 15 yurish) | `experiments/diagnose.py` | F4/F5 sababi | **sizda, ~25 daq** |
| 4 | Sababni §3 ga yozish | reja + `reports/` | yechim tanlanadi | cloud |
| 5 | 3 ta yangi raqib + validatsiya | `temoa/algorithms/modern.py`, `tests/` | line-up to'liq | cloud |
| 6 | §3D yechimi (A + C) | `temoa/algorithms/lshade_dgr.py` | F4/F5 tuzatiladi | cloud |
| 7 | `START.py` qayta yozish | `START.py` | yagona kirish | cloud |
| 8 | **E1 qayta o'lchash** (9 raqib, 51 yurish, D=10) | — | yakuniy CEC jadvali | **sizda, ~2 soat** |
| 9 | **AOS tadqiqoti** (4–5 sxema × 29 × 51) | `experiments/aos_study.py` | **asosiy hissa** | **sizda** |
| 10 | E2 qo'llanma (3 dataset × 2 klassifikator) | `temoa/applications/`, `experiments/app_study.py` | Q1 kategoriya himoyasi | **sizda** |
| 11 | E3 byudjet rejimi | `experiments/budget_study.py` | qaror qoidasi | **sizda** |
| 12 | E4 ablatsiya va sezgirlik (yangilangan) | `experiments/ablation.py` | komponent oqlanishi | **sizda** |
| 13 | D=30 darvozasi | — | o'lcham bo'yicha kengayish | **sizda, ~6 soat** |
| 14 | Maqola matni (§7D tuzilishi) | `reports/PAPER.md` | BEEI ga topshirish | cloud |

Har bosqich mustaqil natija beradi; keyingisiga o'tishdan oldin natija
`reports/` ga yoziladi va siz ko'rasiz.

## Verification

| # | Tekshiruv | Mezon |
|---|---|---|
| 1 | `python START.py --check` | barcha testlar o'tishi shart (hozir 72 ta) |
| 2 | Yangi raqiblar | nashr etilgan CEC'2017 jadvaliga log10 farqi < 0.5 |
| 3 | §3D yechimi | F4/F5 da median xato **kamayishi** VA hybrid sinfda rank **yomonlashmasligi** (regressiya nazorati) |
| 4 | `python tools/compare_runs.py` | ikki mashinada bit-identical |
| 5 | `python tools/class_ranks.py` | sinf bo'yicha ranklar, 2 funksiyali sinfda p-qiymat chop etilmasligi |
| 6 | E2 `tests/test_feature_weighting.py` | `J` determinizmi; **test to'plami optimizatsiyada o'qilmasligi** (kod darajasida); `w=1` nazorati vaznsiz baseline bilan ±1e-9 mos kelishi |
| 7 | Byudjet | har algoritmda `overrun == 0`, `fes/max_fes > 0.98` |
| 8 | Prior art | har bir "yangi" da'vo `reports/PRIOR_ART.md` da sana bilan tekshirilgan |
| 9 | **Jurnal moslik** | sarlavha/abstrakt/kalit so'zlar §7B jadvaliga mos; abstrakt 100–200 so'z, **o'tgan zamon**; maqola ≤ 12 bet |
