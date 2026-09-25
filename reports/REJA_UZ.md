<!-- Bu fayl rejaning ishchi nusxasi. Asl nusxa Claude ning ichki papkasida
     edi va siz uni ocha olmasdingiz; endi u shu yerda va commit qilingan.
     Har qanday o'zgarish shu faylda bo'ladi. -->

# BEEI maqolasi: gibrid optimizatsiya + ML giperparametr/vazn sozlash

## QADAM 0 — bu rejani siz ocha oladigan joyga ko'chirish

**Muammo:** bu fayl `/root/.claude/plans/` da turibdi, ya'ni ish papkangizdan
tashqarida. Claude ilovasi uni **ocha olmaydi** — shuning uchun siz men
yozganlarni ko'rmagansiz. Bu mening xatom.

**Tasdiqlangandan keyin birinchi bajariladigan ish:** shu rejaning to'liq
matni `reports/REJA_UZ.md` ga ko'chiriladi, commit va push qilinadi.
Shundan keyin siz uni istalgan vaqtda ocha olasiz. Keyingi barcha
o'zgarishlar ham o'sha faylda bo'ladi.

---

## BAJARILDI — 1-bosqich: raqiblar ro'yxati va algoritm nomi

Buyruq: *"ANIQ 8 TA Q1 NATIJAGA MOS RAQIB QOLDIR QOLGAN KUCHSIZLARNI OLIB
TASHLA."* Bajarildi. Quyida **nima qilinganining aniq ro'yxati** va har biri
qanday tekshirilgani.

### 1. Yakuniy line-up — `temoa/registry.py`

| Guruh | A'zolar | Soni |
|---|---|---|
| Bizniki | `L-SHADE-DGR` (**yangi nom**), `TEMOA_V10`, `TEMOA_V11`, `TEMOA_V12` | 4 |
| Raqib — ishga tushiriladi | `LSHADE`, `jSO`, `BIPOP_CMAES`, `IPOP_CMAES`, `CMAES`, `sepCMAES` | 6 |
| Raqib — nashr etilgan jadvaldan | `EA4eig` (CEC'2022), `L-SRTDE` (CEC'2024) | 2 |
| **Jami raqib** | | **8** |

`LEGACY_SWARM` va `WEAKENED` lug'atlari **butunlay o'chirildi**. PSO, GWO, WOA,
SCA, HHO, DE/rand/1/bin va ikkita zaiflashtirilgan nazorat endi hech bir
eksperimentda qatnashmaydi. `baselines.py` fayli asl tadqiqotni qayta ishlab
chiqarish uchun **saqlanadi**, lekin hech qayerdan import qilinmaydi — buni
`test_the_original_baselines_are_kept_but_unused` `ast` bilan tekshiradi
(matn qidirish emas, haqiqiy import daraxti).

### 2. Algoritm nomi: `TEMOA_V13` emas, `L-SHADE-DGR`

Yangi fayl: `temoa/algorithms/lshade_dgr.py`. `TARGET = "L-SHADE-DGR"`.
Docstring har bir komponentning **egasini** aytadi (L-SHADE, jSO,
LSHADE-cnEpSin/EA4eig/L-SRTDE, HHO, Auger & Hansen) va har bir olib
tashlangan qismning **o'lchov sababini** yozadi.

**Nomni o'zgartirish algoritmni o'zgartirmaganini isbotladim.** Bu muhim:
agar tahrir paytida biror koeffitsient ham o'zgargan bo'lsa, V12 ga qarshi
yozilgan barcha ablatsiya raqamlari jimgina noto'g'ri bo'lib qolardi.

| Tekshiruv | Natija |
|---|---|
| V12 (shovqin mexanizmi o'chirilgan) vs `L-SHADE-DGR`, 1 avlod | **bit-identical** (F1, F4, F11, F21, F25) |
| ... 8 avloddan keyin | maksimal nisbiy farq **3.2e-13** |
| V12 dan bitta ortiqcha `wf/wf.sum()` normallashtirish olib tashlansa | **8000 FES da ham bit-identical** |

Ya'ni yagona farq — allaqachon normallashtirilgan vektorni qayta
normallashtirish (matematik jihatdan no-op, sonli jihatdan ~1 ULP), keyin
xaotik kuchayish. Test: `test_lshade_dgr_is_v12_with_the_noise_mechanism_removed`.

### 3. Yo'l-yo'lakay topilgan xato — urug'lash (seeding) sxemasi

Bu rejada yo'q edi; raqiblarni olib tashlashda **ochilib qoldi va tuzatildi**.

`run_all.py` har bir algoritmga urug'ni uning `sorted(ALL_ALGORITHMS)`
ichidagi **o'rnidan** berardi. Demak zaif baseline'larni olib tashlash
qolgan **hamma algoritmning tasodifiy oqimini o'zgartirardi**. Hech narsa
xato bermaydi — shunchaki raqamlar boshqa raqamlarga aylanadi, va `--resume`
bitta faylga **ikkita har xil urug' sxemasini** aralashtirib yuboradi.

Tuzatish: `algorithm_seed(name) = blake2b-8(name) mod (2**31−1)` — urug'
faqat **nomga** bog'liq. Endi raqib qo'shish yoki olib tashlash boshqalarning
oqimiga tegmaydi. Manifestda `seed_scheme_version: 2` yoziladi va **v1 ostida
yozilgan natijani davom ettirish rad etiladi** (xato xabari bilan).

> **Muhim oqibat:** `results_gate10` (5 yurish, eski line-up) endi hozirgi kod
> bilan qayta ishlab chiqarilmaydi. Baribir E1 protokoli 51 yurish talab
> qiladi, ya'ni darvoza qaytadan ishga tushadi — lekin buni **ochiq yozib
> qo'yamiz**, jimgina o'tkazib yubormaymiz.

### 4. Testlar: 60 → 72, hammasi o'tadi

| Fayl | Testlar | Holat |
|---|---|---|
| `tests/test_all.py` | 20 (+1: nasl aniqligi) | PASS |
| `tests/test_suites.py` | 14 | PASS |
| `tests/test_registry.py` | **11 (yangi fayl)** | PASS |
| `tests/test_competitor_validation.py` | 8 | PASS |
| `tests/test_noise.py` | 19 | PASS |
| **Jami** | **72** | **PASS** |

`test_registry.py` uchta narsani qulflab qo'yadi: (a) `TARGET` ro'yxatda
bor; (b) roppa-rosa 8 ta raqib, zaif baseline'lar yo'q; (c) urug'
qiymatlari **pinned** — ularni o'zgartirish uchun testni tahrirlash kerak,
ya'ni tasodifan o'zgarib ketmaydi.

`L-SHADE-DGR` `ALL_ALGORITHMS` ichida bo'lgani uchun mavjud tekshiruvlar
avtomatik unga ham tegadi: byudjetdan oshmaslik, byudjetning >98% ini
sarflash, bir urug'dan bit-identical natija. Qo'shimcha: 6 xil
konfiguratsiyada (`RESTART`, `EIGEN_GATE`, `DIV_GUARD` o'chirilgan, `OPS`
qisqartirilgan) D=10 va D=30 da to'liq byudjet — overrun **0**.

### Keyingi qadam

Reja bo'yicha navbatdagi ish — **E1 darvozasi** (CEC'2017, D=10, 29 funksiya,
**51 yurish**, 100 000 FES). Bu sizning kompyuteringizda ishga tushadi:

```
python START.py --jobs 20
```

Buyrug'ingizni kutaman.

---

## SIZ QAROR QILADIGAN 4 TA NARSA

Quyida to'rtta bandingiz bo'yicha **taklif va yechim** — tahlil emas, qaror.
Har biriga mening tavsiyam qo'yilgan.

### Q1 — Qaysi kategoriyada bellashamiz (jadval §0A da)

**Taklif:** asosiy da'voni **D3 (CEC hybrid, 10 funksiya) + D5 (qo'llanma)**
ga qo'yamiz. D1/D2 da "tengmiz" deymiz, **D4 (composition) da ortdamiz deb
ochiq yozamiz**.

**Yechim:** D4 zaifligini yashirmaymiz — IPOP restart qo'shamiz va
yaxshilanishni o'lchaymiz; yetmasa, cheklov sifatida qayd etamiz.

### Q2 — Raqiblar kuchlimi (tahlil §0B da)

**Taklif:** uch o'zgarish.

1. **Qo'shiladi (hozir yo'q, majburiy):** TPE (Optuna), random search,
   ReliefF, LASSO, mutual information, RF-importance, **binar-maska
   formulyatsiyasi**.
2. **Nashr etilgan jadvaldan taqqoslanadi:** EA4eig, L-SRTDE, NL-SHADE-RSP.
3. **Butunlay olib tashlanadi:** PSO, GWO, WOA, SCA, HHO, DE/rand/1 va ikkita
   zaiflashtirilgan nazorat. Yakuniy ro'yxat — **8 ta raqib**, §0B da.

**Yechim:** 1-punkt bajarilmasa, E2 hech narsa isbotlamaydi. Shuning uchun u
E2 ning **bajarilish sharti** qilib qo'yiladi.

### Q3 — Halol yangilik bo'la oladimi (tahlil §0C da)

**Taklif:** algoritmik ustunlik da'vosidan **voz kechamiz**. Maqolaning
hissasi uchta bo'ladi:

1. Birgalikda uzluksiz vazn + giperparametr formulyatsiyasi (binar-maskadan
   ustunlik **o'lchanadi**)
2. O'lchovga asoslangan gibrid — zararli komponentlar **olib tashlangan**
   (spiral, ES dumi), har biri ablatsiya bilan
3. Rejim chegarasi — qaysi byudjetdan boshlab metaevristika BO/TPE dan ustun

**Yechim:** maqolaning markazi algoritm emas, **masala formulyatsiyasi**.
Bu Control and Optimization kategoriyasiga to'liq mos va BEEI ning odatiy
darajasidan yuqori.

### Q4 — Nom va matematika

**Taklif (nom):** `L-SHADE-DGR` — L-SHADE with Diversity Guard and Restart.
Metafora emas, nasabni ochiq aytadi (jSO, LSHADE-cnEpSin, L-SRTDE kabi).
Muqobillar §0D da.

**Taklif (matematika):** §1A da to'rtta invariantlik tasdig'i qo'shildi.
Ulardan biri **mening oldingi g'oyamni bekor qiladi**: DE diagonal affin
transformatsiyaga invariant, ya'ni guruh-normallashtirish hissa bo'la
olmaydi — u standart tayyorgarlik. §1B da hybrid sinfdagi ustunlik uchun
**sinaladigan gipoteza** va uni o'lchash formulasi (`align(t)`) berildi.

---

## Context

**Jurnal.** Bulletin of Electrical Engineering and Informatics. Scopus CiteScore
bo'yicha **Control and Optimization kategoriyasida Q1** (81-persentil, 37/198).
Boshqa kategoriyalarida Q2. Scimago SJR bo'yicha esa Q3 — ikkalasi ham to'g'ri,
har xil metrika; muassasa qaysi birini hisoblashini bilish kerak.

**Bundan kelib chiqadigan qat'iy qoida:** maqola **birinchi navbatda
optimizatsiya maqolasi** bo'lishi kerak. Agar u "IDS uchun yangi usul" yoki
"neyron tarmoq uchun yangi usul" deb yozilsa, Computer Networks (Q2) yoki
Information Systems (Q2) kategoriyasiga tushadi va **Q1 yo'qoladi**. Sarlavha,
abstrakt va kalit so'zlar optimizatsiyani oldinga qo'yishi shart.

**BEEI ning qat'iy format talablari** (ularga muvofiq bo'lmasa, tashqi
taqrizsiz qaytariladi):

| Talab | Qiymat |
|---|---|
| Uzunlik | maks. **12 bet**, ~5 000 so'z |
| Shrift/format | Times New Roman 10pt, single space, rasmiy Word/LaTeX shablon |
| Abstrakt | 100–200 so'z, **o'tgan zamonda** |
| Kalit so'zlar | maks. 7 |

12 bet — tadqiqot hajmini belgilaydi: nima maqolada, nima repozitoriyda
qolishini oldindan loyihalash kerak.

### Shu paytgacha o'lchanganlar (rejaning asosi)

| Topilma | Dalil |
|---|---|
| V11 CEC'2017 **hybrid sinfida 1-o'rin** (rank 2.05, jSO 2.35) | darvoza, 29 funksiya, D=10 |
| V11 umumiy 1-o'rin (3.19), lekin jSO (3.28) dan farqi **ahamiyatsiz** | darvoza |
| V11 kompozitsiya sinfida zaif (3.95 vs BIPOP 3.70) | darvoza |
| V11 shovqinli sinflarda eng kuchli | legacy o'lchov, C6/C7 rank 1.00 |
| Spiral operator har bir funksiyada zararli; ES dumi hissasiz | ablatsiya |
| **Shovqinli kredit mexanizmi: nazariya to'g'ri, natijaga ta'siri yo'q** | ifloslanish 8–9× kamaydi, yakuniy xato o'zgarmadi |
| DE oilasi yomon shartlanganlikdan 500–2500× yo'qotadi; CMA-ES yo'qotmaydi, lekin multimodal'da zaif | premise test (ogohlantirish: birinchi versiyasida plato artefakti bor edi, toza versiyasi o'tkazilmadi) |

### Foydalanuvchi savoliga javob — dalil bilan

**Neyron tarmoq vaznlarini metaevristika bilan o'qitish — rad etiladi.**
Chuqur tarmoqda 10⁴–10⁶ o'zgaruvchi; adabiyot faqat *"low dimensional neural
networks"* da raqobatbardoshlik qayd etadi va *"convergence performance is not
as good as that of Adam"* deydi. Q1 sifatidagi natija bu yerda deyarli imkonsiz.

**Giperparametr sozlash — qabul qilinadi, lekin faqat DE yutadigan rejimda.**
Adabiyotning o'zi chegarani beradi: *"SMAC significantly outperforms DE with
small budgets, but for larger budgets, DE consistently achieves more wins than
SMAC"*; *"for ... relatively cheap objective functions, for which one can afford
more than hundreds of evaluations, CMA-ES is recommended"*.

Demak:
- chuqur tarmoq HPO (≈10² baholash) → **BO hududi, kirmaymiz**
- arzon modellar (SVM, XGBoost, RF) HPO (10³–10⁴ baholash) → **bizning hudud**

Va uni **uzluksiz feature weighting** bilan birlashtirsak, D = 40–90 bo'ladi —
darvozada o'lchangan kuchli zonamiz. Mavjud ishlar (MMAO-Cls, PSO-XGBoost)
**binar maska** ishlatadi; uzluksiz vaznlash kam band.

---

## 0A. Qaysi yo'nalish va kategoriyalarda bellashamiz

Ikkita "kategoriya" tizimi bor va ularni aralashtirmaslik kerak.

### (a) Scopus ASJC — jurnal darajasi, Q1 ni belgilaydi

| ASJC kategoriya | BEEI kvartili | Bizga aloqasi |
|---|---|---|
| **Control and Optimization** | **Q1** (81-pt, 37/198) | **maqsad** — maqola shu yerga tushishi kerak |
| Control and Systems Engineering | Q2 | — |
| Computer Networks and Communications | Q2 | agar "IDS usuli" deb yozsak, bu yerga tushadi → Q1 yo'qoladi |
| Information Systems | Q2 | — |
| Electrical and Electronic Engineering | Q2 | — |

### (b) Masala kategoriyalari — algoritm haqiqatan bellashadigan joy

| # | Yo'nalish | Kategoriya | n | Bizning o'lchangan holat |
|---|---|---|---|---|
| D1 | Uzluksiz BBO | CEC'2017 unimodal (F1, F3) | 2 | teng — hamma yechadi |
| D2 | | CEC'2017 simple multimodal (F4–F10) | 7 | teng (3.43 = jSO 3.43) |
| D3 | | **CEC'2017 hybrid (F11–F20)** | **10** | **1-o'rin (2.05 vs jSO 2.35)** ← yagona haqiqiy zonamiz |
| D4 | | CEC'2017 composition (F21–F30) | 10 | **4-o'rin** (3.95 vs BIPOP 3.70) ← zaif |
| D5 | ML qo'llanmasi | birgalikda uzluksiz vazn + giperparametr | 3 dataset × 2 klassifikator | **hali o'lchanmagan** |

**Halol xulosa:** 29 funksiyadan **faqat 10 tasida** (hybrid sinf) o'lchangan
ustunligimiz bor, va u ham 5 yurishda — ahamiyatlilik tasdiqlanmagan.
Qolgan 19 tasida tengmiz yoki ortdamiz. Maqolaning da'vosi shu chegaradan
oshmasligi kerak.

---

## 0B. Raqiblarimiz haqiqatan kuchlimi

Halol tasnif. Zaif raqib bilan bellashish foydasiz — bu to'g'ri.

### YAKUNIY RO'YXAT — 8 ta raqib (buyruq bo'yicha)

Optimizator taqqoslash (D1–D4, CEC'2017). Har birining maqomi hujjatlangan.

| # | Raqib | Maqomi | Taqqoslash usuli |
|---|---|---|---|
| 1 | **L-SHADE** | CEC'2014 **g'olibi** | ishga tushiriladi |
| 2 | **jSO** | CEC'2017 yetakchi ishtirokchisi; bizning yadromiz | ishga tushiriladi |
| 3 | **BIPOP-CMA-ES** | uzluksiz BBO ning eng kuchli umumiy etaloni | ishga tushiriladi |
| 4 | **IPOP-CMA-ES** | standart restart CMA-ES (Auger & Hansen) | ishga tushiriladi |
| 5 | **CMA-ES** (restart) | uzluksiz BBO ning etaloni (Hansen) | ishga tushiriladi |
| 6 | **sep-CMA-ES** | diagonal variant — **aylanishga moslashuvni izolyatsiya qiluvchi nazorat** | ishga tushiriladi |
| 7 | **EA4eig** | CEC'**2022 g'olibi**; ansambl + eigen — strukturaviy eng yaqin raqib | nashr etilgan jadvaldan |
| 8 | **L-SRTDE** | CEC'**2024 g'olibi**; success-rate moslashuv | nashr etilgan jadvaldan |

7 va 8 ni qayta yozmaymiz: noto'g'ri implementatsiya **bizning foydamizga**
xato qiladi — bu biz tanqid qilgan aybning aynan o'zi. Ular rasmiy musobaqa
jadvalidan, **protokol aniq mos kelganda** (29 funksiya, D=10/30, 51 yurish,
10 000·D FES) taqqoslanadi.

### OLIB TASHLANADI — zaif raqiblar

**PSO, GWO, WOA, SCA, HHO, DE/rand/1/bin, PSO_orig, HHO_orig** — butunlay
chiqariladi. 2025 sharhlari aynan shu naqshni "zaif baseline muammosi" deb
belgilaydi; ular hech qanday Q1 da'vosini ko'tara olmaydi.

Kodda: `LEGACY_SWARM` va `WEAKENED` lug'atlari `registry.py` dan olib
tashlanadi, `baselines.py` esa **tarixiy hujjat sifatida repozitoriyda
qoladi** (asl tadqiqotni qayta ishlab chiqarish uchun), lekin hech bir
eksperimentda ishlatilmaydi.

### Qo'llanma (D5) uchun alohida majburiy baseline'lar

Bular "raqib" emas — ular **boshqa sohaning amaliyoti** va ularsiz E2 hech
narsa isbotlamaydi. 8 talik ro'yxatga kirmaydi, lekin majburiy:

| Guruh | Nima |
|---|---|
| HPO amaliyoti | **TPE (Optuna)**, **random search** |
| Hukmron formulyatsiya | **binar-maska** kodlash (bir xil optimizator bilan — formulyatsiya farqini izolyatsiya qiladi) |
| Filtr usullari | ReliefF, mutual information, LASSO, RF-importance |
| Nazorat | vaznsiz (barcha xususiyatlar teng) |

### Eng muhim ogohlantirish — EA4eig

EA4eig **to'rtta algoritmning ansambli**: CMA-ES + CoBiDE + jSO varianti +
IDE, Eigen crossover bilan. Ya'ni u **ham portfel, ham eigen-crossover**
ishlatadi — strukturaviy jihatdan bizning eng yaqin raqibimiz, va u
CEC'2022 g'olibi.

> **Xulosa:** "adaptiv operator portfeli + eigen-crossover" ni yangilik deb
> da'vo qilish mumkin emas. EA4eig buni bizdan oldin qilgan va yutgan.

### Bizdagi bo'shliq

Hozirgi holatda **qo'llanma tomonida bitta ham jiddiy raqib yo'q**. TPE,
random search va filtr usullari qo'shilmasa, E2 ning natijasi hech narsa
isbotlamaydi — bu biz tanqid qilgan xatoning aynan o'zi bo'lardi.

---

## 0C. Bu g'alaba BEEI uchun halol optimizatsiya yangiligi bo'la oladimi

Savolni uchga bo'lib, har biriga alohida javob beraman.

| Mumkin bo'lgan da'vo | Halolmi? | Sabab |
|---|---|---|
| "CEC'2017 da jSO/L-SHADE/CMA-ES dan umuman ustunmiz" | **YO'Q** | 3.19 vs 3.28 — 5 yurishda shovqin. 51 yurish ham bu farqni ahamiyatli qilishi shubhali |
| "CEC g'oliblaridan (EA4eig, L-SRTDE) ustunmiz" | **YO'Q** | Sinamaganmiz, va EA4eig strukturaviy jihatdan bizdan kuchliroq ansambl |
| "CEC hybrid sinfida eng yaxshimiz" | **EHTIMOL** | 2.05 vs 2.35 — yo'nalish bor, lekin 51 yurish tasdiqlashi kerak, va EA4eig bu sinfda sinalmagan |
| "Operator portfeli + eigen-crossover yangilik" | **YO'Q** | EA4eig aynan shu |
| **"Birgalikda uzluksiz vazn + giperparametr formulyatsiyasi, binar-maska hukmron yondashuvdan ustun"** | **HA** | Formulyatsiya yangi, taqqoslash halol qilinsa isbotlanadi |
| **"O'lchovga asoslangan gibrid: zararli komponentlar olib tashlangan"** | **HA** | Bizda ablatsiya dalili bor (spiral, ES dumi), va "ko'proq operator = yaxshiroq" taxminining rad etilishi o'z-o'zidan natija |
| **"Rejim chegarasi: qaysi byudjetdan boshlab metaevristika BO dan ustun"** | **HA** | O'lchanadi, qaror qoidasi sifatida beriladi |

**Yakuniy halol javob:** algoritmik ustunlik da'vosi **BEEI uchun ham
yetarli emas**, chunki u tasdiqlanmagan va CEC g'oliblari bilan sinalmagan.
Lekin **qo'llanma tomonidagi formulyatsiya + o'lchovga asoslangan gibrid +
rejim chegarasi** uchligi halol, tekshiriladigan va BEEI ning odatiy
darajasidan **yuqori**.

Shuning uchun maqolaning markazi algoritmning o'zi emas, **masala
formulyatsiyasi va unga mos optimizator** bo'lishi kerak. Bu Control and
Optimization kategoriyasiga to'liq mos — u "optimizatsiya masalasi va uni
yechish usuli" haqida.

---

## 0D. Algoritm nomi

**Eski nom (TEMOA_V1x) tashlanadi.** Sabablari: versiya raqami maqolaga
yaramaydi, va "TEMOA" ning ochilishi hech qayerda yo'q.

**Nom qoidasi:** metafora/hayvon nomi **ishlatilmaydi** — 2025 sharhlari
aynan shunga qarshi. Bu sohada halol konventsiya — **nasabni e'lon qiluvchi
nom** (jSO, iL-SHADE, LSHADE-cnEpSin, LSHADE-SPACMA, L-SHADE-RSP, L-SRTDE).
U taqrizchiga darhol nimadan qurilganini aytadi.

Uchta variant, tavsiya bilan:

| Variant | To'liq | Nimani aytadi | Baho |
|---|---|---|---|
| **L-SHADE-DGR** | L-SHADE with Diversity Guard and Restart | nasab + o'lchangan ikkita mexanizm | **tavsiya** — eng halol, konventsiyaga to'liq mos |
| DGR-DE | Diversity-Guarded Restart Differential Evolution | mustaqilroq, lekin nasabni yashiradi | o'rtacha |
| HG-SHADE | Heterogeneous-Group SHADE | masala sinfini aytadi | qo'llanmaga bog'lab qo'yadi |

**Tavsiyam: `L-SHADE-DGR`.** U hech narsani oshirib ko'rsatmaydi va
taqrizchi "bu L-SHADE ning varianti" deb o'zi topib olishidan oldin biz
o'zimiz aytgan bo'lamiz — bu ishonch qozonadi.

---

## 1. Maqolaning o'zagi: masala sinfi

Maqola bitta masala sinfini ta'riflaydi va unga mos optimizator beradi.

**Ta'rif.** `x = (w, θ) ∈ ℝ^(d+k)`, minimallashtiriladi
```
J(x) = 1 − M_CV(w, θ)  +  λ · ‖w‖₁ / d
```
- `w ∈ [0,1]^d` — uzluksiz xususiyat vaznlari (binar maska emas)
- `θ ∈ ℝ^k` — klassifikator giperparametrlari (log-masshtabda, uzluksizlashtirilgan)
- `M_CV` — stratifikatsiyalangan k-fold CV metrikasi, **har baholashda boshqa
  tasodifiy bo'linish** → maqsad funksiya tabiiy shovqinli
- `λ‖w‖₁/d` — siyraklik jazosi

**Sinfning to'rtta xossasi, va ularning har biri o'lchangan:**

| Xossa | Nega shu sinfda | Bizda o'lchangan dalil |
|---|---|---|
| Geterogen (guruhlar har xil masshtabda) | `w ∈ [0,1]`, `θ` log-masshtabda | CEC hybrid sinfida 1-o'rin |
| Separabel emas | xususiyatlar guruh bo'lib ta'sir qiladi | eigen-crossover 62× beradi |
| Shovqinli | CV bo'linishi har safar boshqa | shovqinli sinflarda rank 1.00 |
| O'rta o'lchamli, arzon baholash | d = 40–78, baholash < 1 s | 10³–10⁴ byudjet → DE hududi |

**Bu — Control and Optimization maqolasi.** ML unda maqsad funksiya sifatida
keladi, mavzu sifatida emas.

### 1A. Invariantlik tahlili — maqolaning matematik o'zagi

Bu bo'lim "nega aynan bu algoritm bu masala sinfiga mos" degan savolga
**isbot darajasida** javob beradi, "tajribada yaxshi chiqdi" darajasida emas.

**Belgilash.** `T(x) = Ax + b`, `A` teskarilanuvchi. Transformatsiyalangan
masala `f_T(y) = f(T(y))`, quti `T⁻¹(Ω)`. Algoritm `T` ga **invariant**
deyiladi, agar u `f` va `f_T` da (mos boshlang'ich holatda) bir xil
traektoriya chizsa.

**1-tasdiq (DE diagonal affin transformatsiyaga invariant).**
`A = diag(a₁,…,a_D)`, `a_i ≠ 0` bo'lsin. DE/rand/1/bin uchun:
```
mutatsiya:  v = x_{r1} + F(x_{r2} − x_{r3})
            T(v) = T(x_{r1}) + F( T(x_{r2}) − T(x_{r3}) )     chunki A chiziqli
binomial:   koordinata bo'yicha tanlov; A diagonal ⇒ koordinatalar aralashmaydi
```
Ikkala operator ham `T` bilan kommutatsiyalanadi ⇒ **invariant**.
*Amaliy ma'nosi:* o'zgaruvchilarni normallashtirish DE uchun **hech narsa
bermaydi**. Guruh-normallashtirishni hissa deb da'vo qilib bo'lmaydi.

**2-tasdiq (DE aylanishga invariant EMAS).**
`A = R`, `RᵀR = I`, `R` diagonal emas. Binomial crossover berilgan bazisda
koordinata tanlaydi; `R` koordinatalarni aralashtiradi ⇒ `crossover∘R ≠
R∘crossover` ⇒ **invariant emas**. Bu DE ning yomon shartlangan
aylantirilgan masalalardagi zaifligining sababi.

**3-tasdiq (CMA-ES to'liq affin invariant).** Kovariatsiya `C` ni moslashtirib,
CMA-ES har qanday to'la rangli `A` ga invariant bo'ladi (Hansen). Shuning
uchun u bizning premise testimizda shartlanganlikdan **2–3×** yo'qotdi,
DE oilasi esa 500–2500×.

**4-tasdiq (eigen-crossover aylanish invariantligini taqriban tiklaydi).**
Populyatsiya kovariatsiyasi `Ĉ` ning xos vektorlari `B` bazisida crossover
qilinsa, `B → R` yaqinlashganda crossover aylanishga invariant bo'ladi.
Bu **ma'lum natija** (LSHADE-cnEpSin, EA4eig) va **bizning hissamiz emas**.

### 1B. Nega aynan hybrid sinf — gipoteza va uni sinash

CEC'2017 hybrid funksiyalari `D` o'zgaruvchini `m` guruhga bo'ladi
(tasodifiy permutatsiya bilan), har guruhga boshqa bazaviy funksiya beradi
va **har guruhni alohida aylantiradi**. Demak Gessian, permutatsiyadan
keyin, **blok-diagonal** va bloklarning shartlanganligi har xil:
```
H  ≈  Pᵀ · blockdiag(H₁, …, H_m) · P ,     cond(H_j) har xil
```
Erkinlik darajalari soni `Σ_j d_j(d_j+1)/2 ≪ D(D+1)/2`.

**Gipoteza (G1).** Blok tuzilishida populyatsiya kovariatsiyasidan olingan
xos bazis to'liq `D×D` kovariatsiyani o'rganishdan **namuna bo'yicha
samaraliroq**, chunki u dominant blokka tez moslashadi. Shuning uchun
eigen-crossover li DE bu sinfda CMA-ES dan ustun keladi.

**G1 ni sinash (yangi, o'lchanadigan):** CEC hybrid funksiyalarining guruh
bo'linishi **ta'rifdan ma'lum**. Har avlodda o'lchaymiz:
```
align(t) = ‖ Π_block · B_top_k ‖_F  /  ‖ B_top_k ‖_F     ∈ [0,1]
```
`Π_block` — haqiqiy blok qism fazosiga proyeksiya, `B_top_k` — `Ĉ` ning eng
katta `k` xos vektori. Agar `align(t)` vaqt bo'yicha 1 ga yaqinlashsa va u
CMA-ES da sekinroq bo'lsa, G1 tasdiqlanadi.

> Bu **o'lchov**, da'vo emas. Tasdiqlanmasa, hybrid sinfdagi ustunlik
> tushuntirilmagan empirik kuzatuv bo'lib qoladi va maqolada shunday
> aytiladi.

### 1C. Qo'llanma masalasining aniq tuzilishi

```
x = (w, θ) ∈ Ω = [0,1]^d × Θ,      Θ = ∏_{j=1..k} [ℓ_j, u_j]   (log-masshtabda)
J(x) = 1 − M_CV(w, θ) + λ‖w‖₁/d
```
Guruh tuzilishi **oldindan ma'lum**: `G₁ = {1..d}` (vaznlar),
`G₂ = {d+1..d+k}` (giperparametrlar). 1-tasdiqqa ko'ra bu guruhlarni
normallashtirish DE uchun bepul — shuning uchun u **standart tayyorgarlik**
deb ataladi, hissa deb emas.

`M_CV` stoxastik: har baholashda `k`-fold bo'linishi qayta tasodifiylanadi.
Shovqin **geteroskedastik** — yomon konfiguratsiyalarda dispersiya katta.
Bu qayd etiladi va `Tracker` ning tavsiya qoidasi bilan hisobga olinadi.

---

## 2. Hissa — halol chegaralangan

Uchta narsa sinaldi va **ikkitasi rad etildi**; buni maqolada ham aytamiz.

**Rad etilganlar (maqolaning "negative results" bo'limi):**
- ΔF-asosidagi kredit: zararli operatorni ko'proq mukofotlagan bo'lardi
- Shovqinga chidamli kredit: ifloslanishni 8–9× kamaytiradi, **lekin yakuniy
  natijani o'zgartirmaydi** → SHADE xotirasi cheklovchi omil emas, selection
  xatosi cheklovchi

**Da'vo qilinadigan hissa (uchta, birortasi ham nazariy teorema emas):**

1. **Masala formulasi.** Uzluksiz vazn + giperparametr birgalikda, bitta
   geterogen uzluksiz masala sifatida. Mavjud ishlar binar maska ishlatadi;
   uzluksiz variantda vazn **ahamiyat darajasini** ham beradi, nafaqat
   tanlash/tanlamaslikni.
2. **O'lchovga asoslangan gibrid.** Har bir komponent ablatsiya bilan
   oqlangan, zararlilari **olib tashlangan** (spiral, ES dumi). Bu "ko'proq
   operator = yaxshiroq" taxminining o'lchov bilan rad etilishi.
3. **Rejim chegarasi.** Qaysi byudjetdan boshlab bizning usul BO/TPE dan
   ustun bo'lishi — o'lchangan va qaror qoidasi sifatida beriladi.

**Prior-art tekshiruvi majburiy qadam**, natija `reports/PRIOR_ART.md` ga
sana bilan yoziladi. Tekshirilmagan narsa "yangi" deyilmaydi.

---

## 3. Algoritm: `L-SHADE-DGR` (§0D ga ko'ra)

**Qoladi (o'lchov bilan oqlangan, iqtibos bilan):** current-to-pbest-w/1 +
arxiv, success-history F/CR (L-SHADE, jSO); eigen-crossover `n_samples > D`
sharti bilan (**EA4eig, L-SRTDE — bizniki emas, da'vo qilinmaydi**); LPSR;
Levy operatori; diversity qo'riqchisi.

**Olib tashlanadi:** WOA spirali (har bir funksiyada zararli), (1+1)-ES dumi
(hissasiz), shovqinli kredit mexanizmi (natijaga ta'siri yo'q).

**Qo'shiladi:**
- **IPOP uslubidagi restart** (Auger & Hansen 2005 — ma'lum, iqtibos bilan).
  Sabab: darvoza kompozitsiya sinfida zaiflikni o'lchadi.
- **Guruh-xabardor boshlang'ich masshtablash.** Qo'llanmada guruhlar
  (`w` va `θ`) **oldindan ma'lum**, shuning uchun har guruh o'z qutisiga
  normallashtiriladi. Bu triviallik, lekin uni **aniq aytish** kerak, chunki
  o'lchov uning narxini ko'rsatdi (DE oilasi uchun 2–3 tartib).

**Murakkablik.** Avlodiga `O(N·D)` operator + `O(D³)` eigendekompozitsiya
(faqat `N/2 > D` bo'lganda). Xotira `O(N·D + D²)`. D = 40–90 da `D³` funksiya
baholashga nisbatan ahamiyatsiz; maqolada aniq aytiladi va D ≥ 500 da
cheklov ekani qayd etiladi.

---

## 4. Eksperimentlar

### E1 — CEC'2017 validatsiya (algoritm sog'lomligi)
D = 10 va 30, 29 funksiya, **51 yurish**, 10 000·D FES.
Raqiblar — §0B dagi **8 talik**: ishga tushiriladi L-SHADE, jSO,
BIPOP-CMA-ES, IPOP-CMA-ES, CMA-ES, sep-CMA-ES; nashr etilgan jadvaldan
EA4eig (CEC'2022 g'olibi) va L-SRTDE (CEC'2024 g'olibi).
Maqsad: umumiy ustunlik da'vosi emas, **sinf bo'yicha qayerda yutishini
ko'rsatish** (§0C ga ko'ra).

### E2 — Qo'llanma: birgalikda vazn + giperparametr
Datasetlar: **NSL-KDD (41), UNSW-NB15 (42), CIC-IDS2017 (78)** — uchalasi
ochiq; rasmiy train/test bo'linishi; **test to'plami optimizatsiyada
ko'rilmaydi**.
Klassifikatorlar: SVM (RBF) va XGBoost — arzon, ya'ni 10³–10⁴ byudjet real.
Raqiblar **uch guruh** (§0B dagi bo'shliq shu yerda yopiladi):

1. *Optimizatorlar, bir xil uzluksiz formulyatsiyada:* `L-SHADE-DGR`, jSO,
   L-SHADE, BIPOP-CMA-ES (8 talikdan ishga tushiriladiganlari), plus
   **TPE (Optuna)** va **random search**.
2. **Binar-maska formulyatsiyasi** — hukmron yondashuv, biz undan ustunlikni
   da'vo qilamiz, shuning uchun u **majburiy**: bir xil optimizator (jSO)
   binar maska + giperparametr kodlashda (MMAO uslubidagi). Bu taqqoslash
   **formulyatsiya farqini izolyatsiya qiladi**, algoritm farqini emas.
3. *Metaevristik bo'lmagan amaliyot:* ReliefF, mutual information, LASSO,
   RF importance, va **vaznsiz nazorat** (barcha xususiyatlar teng).

3-guruhsiz natija hech narsa isbotlamaydi — bu biz tanqid qilgan xatoning
aynan o'zi bo'lardi.

O'lchamlar: F1, balanced accuracy, AUC, **TPR@FPR≤1%**, tanlangan xususiyat
soni (`w_i > 0.05`), wall-clock.

### E3 — Rejim chegarasi (hissa #3)
Bir xil masalada byudjet `B ∈ {100, 300, 1000, 3000, 10000}`.
O'lchanadi: qaysi `B` da bizning usul TPE va random search dan o'tadi.
Natija — qaror qoidasi, "har doim yaxshi" degan da'vo emas.

### E4 — Ablatsiya va sezgirlik
Har bir komponent navbat bilan o'chiriladi (eigen, Levy, diversity qo'riqchisi,
restart, guruh-normallashtirish). Sezgirlik: `λ`, `POP_FACTOR`, `τ`.
Sozlash **faqat alohida validation bo'linishida**, keyin muzlatiladi.

### Hisob byudjeti
E1 (D=10) ≈ 3 soat, E1 (D=30) ≈ 12 soat, E2 ≈ datasetga bog'liq (SVM/XGBoost
10³–10⁴ baholash × 30 yurish × 3 dataset × 2 klassifikator).
**E1 D=10 → E2 → E3 → E1 D=30** tartibida; har biri mustaqil natija.

---

## 5. Statistika (`temoa/stats.py` da tayyor)

best/worst/mean/std/median/IQR; Friedman + Iman–Davenport **har o'lcham va
har dataset uchun alohida**; Holm post-hoc; Nemenyi CD; juftlangan Wilcoxon
signed-rank, Holm katak ichida **va** butun oila bo'yicha; Vargha–Delaney Â₁₂.

Quvvat hisoblangan: 29 funksiya × 7 raqobatchi = 203 test, Holm chegarasi
2.46e-04; 51 yurishda min p = 8.9e-16 → yetadi. 12 yurishda (4.9e-04)
**yetmasdi**. Nemenyi CD 29 funksiyada 1.95 (12 funksiyada 3.03 edi).

---

## 6. Maqola tuzilishi — BEEI ning 12 betiga moslangan

| Bo'lim | Bet | Mazmun |
|---|---|---|
| 1. Introduction | 1.5 | masala sinfi, bo'shliq, hissa ro'yxati |
| 2. Related work | 1.0 | SHADE oilasi, CMA-ES, HPO/feature selection |
| 3. Proposed method | 3.0 | formulyatsiya, algoritm, **psevdokod**, murakkablik |
| 4. Research method | 1.0 | datasetlar, protokol, raqiblar, statistika |
| 5. Results and discussion | 4.5 | E1 xulosa jadvali, E2 asosiy jadvallar, E3 grafigi, E4 ablatsiya, **negative results** |
| 6. Conclusion | 0.5 | scoped da'vo + cheklovlar |
| References | 0.5 | 30–40 manba |

**Maqolada:** sinf bo'yicha o'rtacha rank jadvali (29×4 to'liq emas), Friedman/
Holm natijalari, CD diagramma, E2 asosiy jadvallari, ablatsiya jadvali.
**Repozitoriyda:** to'liq 29 funksiya × o'lcham jadvallari, barcha
konvergensiya grafiklari, sezgirlik sweep'lari, xom ma'lumot, `manifest.json`.

**Sarlavha qoidasi:** algoritm va optimizatsiya oldinda, qo'llanma keyin.
Masalan shaklda: *"A <hybrid> differential evolution for joint feature
weighting and hyperparameter optimization"*. Kalit so'zlar (maks. 7)
optimizatsiya atamalaridan boshlanadi.

---

## 7. Xavflar va javoblar

| Xavf | Javob |
|---|---|
| TPE/BO kichik byudjetda yutadi | **Kutilgan va o'lchanadi** (E3). Da'vo byudjet bilan chegaralanadi, NFL hurmat qilinadi |
| "Bu shunchaki normallashtirish" | Guruh-normallashtirish trivial ekani ochiq aytiladi; hissa unda emas |
| Filter usullari (ReliefF, LASSO) yutadi | Halol qayd etiladi; "wrapper qachon filter'dan ustun" savoli o'z-o'zidan qiymatli |
| Eigen-crossover yangilik emas | Hech qachon da'vo qilinmaydi, EA4eig/L-SRTDE ga iqtibos beriladi |
| 12 betga sig'maydi | Tuzilma oldindan belgilangan; to'liq jadvallar repozitoriyda |
| Q1 kategoriyasiga tushmaslik | Sarlavha/abstrakt/kalit so'zlar optimizatsiya-birinchi |

---

## 8. O'zgartiriladigan fayllar

**Yangi:**
- `temoa/applications/feature_weighting.py` — `J(w,θ)`, CV, dataset yuklash,
  guruh-normallashtirish
- `temoa/algorithms/temoa_v13.py` — V12 dan: shovqin mexanizmi olib tashlanadi,
  restart qoladi, guruh-xabardor boshlash qo'shiladi
- `experiments/app_study.py` (E2), `experiments/budget_study.py` (E3)
- `tests/test_feature_weighting.py`

**O'zgartiriladi:**
- `temoa/registry.py` — **`LEGACY_SWARM` va `WEAKENED` butunlay olib
  tashlanadi**; yakuniy ro'yxat 8 ta raqib + `L-SHADE-DGR`; `PUBLISHED_ONLY`
  ikkitaga qisqaradi (EA4eig, L-SRTDE); TPE/random search o'ramlari qo'shiladi
- `temoa/algorithms/baselines.py` — **fayl qoladi** (asl tadqiqotni qayta
  ishlab chiqarish uchun), lekin hech bir eksperimentda ishlatilmaydi;
  buni fayl boshidagi izohda aniq yozamiz
- `tests/test_all.py`, `tests/test_competitor_validation.py` — olib
  tashlangan algoritmlarga havolalar yangilanadi
- `reports/PRIOR_ART.md` — uch hissa bo'yicha tekshiruv natijasi
- `START.py` — E2/E3 bosqichlari

**O'chiriladi:** `temoa/algorithms/temoa_v12.py` dagi N1–N3 (kod tarixda
qoladi, natija `reports` da negative result sifatida saqlanadi).

**O'zgarmaydi:** `temoa/stats.py`, `temoa/suites.py`, `temoa/tracker.py`.

---

## 9. Verification

1. `python START.py --check` — barcha testlar (hozir 60) o'tishi shart
2. **`tests/test_feature_weighting.py`:**
   - `J` determinizmi bir xil fold urug'ida; boshqa urug'da o'zgarishi (shovqin
     haqiqatan bor)
   - **test to'plami optimizatsiya davomida umuman o'qilmasligi** (kod darajasida)
   - `w = 1` nazorati vaznsiz baseline bilan ±1e-9 mos kelishi
   - guruh-normallashtirish: `θ` chegaralari to'g'ri qaytarilishi
3. E1: `python START.py --dims 10` → `reports/GATE_UZ.md`
4. E2: `python experiments/app_study.py` → qo'llanma jadvallari
5. E3: `python experiments/budget_study.py` → rejim chegarasi grafigi
6. E4: `experiments/ablation.py`, `experiments/sensitivity.py`
7. Har bosqich natijasi hisobotga yoziladi va **keyingi bosqich shundan keyin**
