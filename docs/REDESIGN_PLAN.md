# ACE-SHADE: noldan qayta loyihalash rejasi

> **Doimiy mandat.** "Buni qilib bo'lmaydi" — bu natija emas. Har bir
> o'lchangan zaiflik uchun aniq, matematik asoslangan yechim taklif
> qilinadi va sinaladi. Agar yechim ishlamasa, sababi o'lchanadi va
> keyingi yechim quriladi. To'xtash sharti faqat bitta: o'lchov
> natijasi, taxmin emas.
>
> Ayni paytda o'lchov halol qoladi: algoritmni maksimal darajada kuchli
> qilib loyihalash va natijani bo'yamaslik bir-biriga zid emas.

Nom taklifi: **ACE-SHADE** (*Adaptive Covariance-Ensemble SHADE*).
O'zgartirish mumkin.

---

## 1. Nega qayta loyihalash kerak

Joriy CBA-SHADE o'lchovlari (`docs/RESULTS.md`) uch aniq muammoni
ko'rsatdi. Ularning har biri **tasodifiy emas, konstruktiv sabab**ga ega:

| O'lchangan muammo | Raqam | Konstruktiv sabab |
|---|---|---|
| `RotatedElliptic` 30D | 949x yomon | Kovariatsiya har avlodda populyatsiya namunasidan baholanadi — bu o'lchamda shovqin |
| `NoisySphere` (3 o'lcham) | 1.5–2.2x yomon | CMA quyrug'i shovqindan model quradi; shovqin detektori yo'q |
| `Levy`, `Zakharov` 100D | aniqlikka yetmaydi | Yuqori o'lchamda yaqinlashish tezligi yetishmaydi |

Muhim: **kuchli tomonlar ham konstruktiv sababga ega** va ular
saqlanishi shart:

| O'lchangan kuch | Raqam | Sabab |
|---|---|---|
| `Schwefel` 50D | **75 060x** yaxshi | Ikki parametr rejimi ansambli (jSO 1.18e+02 vs L-SHADE 3.82e-04) |
| `DynamicSphere` 50D | 70x yaxshi | CMA aniqlashtirish |
| `NoisyRastrigin` 50D | 19.5x yaxshi | Populyatsiyaga asoslangan qidiruv shovqinni o'rtachalaydi |

---

## 2. Asosiy matematik tashxis

### 2.1 Namunaviy kovariatsiya bu o'lchamda ishlamaydi

Joriy kod har avlodda eng yaxshi yarmidan kovariatsiya oladi:

```python
C = np.cov(pop[:pop_size // 2], rowvar=False)   # n = 3D namuna, D o'lcham
```

`n` ta namunadan `D` o'lchamda baholangan kovariatsiyaning spektral
xatosi `O(sqrt(D/n))` tartibida. Bu yerda `n = 3D`, demak xato
`O(sqrt(1/3)) ~ 0.58` — ya'ni **58%**. Kichik xos qiymatlarga mos xos
vektorlar amalda shovqin bo'lib qoladi, holbuki yomon shartlangan
masalada (`RotatedElliptic`) aynan o'sha yo'nalishlar hal qiluvchi.

LSHADE-cnEpSin `18*D` populyatsiya bilan ishlaydi (`n = 9D`, xato 0.33)
— shuning uchun u bizdan 949 barobar yaxshi. Bu populyatsiyani
kattalashtirish masalasi emas, **baholagichni almashtirish** masalasi.

### 2.2 Yechim: to'plamli (CMA uslubidagi) kovariatsiya

CMA-ES kovariatsiyani bir avlod namunasidan baholamaydi, balki
avlodlar bo'ylab to'playdi:

```
p_c  <- (1 - c_c) p_c + sqrt(c_c (2 - c_c) mu_eff) (m_new - m_old) / sigma
C    <- (1 - c_1 - c_mu) C + c_1 p_c p_c^T + c_mu * sum_i w_i y_i y_i^T
```

Bu baholagichning samarali namuna hajmi `~1/c_mu ~ (D+2)^2/(2 mu_eff)`
avlod, ya'ni `O(D^2)` tartibida o'sadi. Shuning uchun u o'lcham ortishi
bilan **buzilmaydi** — CMA-ES ning yomon shartlangan masalalardagi
ustunligi aynan shundan.

Xulosa: namunaviy kovariatsiya olib tashlanadi, o'rniga to'plamli
kovariatsiya qo'yiladi. Xos yoyilma standart CMA "dangasa yangilanish"
davri bilan hisoblanadi (`O(D^3)` amortizatsiya qilinib `O(D^2)` ga
tushadi).

### 2.3 Quyruq fazasi o'rniga uzluksiz aralashtirish

Joriy dizayn: byudjetning 85% i DE, keyin tarqalish `TAIL_DIV` dan
past bo'lsa qolgan 15% CMA-ES ga o'tadi.

Ikki o'lchangan nuqson:
- Shovqinli bir ekstremumli masalada ishga tushadi va **zarar qiladi**;
- Foydali bo'lishi mumkin bo'lgan erta bosqichda **ishga tusha olmaydi**.

Yechim: to'plamli `C` allaqachon mavjud bo'lgani uchun alohida faza
keraksiz. Har avlodda avlodlarning `p_cma` ulushi CMA taqsimotidan
olinadi:

```
x = m + sigma * B * D * z,    z ~ N(0, I)
```

`p_cma` qat'iy jadval bilan emas, mavjud kredit mexanizmi bilan
moslashadi. Shu bilan ham chegara (`TAIL_DIV`), ham faza chegarasi
yo'qoladi.

### 2.4 Shovqin detektori

Joriy algoritmda shovqinni aniqlash **umuman yo'q** — o'lchangan
mag'lubiyatning bevosita sababi.

Har `g_nu` avlodda joriy eng yaxshi nuqta qayta baholanadi:

```
nu <- (1 - alpha) nu + alpha * |f_1(x*) - f_2(x*)| / (IQR(f) + eps)
```

`nu` uch joyda ishlatiladi:

| Qayerda | Qanday | Nega |
|---|---|---|
| Tanlov | `f(u) < f(x) - kappa * nu * s` bo'lgandagina qabul | Shovqin ichidagi "yaxshilanish" populyatsiyani surib yuboradi |
| Kovariatsiya | `c_1, c_mu` ni `1/(1 + nu)` ga ko'paytirish | Shovqinli reyting asosida model tez o'rganilmasligi kerak |
| CMA tarmog'i | `p_cma` yuqori chegarasini `1/(1 + lambda_nu * nu)` ga | Shovqinda taqsimot modeli foydasiz |

Narxi: har `g_nu` avlodda 1 ta qo'shimcha baholash — ahamiyatsiz.

---

## 3. Komponentlar jadvali: nima olib tashlandi, nima qo'shildi

### 3.1 OLIB TASHLANADI

| Komponent | Manbasi | Nega olib tashlanadi |
|---|---|---|
| Har avlodda namunaviy kovariatsiya `np.cov(pop[:N/2])` | LSHADE-cnEpSin uslubida, o'z amalga oshirilishimiz | `n = 3D` da spektral xato ~58%; `RotatedElliptic` 949x mag'lubiyatining bevosita sababi (2.1-bo'lim) |
| `TAIL` / `TAIL_FRAC` / `TAIL_DIV` faza mexanizmi | LSHADE-SPACMA uslubida | Qat'iy chegara: shovqinli masalada noto'g'ri ishga tushadi, foydali joyda ishga tushmaydi (2.3) |
| `rho` prior "issiq start" (`t < 0.1` da `p_bank`, `p_eig` ga aralashtirish) | o'zimizniki, evristik | Matematik asosi yo'q; sehrli koeffitsientlar (2.0, 3.0) va sehrli chegara (0.1). Kredit mexanizmi buni o'zi hal qilishi kerak |
| `EIG_FREE`, `EIG_PRIOR` bayroqlari | o'zimizniki | O'lik konfiguratsiya sathi; sozlash natijasi ularni farqlamadi |
| Quyruqda populyatsiyani tashlab, faqat eng yaxshi nuqtadan davom etish | o'zimizniki | Populyatsiyadagi ma'lumot behuda yo'qoladi |

### 3.2 SAQLANADI (o'lchov bilan asoslangan)

| Komponent | Manbasi | Nega saqlanadi |
|---|---|---|
| `current-to-pbest/1` + arxiv | SHADE (Tanabe & Fukunaga, 2013) | Butun oilaning ishlaydigan yadrosi |
| Muvaffaqiyat tarixi bilan `F`/`CR` moslashuvi | SHADE | Yadro mexanizm |
| Chiziqli populyatsiya kamayishi (LPSR) | L-SHADE (2014) | Byudjetni oxirida aniqlashtirishga yo'naltiradi |
| `F`/`CR` cheklovlari va `Fw` vaznlash | jSO (Brest va b., 2017) | 0-bankning aniqlovchi xususiyati |
| Rank asosidagi tanlov bosimi (RSP) | LSHADE-RSP (Stanovov va b., 2018) | Standart, arzon, zararsiz |
| Midpoint-target chegara ishlovi | jSO / L-SHADE | Standart |
| **Ikki parametr rejimi ansambli** | o'zimizniki (yangi) | **Loyihaning eng kuchli o'lchangan natijasi**: `Schwefel` 50D da 75 060x. jSO va L-SHADE rejimlari teskari bog'liq (1.18e+02 vs 3.82e-04) |
| Yaxshilanish MIQDORI krediti (FIR) | o'zimizniki | Ansamblni boshqaradigan mexanizm; kengaytiriladi |

### 3.3 QO'SHILADI

| Yangi komponent | Ilhom manbasi | Matematik asos | Qaysi muammoni hal qiladi |
|---|---|---|---|
| **To'plamli kovariatsiya** `C` (rank-1 + rank-mu, evolyutsiya yo'li bilan) | CMA-ES (Hansen & Ostermeier, 2001) | Samarali namuna hajmi `O(D^2)` avlod; spektral xato o'lcham bilan buzilmaydi (2.2) | `RotatedElliptic` 949x; yuqori o'lchamdagi yaqinlashish |
| **Bitta kovariatsiya — ikki iste'molchi**: (a) eigen-bazisda crossover, (b) CMA taqsimotidan namuna olish | (a) LSHADE-cnEpSin, (b) LSHADE-SPACMA — lekin ikkalasi bitta modeldan | Bir model ikki rolda: bazis sifatida va taqsimot sifatida. Har ikki tarmoq bir xil kredit qoidasi bilan | Faza chegarasini yo'qotadi; `DynamicSphere` yutug'ini saqlab qolib, quyruq zararini olib tashlaydi |
| **Uzluksiz moslashuvchi aralashtirish** `p_cma` | LSHADE-SPACMA da qat'iy jadval; bizda kredit bilan | Jadval o'rniga o'lchangan foyda | Quyruq chegarasining ikkala nuqsoni (2.3) |
| **Shovqin baholagichi** `nu` va uning uch ta'siri | Noisy optimization adabiyoti (uncertainty handling) | Shovqin darajasiga qarab tanlov chegarasi, kovariatsiya tezligi va tarmoq ulushi damplanadi (2.4) | `NoisySphere` uchala o'lchamda |
| **Rejim ansamblini tarmoqlarga kengaytirish** | o'zimizniki | Parametr rejimi va generatsiya tarmog'i bitta kredit mexanizmi bilan boshqariladi | Birlashtirilgan, sehrli koeffitsientsiz boshqaruv |

### 3.4 Yangilik da'volari (maqola uchun)

| # | Da'vo | Eng yaqin oldingi ish | Farq |
|---|---|---|---|
| N1 | Bitta to'plamli kovariatsiya modeli ham DE crossover bazisi, ham CMA namuna tarmog'iga xizmat qiladi | LSHADE-cnEpSin (namunaviy kovariatsiya, faqat bazis); LSHADE-SPACMA (alohida CMA, jadval bilan) | Bitta model, ikki rol, faza yo'q |
| N2 | Yaxshilanish miqdori krediti bilan **parametr rejimlari** va **generatsiya tarmoqlari** ustidan birlashgan arbitraj | EA4eig (algoritmlar ansambli), SaDE (strategiyalar ansambli) | Ansambl algoritm darajasida emas, rejim va tarmoq darajasida; kredit miqdorga asoslangan |
| N3 | O'lchangan shovqin darajasi kovariatsiya o'rganishini, tarmoq ulushini va tanlov ochko'zligini damplaydi | Noisy-CMA-ES (qayta baholash), UH-CMA-ES | DE-CMA gibridida shovqin bilan boshqariladigan damping |

---

## 4. Gibrid algoritm qurishning qoidalari (o'zimga majburiy)

Bu qoidalar `docs/RESULTS.md` da aniqlangan metodologik kamchiliklardan
kelib chiqadi va yangi ishda **buzilmaydi**:

1. **Har bir komponent ablatsiya bilan asoslanadi.** Hissasi
   o'lchanmagan komponent algoritmda qolmaydi.
2. **Sozlash va baholash to'plamlari ajratiladi.** Parametrlar CEC-2014
   da sozlanadi, natijalar CEC-2017 da o'lchanadi. Joriy ishdagi eng
   jiddiy metodologik nuqson shu edi.
3. **Byudjet standart bo'ladi**: `10 000 * D`, CEC protokoli bo'yicha.
   Joriy `3000 * D` L-SHADE va jSO ni sun'iy ravishda zaiflashtirgan.
4. **51 ta mustaqil run**, CEC standarti.
5. **Bir xil sharoit**: bir xil seed sxemasi, bir xil chegara ishlovi,
   bir xil baholash hisoblagichi, bir xil to'xtash sharti.
6. **Statistika**: Friedman + Holm post-hoc (nazorat = yangi algoritm),
   instansiya bo'yicha Mann-Whitney + Holm, CD diagramma.
7. **Murakkablik** `T0/T1/T2` **barcha o'lchamlarda** o'lchanadi.
8. **Hech narsa tanlab olinmaydi**: barcha funksiyalar, barcha
   o'lchamlar, barcha natijalar e'lon qilinadi.
9. **Qayta ishlab chiqarilishi**: kod, seedlar va xom ma'lumot git da.
10. **Har bir olingan komponentning manbasi ochiq ko'rsatiladi.**
    Gibrid ishning qiymati yangi kombinatsiyada, olinganini yashirishda
    emas.

---

## 5. Eksperimental reja

| Bosqich | Mazmun | Hisoblash |
|---|---|---|
| B0 | ACE-SHADE ni yozish + birlik testlari (kovariatsiya yangilanishi, shovqin baholagichi, chegara ishlovi) | mahalliy |
| B1 | CEC-2014 (sozlash to'plami) da parametrlarni sozlash, 10D/30D, 15 run | Actions, ~1 soat |
| B2 | Parametrlarni **muzlatish** va e'lon qilish | — |
| B3 | CEC-2017 (baholash to'plami), D = 10/30/50/100, 51 run, `10000*D` | Actions, funksiya bo'yicha 30 ta parallel ish |
| B4 | Ablatsiya: har bir yangi komponent alohida o'chirilgan holda | Actions |
| B5 | Statistik tahlil + `docs/RESULTS_V2.md` | mahalliy |

Raqobatchilar: L-SHADE, jSO, LSHADE-cnEpSin, LSHADE-SPACMA, LSHADE-RSP,
AGSK, NL-SHADE-RSP + nazorat sifatida DE va CMA-ES.

---

## 6. Xavflar ro'yxati (oldindan qayd etilgan)

| Xavf | Ehtimollik | Javob |
|---|---|---|
| To'plamli `C` DE yadrosi bilan mos kelmasligi (DE qadamlar CMA qadamlar taqsimotiga mos emas) | o'rta | `y_i` faqat **qabul qilingan** trialardan olinadi va `sigma` ga normallashtiriladi; ablatsiya bilan tekshiriladi |
| `10000*D` byudjetda L-SHADE va jSO kuchayadi va farq kamayadi | **yuqori** | Bu halol taqqoslash narxi. Joriy ustunlikning bir qismi byudjet tanlovidan kelgan bo'lishi mumkin |
| CEC-2017 ni to'g'ri amalga oshirish (shift/rotate ma'lumotlari) | o'rta | Rasmiy ma'lumot fayllari ishlatiladi, sintetik emas |
| 100D da hisoblash narxi | o'rta | Funksiya bo'yicha shardlash; `O(D^2)` amortizatsiya |
| Yangi komponentlar bir-birini bekor qilishi | o'rta | B4 ablatsiyasi har birini alohida o'lchaydi |

---

## 7. Foydalanuvchidan kutilayotgan qaror

Quyidagilar tasdiqlanishi kerak:

1. **Nom**: ACE-SHADE yoki boshqa?
2. **Benchmark**: CEC-2017 (tavsiya) yoki CEC-2022?
3. **Sozlash to'plami**: CEC-2014 (tavsiya) yoki BBOB kichik to'plami?
4. **Raqobatchilar ro'yxati** yuqoridagidek yetarlimi?
5. Qaysi bosqichdan boshlash: B0 (kod yozish) darhol?
