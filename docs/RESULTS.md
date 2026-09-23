# CBA-SHADE: eksperimental natijalar

> **Holat: to'liq tajriba hali ishlamoqda.** Protokol bo'limi yakuniy;
> raqamli bo'limlar 100D bo'lagi tugagach to'ldiriladi. Joriy holat:
> `results/STATUS.md`.

## 1. Eksperimental protokol

### 1.1 Taqqoslanayotgan algoritmlar

Yetti algoritm bir xil sharoitda taqqoslanadi. Barcha bazaviy usullar shu
repozitoriyda qaytadan amalga oshirilgan va bir xil baholash hisoblagichidan
(`Tracker`) foydalanadi, shuning uchun byudjet hisobi ular orasida bir xil.

| Algoritm | Turi | Manba |
|---|---|---|
| **CBA-SHADE** | taklif etilayotgan | shu ish |
| L-SHADE | DE, chiziqli populyatsiya kamayishi | Tanabe & Fukunaga, 2014 |
| jSO | L-SHADE ning parametr cheklovli varianti | Brest va b., 2017 |
| LSHADE-cnEpSin | L-SHADE + eigen crossover + sinusoidal moslashuv | Awad va b., 2017 |
| SHADE | tarixga asoslangan moslashuvchi DE | Tanabe & Fukunaga, 2013 |
| CMA-ES | kovariatsiya matritsasi moslashuvi | Hansen & Ostermeier, 2001 |
| DE/rand/1/bin | klassik nazorat | Storn & Price, 1997 |

### 1.2 Test to'plami

12 ta funksiya, har biri uchta o'lchamda (30, 50, 100) — jami **36 instansiya**.
Har bir funksiya optimumi tasodifiy siljitilgan, shuning uchun koordinata
boshidagi yechim afzallik bermaydi.

| Funksiya | Xususiyati | Sinov maqsadi |
|---|---|---|
| Ackley, Griewank, Levy, Schwefel | ko'p ekstremumli | global qidiruv |
| Rastrigin (shovqinli) | ko'p ekstremumli + shovqin | shovqinga chidamlilik |
| Sphere (shovqinli), Sphere (dinamik) | bir ekstremumli, o'zgaruvchan | kuzatuv qobiliyati |
| Rosenbrock | tor vodiy | mahalliy model sifati |
| BentCigar, Zakharov, RotatedElliptic | yomon shartlangan | kovariatsiyani o'rganish |
| Composition | aralash | umumiy mustahkamlik |

### 1.3 Byudjet va takrorlashlar

- Baholashlar soni: `FES = 3000 · D` (30D → 90 000; 50D → 150 000; 100D → 300 000)
- Mustaqil ishga tushirishlar: har bir (algoritm, funksiya, o'lcham) uchun **30**
- Jami: 7 × 12 × 3 × 30 = **7 560 ishga tushirish**

### 1.4 Seed sxemasi

Seed `42 + 1000·D + run` formulasi bilan beriladi va har bir ishga tushirishdan
oldin qayta o'rnatiladi. Shuning uchun **berilgan (o'lcham, run) juftligida
barcha algoritmlar bir xil boshlang'ich populyatsiyadan va bir xil siljitish
vektoridan boshlaydi** — taqqoslash juftlashtirilgan bo'ladi va tasodifiy
boshlang'ich holat farqi natijaga ta'sir qilmaydi.

### 1.5 Aniqlik chegarasi

CEC musobaqalari konventsiyasiga muvofiq `1e-8` dan kichik xato nol deb
qabul qilinadi (`PRECISION_FLOOR`). Bu metodologik jihatdan zarur: busiz
`1.5e-32` va `4.1e-31` kabi ikki aniq yechim "g'alaba" va "mag'lubiyat"
sifatida sanaladi va ranklar mashina aniqligidagi shovqin bilan buziladi.
Xom qiymatlar `full_raw_results.csv` da o'zgarishsiz saqlanadi.

### 1.6 Statistik tahlil

1. **Friedman testi** — barcha algoritmlar bo'yicha umumiy farqni tekshiradi;
   χ² statistikasi va Iman–Davenport F tuzatishi bilan.
2. **Post-hoc taqqoslash** — CBA-SHADE nazorat sifatida, har bir raqib bilan
   juftlik taqqoslash; `z` statistikasi, **Holm tuzatishi** bilan `p`
   qiymatlari va Nemenyi tanqidiy farqi (CD).
3. **Mann–Whitney U testi** — har bir instansiyada juftlik taqqoslash,
   Holm tuzatishi bilan; natija `+/=/−` (g'alaba/teng/mag'lubiyat) jadvalida.
4. **Hisoblash murakkabligi** — CEC protokoli bo'yicha `T0`, `T1`, `T2`.

Barcha testlar `α = 0.05` darajasida.

---

## 2. Umumiy natijalar

*(100D bo'lagi tugagach to'ldiriladi)*

## 3. Friedman testi va ranklar

*(100D bo'lagi tugagach to'ldiriladi)*

## 4. Post-hoc taqqoslash (nazorat: CBA-SHADE)

*(100D bo'lagi tugagach to'ldiriladi)*

## 5. Instansiya bo'yicha g'alaba/teng/mag'lubiyat

*(100D bo'lagi tugagach to'ldiriladi)*

## 6. O'lcham bo'yicha tahlil

*(100D bo'lagi tugagach to'ldiriladi)*

## 7. Hisoblash murakkabligi

*(100D bo'lagi tugagach to'ldiriladi)*

## 8. Zaif tomonlar va cheklovlar

*(100D bo'lagi tugagach to'ldiriladi — CBA-SHADE yutqazgan yoki teng chiqqan
har bir instansiya raqamlari bilan ochiq keltiriladi)*

## 9. Maqola uchun keyingi qadamlar

*(100D bo'lagi tugagach to'ldiriladi)*
