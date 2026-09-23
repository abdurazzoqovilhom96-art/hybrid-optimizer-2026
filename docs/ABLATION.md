# Ablatsiya tajribalari

Barcha tajribalar asl protokolda o'tkazilgan: `FES = 3000 x D`, bir xil seed
sxemasi (`42 + 1000*D + run`), shovqinli masalalarda yakuniy qiymat shovqinsiz
`true_func` orqali hisoblangan. Ko'rsatilgan qiymat — **mediana**.

---

## 1-tajriba. TEMOA_V10_HYBRID komponentlari (yutqazayotgan masalalar, 11 run)

Maqsad: hozirgi algoritmning qaysi qismi `Schwefel` va `NoisyRastrigin` da
yo'qotishga sabab bo'layotganini aniqlash.

| Konfiguratsiya | Schwefel 30D | Schwefel 50D | Schwefel 100D | NoisyRast. 50D | NoisyRast. 100D |
|---|---|---|---|---|---|
| BASE (hozirgi) | 2428.0 | 5704.9 | 14865.8 | 58.7 | 144.4 |
| − CR_FLOOR | 2191.2 | 6336.7 | 12714.4 | 58.7 | 147.3 |
| − (1+1)-ES quyrug'i | 2428.0 | 5704.9 | 14865.8 | 58.7 | 144.4 |
| − eigen crossover | 1895.1 | 4323.6 | 11569.1 | 53.8 | 120.5 |
| operator portfeli = bir tekis | 1914.9 | 5606.4 | 12321.0 | 52.8 | 139.4 |
| **faqat current-to-pbest (portfel yo'q)** | **0.0** | **745.5** | **7422.5** | **39.1** | 146.5 |
| N_init = 18·D | 1717.4 | 5369.5 | 10028.3 | 56.8 | 121.7 |
| N_min = 20 | 2210.9 | 6495.2 | 10442.6 | 65.7 | 149.3 |
| 18·D + N_min=20 − CR_FLOOR | 1717.4 | 4639.0 | 10462.2 | 51.8 | 133.9 |
| DE (raqib, mos yozuvlar) | 2031.1 | 4282.0 | 10703.8 | 35.0 | 109.4 |

### Xulosalar

1. **Metafora asosidagi operator portfeli — yo'qotishning asosiy manbai.**
   GWO lider-DE (`op=1`), WOA spirali (`op=2`) va HHO Levy sakrashi (`op=3`)
   o'chirilib, faqat `current-to-pbest-w/1` qoldirilganda Schwefel 30D xatosi
   **2428 → 0.0**, 50D **5705 → 746**, 100D **14866 → 7423** ga tushadi.

   Sababi mexanizmda: kredit muvaffaqiyat *ulushi*ga (`improved.mean()`)
   asoslangan, ya'ni qadam *kattaligi* hisobga olinmaydi. Ko'p ekstremumli
   relyefda eng mayda va "xavfsiz" qadam qo'yadigan operator eng yuqori ulushni
   oladi, natijada probability matching ekspluatatsiyaga qulaydi va
   tadqiqot operatorlari `P_MIN = 0.05` ga siqiladi.

2. **(1+1)-ES quyrug'i mutlaqo ishlamaydi.** Beshta instansiyada ham natija
   BASE bilan raqamma-raqam bir xil (`LS_FRACTION = 0.05`). Byudjetning 5% i
   behuda ketadi.

3. **Eigen-crossover** bu masalalarda zarar qiladi (separabel relyef), lekin
   `RotatedElliptic` da hal qiluvchi darajada foydali. Demak uni olib tashlash
   emas, moslashuvni tuzatish kerak: hozirgi `[0.1, 0.9]` cheklovi uni hech
   qachon to'liq o'chira olmaydi.

4. **Populyatsiya hajmi** `6·D` (va `POP_MAX = 500` cheklovi) juda kichik;
   `18·D` (L-SHADE standarti) yuqori o'lchamlarda barqaror yaxshilanish beradi.

5. `N_min = 20` ga oshirish yordam bermaydi — LPSR ning yakuniy nuqtasi
   muammo emas.

---

## 2-tajriba. CBA-SHADE ning o'z komponentlari (30D, 11 run)

12 ta funksiya bo'yicha o'rtacha rank (kichikroq = yaxshiroq); mos yozuvlar
sifatida L-SHADE, jSO, LSHADE-cnEpSin, CMA-ES va eski TEMOA qo'shilgan.

| Konfiguratsiya | O'rtacha rank |
|---|---|
| pop = 8·D | **5.75** |
| pop = 12·D, quyruq 10% | 5.79 |
| TEMOA (eski) | 5.92 |
| pop = 12·D, eigen yo'q | 7.58 |
| pop = 12·D | 7.67 |
| pop = 12·D, ρ-prior yo'q | 8.25 |
| pop = 12·D, eigen oralig'i [0.1, 0.9] | 8.38 |
| pop = 12·D, kredit = muvaffaqiyat ulushi | 8.38 |
| pop = 12·D, RSP yo'q | 8.92 |
| pop = 18·D | 11.58 |

### Xulosalar

- **Populyatsiya hajmi — eng ta'sirli parametr.** `18·D` (L-SHADE standarti)
  bu byudjet uchun juda katta: L-SHADE uni `10000·D` uchun sozlagan, bu yerda
  esa byudjet `3000·D`.
- **Eigen-crossover zarur:** uni o'chirganda RotatedElliptic 30D da
  `0.84 → 2290` (2700 marta yomon).
- **RSP, ρ-prior va miqdor krediti** — uchalasi ham ijobiy, lekin ta'siri
  populyatsiya hajmiga qaraganda ancha kichik.
- **CMA quyrug'i qarama-qarshi ta'sir qiladi:** DynamicSphere 30D da
  `0.084 → 0.0038` (foydali), NoisyRastrigin 30D da `1.55 → 14.0` (zararli).

---

## 3-tajriba. Populyatsiya hajmi x quyruq rejimi (30D, 15 run)

2-tajribadagi quyruq muammosini hal qilish uchun quyruq **shartli** qilindi:
u faqat populyatsiyaning nisbiy tarqalishi `TAIL_DIV` dan past tushganda
(ya'ni qidiruv bitta havzaga yig'ilganda) ishga tushadi.

| Konfiguratsiya | O'rtacha rank |
|---|---|
| **pop 6·D, shartli quyruq** | **4.375** |
| pop 6·D, quyruq yo'q | 5.292 |
| pop 8·D, doimiy quyruq | 5.750 |
| pop 8·D, shartli quyruq | 5.833 |
| SHADE | 6.500 |
| pop 8·D, quyruq yo'q | 7.125 |
| pop 10·D, shartli quyruq | 7.167 |
| pop 12·D, shartli quyruq | 8.125 |
| LSHADE-cnEpSin | 8.875 |
| pop 10·D, quyruq yo'q | 9.208 |
| jSO | 9.458 |
| pop 12·D, quyruq yo'q | 10.250 |
| DE | 10.292 |
| CMA-ES | 10.375 |
| L-SHADE | 11.375 |

### Xulosalar

- **Shartli quyruq har bir populyatsiya hajmida quyruqsiz variantdan ustun**
  (6·D: 4.375 va 5.292; 8·D: 5.833 va 7.125; 10·D: 7.167 va 9.208;
  12·D: 8.125 va 10.250). Ya'ni quyruqning o'zi emas, uni *qachon* ishga
  tushirish muhim edi.
- **`POP_FACTOR = 6` eng yaxshi** (6 → 8 → 10 → 12 bo'yicha rank monoton
  yomonlashadi). Qizig'i shundaki, eski TEMOA ham `6·D` ishlatgan — demak
  populyatsiya hajmi to'g'ri tanlangan, muammo operator portfelida edi.
- **NoisyRastrigin 30D hal qilindi:** `0.205` (eski TEMOA `31.8`,
  DE `53.2`, L-SHADE `1.65`) — ya'ni ota-onasidan 8 marta yaxshi.
- **Schwefel — qolgan yagona zaif nuqta:** barcha konfiguratsiyalar `118.4`
  qiymatida qotib qoladi, L-SHADE esa `0.071` beradi. Bu tasodifiy emas:
  jSO uslubidagi xotira boshlang'ich qiymati (`M_CR = 0.8` va doimiy `0.9`
  terminal katak) yuqori CR ga moyil, bu esa separabel Schwefel uchun zarar.
