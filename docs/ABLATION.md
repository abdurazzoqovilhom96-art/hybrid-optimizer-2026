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
