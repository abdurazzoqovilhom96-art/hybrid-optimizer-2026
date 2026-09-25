# Prior art — adabiyot tekshiruvi

**Sana:** 2026-09-25 · **Maqsad:** hech qanday yangilik da'vosi tekshiruvsiz
qilinmasligi. Har bir g'oya uchun: kim qilgan, qayerda, bizga nima qoladi.

Bu fayl **ish davomida to'ldiriladi**. Tekshirilmagan g'oyaga "yangi" deb yorliq
qo'yilmaydi.

---

## 1. Qat'iy prior art — bizniki EMAS, da'vo qilinmaydi

| G'oya | Kim qilgan | Bizga nima qoladi |
|---|---|---|
| **Eigen / kovariatsiya-bazisli crossover** | **EA4eig** — CEC'2022 g'olibi; **L-SRTDE**; LSHADE-cnEpSin | Faqat **foydalanish**, iqtibos bilan. Hissa sifatida aytilmaydi. TEMOA'ning bu qismi original emas |
| Kovariatsiyani to'liq moslash, rotatsiyaga invariantlik | **CMA-ES** (Hansen & Ostermeier) | Raqib sifatida majburiy. Bizning "ill-conditioned'da kuchlimiz" da'vomiz uning maydonida sinaladi |
| **IPOP / BIPOP restart** | Auger & Hansen (2005); Hansen (2009) | Schwefel qulashiga qarshi **ma'lum yechim**, iqtibos bilan ishlatiladi |
| Success-history parametr moslash (F, CR xotirasi) | SHADE, **L-SHADE** (Tanabe & Fukunaga), **jSO** (Brest va b.) | TEMOA'ning yadrosi. Bizniki emas |
| Linear population size reduction (LPSR) | L-SHADE | Bizniki emas |
| **Probability matching / Adaptive Pursuit / DMAB / extreme-value credit** | Thierens; Fialho, Da Costa, Schoenauer, Sebag | AOS'ning standart oilasi. Bizning taqqoslash bazamiz |
| **RL bilan operator/parametr tanlash** | **DE-DDQN** (Sharma, Komninos, López-Ibáñez, Kazakov, GECCO'19); RL-DAS; LDE | Mavjud. Bizning farqimiz (agar bo'lsa) **mukofot ufqida**, RL ishlatishning o'zida emas |
| DE + Hyperband, ko'p-fidelity HPO | **DEHB** (Awad, Mallik, Hutter) | HPO yo'nalishi band |
| DE neyroarxitektura qidiruvi uchun | **DE-NAS** (Awad va b.), tabular NAS-Bench'larda | NAS yo'nalishi band |
| Evolyutsion mixed-precision kvantlash | **EvoQ**, MixQuant | Kvantlash yo'nalishi band |
| Binar metaevristik feature selection | juda ko'p; 2025 sharhlari to'yinganlikni qayd etadi | Asosiy hissa sifatida yaroqsiz |
| MOEA'da konvergensiya+diversity kreditli AOS | ko'p maqsadli AOS adabiyoti | **S2 nomzodimizga yaqin** — pastga qarang |

**Alohida ogohlantirish.** 2025-yilgi tahlil 26 ta nashr etilgan metaevristika
avvalgilariga **strukturaviy jihatdan aynan bir xil** ekanini va 512 juft yuqori
o'xshashlikda ekanini ko'rsatdi. "To'rtta metaevristikani birlashtirdik" degan
ramka aynan shu tanqid ostida — shuning uchun u tashlandi.

---

## 2. Bizniki — o'lchangan, birovdan olinmagan

Bular **natijalar**, g'oya emas; ularni hech kim bizdan oldin o'lchamagan,
chunki ular aynan shu algoritmga tegishli:

| Topilma | Dalil |
|---|---|
| WOA-spiral operatori sinalgan **har bir** funksiyada dominatsiya qilingan | operator ablatsiyasi, 30D, 8 yurish |
| (1+1)-ES dumi o'lchov doirasida **hech narsa bermaydi** | ikkita funksiyada bit-ma-bit bir xil natija |
| Eigen-crossover RotatedElliptic'da **62×** beradi, boshqa joyda yo'q | op0-only, eigen bilan/siz |
| `POP_FACTOR` RotatedElliptic'da **6600×** ta'sir qiladi | NP=180 vs NP=500 |
| **Spiral Schwefel'da fitness foydasining 66% ini beradi** va ayni paytda yurishni 6 tartibga yo'qotadigan qulashni keltirib chiqaradi | V10 instrumentatsiyasi (`logger=`) |
| `CR_FLOOR` gipotezasi **rad etildi** | o'chirilganda 2685 → 2350, ya'ni sabab emas |
| ΔF-asosidagi kredit gipotezasi **rad etildi** | zararli operator foydaning 66% ini beradi → ΔF kredit uni ko'proq mukofotlagan bo'lardi |

Oxirgi ikkitasi — **salbiy natijalar**, va ular alohida qiymatga ega: ular
"aniq yechim" bo'lib ko'ringan ikki yo'lning berk ekanini o'lchov bilan
ko'rsatadi.

---

## 3. Nomzod g'oyalar — HALI TEKSHIRILMAGAN, da'vo qilinmaydi

Bu uchtasi §2 dagi o'lchovdan kelib chiqadi. **Har biri uchun to'liq adabiyot
tekshiruvi bajarilmaguncha "yangi" deyilmaydi.**

### S1 — Lineage credit (avlod bo'yicha kredit)
Operatorni farzandining **darhol** foydasi bilan emas, `k` avloddan keyin omon
qolgan avlodlarining sifati bilan mukofotlash.

- **Prior-art xavfi: O'RTA.** Qidiruvda "delayed mutations should be valued
  through the executable lineage they open" degan iborani uchratdim — ya'ni
  savol adabiyotda qo'yilgan. Kim, qayerda va qanday hal qilganini aniqlash
  kerak.
- **Tekshirish rejasi:** genetik dasturlashda "lineage/ancestry-based credit",
  MOEA'da "delayed reward operator selection", va DE'da "multi-generation
  credit assignment" bo'yicha qidiruv. Natija shu yerga yoziladi.

### S2 — Diversity-discounted credit
Kreditni operator o'sha avlodda keltirgan diversity yo'qotishiga proporsional
jazolash.

- **Prior-art xavfi: YUQORI.** Ko'p maqsadli AOS adabiyotida kredit allaqachon
  konvergensiya **va** diversity yaxshilanishidan tuziladi. Bir maqsadli DE
  uchun ham o'xshash ishlar bo'lishi ehtimoli katta.
- **Ehtimoliy xulosa:** bu mustaqil hissa bo'lolmaydi. Ko'pi bilan ma'lum
  usulning bir maqsadli holatga ko'chirilishi — va buni shunday atash kerak.

### S3 — Uzoq gorizontli RL (AOS ni bandit emas, MDP sifatida)
Holat = qidiruv holati fichalari, mukofot = yurish oxiridagi natija, γ→1.

- **Prior-art xavfi: O'RTA.** DE-DDQN allaqachon DDQN ishlatadi va uning
  Q-funksiyasi diskontlangan kelajakni hisobga oladi. Ammo uning **uchala
  mukofoti ham (R1, R2, R3) darhol fitness farqiga asoslangan**. Ya'ni
  mashinasi uzoq gorizontli, signali esa myopik.
- **Ochiq savol:** DE-DDQN bizning diversity-mortgaging holatimizni ushlaydimi?
  Buni **o'lchash mumkin** — va agar ushlamasa, bu aniq, tor va halol hissa.
- **Tekshirish rejasi:** DE-DDQN kodini (GitHub'da mavjud) bizning diagnostik
  landshaftimizda ishga tushirish.

---

## 4. Eng past xavfli hissa — diagnostik asbob

"Diversity-mortgaging" hodisasini **boshqariladigan masala oilasi** sifatida
rasmiylashtirish: operator myopik jihatdan jozibador, global jihatdan zararli
bo'ladigan, sozlanadigan landshaft. Har qanday AOS usulini shu orqali o'tkazish
mumkin.

- **Prior-art xavfi: PAST.** Benchmark va diagnostik asboblar odatda mustaqil
  nashr qilinadi, va bizning versiyamiz aniq o'lchangan hodisadan kelib chiqadi.
- **Nega muhim:** bu §3 dagi uchala nomzod ham ishlamay qolsa ham qoladigan
  natija. Ya'ni berk ko'chaga qarshi sug'urta.

---

## 5. Qoidalar

1. **Tekshirilmagan g'oya "yangi" deb atalmaydi** — na kodda, na hisobotda, na
   maqolada.
2. Har bir tekshiruv natijasi shu faylga yoziladi, sanasi bilan.
3. Prior art topilsa — **iqtibos bilan foydalanamiz**, yashirmaymiz. Ma'lum
   usulni yangi kontekstga ko'chirish ham qiymat, lekin u shunday atalishi kerak.
4. Shubha bo'lsa — da'vo qilmaymiz.

---

## 6. Manbalar

- [How novel is the novelty? Taxonomic framework of metaheuristics, *AI Review* 2025](https://link.springer.com/article/10.1007/s10462-025-11456-8)
- [Applications, classifications and challenges of recent metaheuristics, *AI Review* 2025](https://link.springer.com/article/10.1007/s10462-025-11377-6)
- [Adaptive Operator Selection for Optimization (Fialho, PhD thesis)](https://groups.csail.mit.edu/EVO-DesignOpt/pb/uploads/Site/FialhoThesisDraft.pdf)
- [Credit Assignment in Adaptive Evolutionary Algorithms](https://arxiv.org/pdf/0907.0592)
- [Deep Reinforcement Learning Based Parameter Control in DE (DE-DDQN)](https://doi.org/10.1145/3321707.3321813) · [kod](https://github.com/mudita11/DE-DDQN)
- [DEHB: Evolutionary Hyperband for HPO](https://arxiv.org/pdf/2105.09821)
- [Differential Evolution for Neural Architecture Search](https://arxiv.org/pdf/2012.06400)
- [EvoQ: Mixed Precision Quantization via Sensitivity Guided Evolutionary Search](https://ieeexplore.ieee.org/document/9207413/)
- [Analysis and simplification of the winner of the CEC 2022 competition (EA4eig)](https://pubmed.ncbi.nlm.nih.gov/40367330/)
- [Alternative restart strategies for CMA-ES](https://dl.acm.org/doi/10.1007/978-3-642-32937-1_30)
- [Impacts of invariance in search: CMA-ES and PSO on ill-conditioned problems](https://www.sciencedirect.com/science/article/abs/pii/S1568494611000974)
- [NAS-Bench-Suite: NAS Evaluation is (Now) Surprisingly Easy](https://arxiv.org/html/2201.13396)
- [Opfunu: Python library for optimization benchmark functions](https://openresearchsoftware.metajnl.com/articles/10.5334/jors.508)
