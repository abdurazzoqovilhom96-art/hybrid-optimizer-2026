# Avtonom rejim promti

Quyidagi matn Routine yoki `/loop` orqali shu sessiyaga davriy yuboriladi.
To'xtatish uchun: "avtonom rejimni to'xtat" deb yozing.

Arxitektura: **og'ir hisoblash GitHub Actions'da bajariladi**, bu sessiya
faqat boshqaradi va yakuniy tahlilni yozadi. Shuning uchun konteyner uzilib
qolsa ham eksperiment to'xtamaydi.

---

## AVTONOM REJIM — CBA-SHADE eksperimentini yakunlash

Foydalanuvchi yo'q. **Hech narsa so'rama, tasdiq kutma.** Har uyg'onishda
bitta qadam bajar, natijani push qil, tugat.

### 0. HOLATNI GIT'DAN O'QI (har safar)

```bash
cd /home/user/hybrid-optimizer-2026 2>/dev/null || git clone -b claude/youthful-pascal-sjxqnn \
  https://github.com/abdurazzoqovilhom96-art/hybrid-optimizer-2026.git /home/user/hybrid-optimizer-2026
cd /home/user/hybrid-optimizer-2026
git checkout claude/youthful-pascal-sjxqnn && git pull --rebase origin claude/youthful-pascal-sjxqnn
cat results/STATUS.md
```

So'ng `mcp__github__actions_list` bilan oxirgi workflow ishlarini ko'r
(owner `abdurazzoqovilhom96-art`, repo `hybrid-optimizer-2026`).

### 1. FAZA A — parametrlarni muzlatish

`results/STATUS.md` da `FAZA_A: tugadi` yo'q bo'lsa:

- Agar `/tmp/.../scratchpad/r5.log` mavjud bo'lsa, uni o'qi va o'rtacha rank
  bo'yicha eng yaxshi `TAIL_DIV` / `POP_FACTOR` / `MEM_INIT` ni `cba_shade.py`
  ning standart qiymatlari sifatida muzlat.
- Fayl yo'q bo'lsa (konteyner yangilangan) — joriy qiymatlarni yakuniy deb
  qabul qil va fazani tugagan deb belgila. **Qayta sozlash tajribasi
  o'tkazma** — vaqt yo'q va bu natijalarni tanlab olish xavfini tug'diradi.
- `docs/ABLATION.md` ni yangila, commit + push, STATUS.md ga `FAZA_A: tugadi`.

### 2. FAZA B — to'liq eksperimentni GitHub'da ishga tushirish

`FAZA_A` tugagan va `FAZA_B` boshlanmagan bo'lsa:

```bash
date -u > .github/RUN_EXPERIMENT
git add -A && git commit -m "Start the full experiment run" && git push origin claude/youthful-pascal-sjxqnn
```

STATUS.md ga `FAZA_B: ishga tushdi` va workflow run havolasini yoz, push qil.
Bu 30D/50D/100D ni **parallel** ishlatadi (~2.5 soat) va tugagach natijalarni
`results/full/` ga o'zi commit qiladi.

### 3. FAZA B kuzatuvi

`FAZA_B: ishga tushdi` bo'lsa: `actions_list` bilan holatni ko'r.

- `in_progress` yoki `queued` — hech narsa qilma, STATUS.md dagi vaqtni
  yangilab push qil, tugat.
- `completed` + `success` — `git pull` qil, `results/full/` paydo bo'lganini
  tekshir, `FAZA_B: tugadi` deb belgila va darhol FAZA C ga o't.
- `completed` + `failure` — `get_job_logs` bilan xatoni o'qi, `cba_shade.py`
  yoki workflow'dagi xatoni tuzat, marker faylni yangilab qayta ishga tushir.
  **Uch marta muvaffaqiyatsiz bo'lsa** to'xta va `results/STATUS.md` ga
  sababni batafsil yozib push qil.

### 4. FAZA C — yakuniy tahlil (`docs/RESULTS.md`)

`results/full/tables/` dagi barcha CSV larni o'qi va `docs/RESULTS.md` yoz.
**Majburiy bo'limlar:**

1. Protokol: algoritmlar, 12 funksiya, 3 o'lcham, run soni, byudjet `3000·D`,
   seed sxemasi, CEC aniqlik chegarasi `1e-8`.
2. `mean ± std` jadvali (36 instansiya x barcha algoritmlar).
3. Friedman: χ², p, Iman–Davenport F, o'rtacha ranklar.
4. **Post-hoc** (nazorat = CBA-SHADE): z, Holm bilan tuzatilgan p, CD.
   Bu eng muhim jadval — oldingi maqolaning aynan shu yerda muammosi bor edi.
5. Mann–Whitney U + Holm bo'yicha `+/=/−` jadvali.
6. O'lcham bo'yicha ajratilgan ranklar (30 / 50 / 100 alohida).
7. Hisoblash murakkabligi (T0/T1/T2), `results/complexity/` dan.
8. **ZAIF TOMONLAR** — CBA-SHADE yutqazgan yoki teng chiqqan har bir
   instansiya, raqamlari va ehtimoliy sababi bilan. `1e-8` chegarasidan
   keyin qaysi funksiyalar umuman ajratuvchi ekanini ham ayt.
9. Maqola uchun keyingi qadamlar (CEC-2017/2022 to'plami, `10000·D` byudjet,
   51 run, AGSK / NL-SHADE-RSP kabi yangi raqiblar).

Commit + push. STATUS.md ga `FAZA_C: tugadi`.

### 5. FAZA D — yakunlash

Routine/loop ni to'xtat va foydalanuvchiga qisqa yakuniy xabar yoz: asosiy
raqamlar, qaysi fayllarni ochish kerak, qanday zaif tomonlar qoldi.

---

## QAT'IY QOIDALAR

1. **Faqat `claude/youthful-pascal-sjxqnn` branchiga push qil**, hech qachon `main` ga.
2. Teg push qilishga ruxsat yo'q (403) — ishga tushirish faqat marker fayl orqali.
3. **Natijalarni bezama.** CBA-SHADE yutqazgan bo'lsa raqamlari bilan ochiq yoz.
   Maqsad halol Q1 darajasi, chiroyli jadval emas.
4. **FAZA A tugagach algoritmni va parametrlarni o'zgartirma.** Natija yoqmasa
   ham qayta sozlama — bu natijalarni tanlab olish bo'ladi.
5. Har bir qadamdan keyin **darhol push qil** — konteyner ephemeral.
6. Xatolikni STATUS.md ga yoz, push qil, keyingi uyg'onishda qayta urin.
7. **Foydalanuvchidan hech narsa so'rama.** Qarorni o'zing qabul qilib,
   sababini STATUS.md ga yozib qo'y.
