# Avtonom rejim promti

Bu matn soatlik Routine (rejalashtirilgan trigger) orqali shu sessiyaga
yuboriladi. Uni to'xtatish uchun: "avtonom rejimni to'xtat" deb yozing.

---

## AVTONOM REJIM — CBA-SHADE eksperimentini boshidan oxirigacha yakunlash

Sen bu ishni foydalanuvchisiz, mustaqil yakunlaysan. **Hech narsa so'rama,
tasdiq kutma, "davom etaymi?" deb so'rama.** Har bir uyg'onishda quyidagi
holat mashinasini bajarasan.

### 0. HOLATNI ANIQLASH (har safar, istisnosiz)

```bash
cd /home/user/hybrid-optimizer-2026 2>/dev/null || \
  git clone -b claude/youthful-pascal-sjxqnn \
    https://github.com/abdurazzoqovilhom96-art/hybrid-optimizer-2026.git \
    /home/user/hybrid-optimizer-2026
cd /home/user/hybrid-optimizer-2026
git checkout claude/youthful-pascal-sjxqnn && git pull --rebase origin claude/youthful-pascal-sjxqnn
python3 -c "import numpy, pandas, scipy, joblib, matplotlib" 2>/dev/null || \
  pip install --quiet numpy pandas scipy joblib matplotlib
cat results/STATUS.md 2>/dev/null
pgrep -af "cba_shade.py|round[0-9]*.py" || echo "fonda ish yo'q"
```

Konteyner istalgan paytda qayta tiklanishi mumkin — shuning uchun **holat
faqat git'dan o'qiladi**, xotiradan emas.

### 1. AGAR FONDA ISH KETAYOTGAN BO'LSA

Hech narsa boshlama, hech narsani o'zgartirma. Faqat `results/STATUS.md` dagi
"oxirgi tekshiruv" vaqtini yangilab, push qil va keyingi uyg'onishni kut.

### 2. FAZA A — parametrlarni muzlatish (agar hali muzlatilmagan bo'lsa)

`results/STATUS.md` da `FAZA_A: tugadi` yo'q bo'lsa:

1. `/tmp/.../scratchpad/r4.log` (4-bosqich sozlash natijasi) mavjud bo'lsa —
   uni o'qi. Yo'q bo'lsa (konteyner yangilangan), bu fazani tugagan deb hisobla:
   joriy `cba_shade.py` dagi qiymatlar yakuniy.
2. O'rtacha rank bo'yicha eng yaxshi konfiguratsiyani tanla va uni
   `CBA_SHADE` ning standart qiymatlari sifatida muzlat.
3. `docs/ABLATION.md` ga 4-bosqich jadvalini qo'sh.
4. Commit + push. `results/STATUS.md` ga `FAZA_A: tugadi` yoz.

### 3. FAZA B — to'liq eksperiment (uchta bo'lak, ketma-ket)

Har bir bo'lak alohida, **fonda** (`run_in_background: true`) ishga tushiriladi.
Bir vaqtda faqat bittasi. Tartib: 30 → 50 → 100.

```bash
DIMS=30  RUNS=30 OUT_DIR=out_d30  python3 cba_shade.py   # ~35 daqiqa
DIMS=50  RUNS=30 OUT_DIR=out_d50  python3 cba_shade.py   # ~55 daqiqa
DIMS=100 RUNS=30 OUT_DIR=out_d100 python3 cba_shade.py   # ~2.2 soat
```

Bo'lak tugaganda (`out_dNN/tables/summary_results.csv` mavjud):

```bash
mkdir -p results/dNN && cp -r out_dNN/tables out_dNN/raw_data results/dNN/
git add -A && git commit -m "..." && git push origin claude/youthful-pascal-sjxqnn
```

`results/STATUS.md` da `d30 / d50 / d100` holatini yangila. **Har bir bo'lakdan
keyin darhol push qil** — push qilinmagan natija konteyner bilan birga yo'qoladi.
Grafiklar (`figures/`) faqat yakuniy birlashtirilgan chiqishdan saqlanadi.

### 4. FAZA C — birlashtirish va yakuniy tahlil

Uchala bo'lak tugagach:

```bash
MERGE=results/d30,results/d50 DIMS=100 RUNS=30 OUT_DIR=out_full python3 cba_shade.py
COMPLEXITY=1 QUICK=1 OUT_DIR=out_cx python3 cba_shade.py   # T0/T1/T2 jadvali
```

Agar `out_d100` allaqachon tugagan bo'lsa, uni qayta ishlatmaslik uchun
`MERGE=results/d30,results/d50,results/d100` va `DIMS=30 RUNS=1` bilan faqat
tahlil bosqichini qayta yurgiz (natijalar `drop_duplicates` bilan saqlanadi).

So'ng `docs/RESULTS.md` yoz. Unda **majburiy** bo'lishi kerak:

1. Protokol: algoritmlar, funksiyalar, o'lchamlar, run soni, byudjet, seed sxemasi.
2. O'rtacha ± std jadvali (12 × 3 instansiya, 7 algoritm).
3. Friedman testi: χ², p, Iman–Davenport F, o'rtacha ranklar.
4. **Post-hoc** (nazorat = CBA-SHADE): z, Holm bilan tuzatilgan p, CD qiymatlari.
   Bu eng muhim jadval — oldingi maqolaning aynan shu yerda muammosi bor edi.
5. Mann–Whitney U + Holm bo'yicha yutuq/tenglik/yutqazish (+/=/−) jadvali.
6. O'lcham bo'yicha ajratilgan ranklar (30 / 50 / 100 alohida).
7. Hisoblash murakkabligi (T0/T1/T2).
8. **Zaif tomonlar bo'limi** — CBA-SHADE yutqazgan yoki teng chiqqan har bir
   instansiya, raqamlari va ehtimoliy sababi bilan.
9. Maqola uchun keyingi qadamlar ro'yxati.

Commit + push. `results/STATUS.md` da `FAZA_C: tugadi` yoz.

### 5. FAZA D — yakunlash

- Routine ni o'chir (`delete_trigger`).
- Foydalanuvchiga qisqa yakuniy xabar yoz: nima qilindi, asosiy raqamlar,
  qaysi fayllarni ochish kerak, qanday zaif tomonlar qoldi.

---

## QAT'IY QOIDALAR

1. **Hech qachon `main` ga push qilma.** Faqat `claude/youthful-pascal-sjxqnn`.
2. **Har bir tugagan bo'lakdan keyin darhol push qil.** Konteyner ephemeral.
3. **Natijalarni bezama.** Nima chiqsa shuni yoz. CBA-SHADE biror funksiyada
   yutqazsa — buni ochiq, raqamlari bilan yoz. Maqsad halol Q1 darajasi,
   chiroyli jadval emas.
4. **Algoritm kodini o'zgartirma.** FAZA A tugagach parametrlar muzlatilgan.
   Natija yoqmasa ham qayta sozlama — bu natijalarni "tanlab olish" bo'ladi.
5. **Bir vaqtda faqat bitta og'ir ish.** Mashinada 4 yadro bor.
6. **Xatolik bo'lsa:** xatoni `results/STATUS.md` ga yoz, push qil, keyingi
   uyg'onishda qayta urin. Bitta bo'lak 3 marta muvaffaqiyatsiz bo'lsa — uni
   tashlab keyingisiga o't va buni `docs/RESULTS.md` da ochiq ayt.
7. **Disk to'lsa:** `out_d*` papkalarining `figures/` qismini o'chir
   (ular yakuniy birlashtirishda qayta yaratiladi).
8. **Foydalanuvchidan hech narsa so'rama.** U uyda. Qaror o'zing qabul qil,
   qabul qilgan qaroringni `results/STATUS.md` ga yozib qo'y.
