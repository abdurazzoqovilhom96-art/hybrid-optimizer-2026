# Eksperimentni Windows kompyuterda boshlash

Bu fayl bitta narsa uchun: eksperimentni **ILHOM** kompyuterida (i7-13700F,
16 yadro / 24 tred, 32 GB) noldan ishga tushirish.

---

## Muhim: kod `main` branchda EMAS

`main` branchda faqat **3 ta fayl** bor: `README.md`, `.gitignore` va asl
`hybrid 2026.py`. `START.py`, `run_windows.bat`, `temoa/` paketi va testlar —
hammasi **`claude/adoring-archimedes-942q36`** branchida.

Shuning uchun oddiy `git pull` hech narsa keltirmaydi va ishga tushiradigan
fayl ham topilmaydi. To'g'ri buyruq — branchga **o'tish**.

---

## A varianti — repozitoriya kompyuteringizda allaqachon bor

`cmd.exe` ni oching (Win+R → `cmd` → Enter) va repozitoriya papkasiga o'ting:

```bat
cd /d C:\yo'l\hybrid-optimizer-2026

git fetch origin
git checkout claude/adoring-archimedes-942q36
git pull

run_windows.bat --jobs 20
```

## B varianti — noldan klon (ishonchliroq)

```bat
cd /d C:\

git clone -b claude/adoring-archimedes-942q36 https://github.com/abdurazzoqovilhom96-art/hybrid-optimizer-2026.git

cd hybrid-optimizer-2026
run_windows.bat --jobs 20
```

---

## Nima bo'lishi kerak

`run_windows.bat` to'rt bosqichni bajaradi va har birida nima bo'layotganini
yozadi:

| Bosqich | Nima qiladi | Vaqti |
|---|---|---|
| 1/4 environment | Python va 7 ta kutubxonani tekshiradi, yo'qini **o'zi o'rnatadi** | ~1 daqiqa |
| 2/4 tests | **72 ta test**. Bittasi yiqilsa eksperiment boshlanmaydi | ~6 daqiqa |
| 3/4 experiment | CEC'2017, D=10, 29 funksiya, 51 yurish, 100 000 FES × 10 algoritm | ~2.5 soat |
| 4/4 analysis | Jadval, Friedman/Holm, CD diagramma | ~1 daqiqa |

So'ngida `start? [y/N]` deb so'raydi — **`y`** bosing va Enter. So'ramasligi
uchun `--yes` qo'shing:

```bat
run_windows.bat --jobs 20 --yes
```

Ctrl+C bilan istalgan vaqtda to'xtatsa bo'ladi. Qaytadan ishga tushirsangiz
**o'sha joydan davom etadi** va natija uzilmagan yurish bilan bit-identical
bo'ladi — buning testi bor.

---

## Agar ishlamasa

Ekranda chiqqan xabarni menga yuboring. Eng ko'p uchraydigan uchtasi:

| Xabar | Sababi | Yechim |
|---|---|---|
| `'run_windows.bat' is not recognized` yoki `The system cannot find the file` | noto'g'ri papka yoki noto'g'ri branch | yuqoridagi B variantini bajaring |
| `[X] "python" was not found on PATH` | Python o'rnatilmagan | https://www.python.org/downloads/ — o'rnatishda **"Add python.exe to PATH"** ni belgilang |
| `[X] START.py is not in this folder` | branch noto'g'ri | `git checkout claude/adoring-archimedes-942q36` |

Oyna ochilib darhol yopilib qolsa: `cmd.exe` dan ishga tushiring, shunda xato
xabari ekranda qoladi.

---

## Natija qayerda

| Fayl | Nima |
|---|---|
| `results_cec2017\raw\results.csv` | xom natija (har yurish bitta qator) |
| `results_cec2017\manifest.json` | protokol, urug'lar, mashina ma'lumoti |
| `reports\` | jadvallar va diagrammalar |

Tugagach `results_cec2017` papkasini menga yuboring — tahlilni men qilaman.
