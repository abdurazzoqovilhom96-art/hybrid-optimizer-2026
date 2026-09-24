"""opfunu ning ildiz funksiyalarini vektorlashtirilgan variantlar bilan almashtirish.

opfunu CEC funksiyalarini rasmiy siljitish/aylantirish ma'lumotlari bilan
to'g'ri hisoblaydi, lekin `katsuura_func` Python sikli bilan yozilgan va
har bir ichida 32 ta skalyar uchun `np.sum` chaqiradi. U butun eksperiment
narxini belgilaydi:

    katsuura_func      2235 us  ->  33 us   (66x, D=20)

`schaffer_f7_func` ham siklga ega, lekin u atigi 23 us va uni
vektorlashtirish yig'indi tartibi tufayli 2.6e-12 nisbiy chetlanish
beradi. Tejash kichik, chetlanish esa bepul emas - shuning uchun u
o'zgartirilmaydi.

Bu yerdagi variant aynan bir xil matematikani numpy bilan bajaradi
(farq 1.4e-15, ya'ni qo'shmaqiymat yaxlitlash darajasida).
Almashtirish `apply()` chaqirilganda amalga oshadi va `tests/test_fastops.py`
har bir funksiyani asl variant bilan tasodifiy nuqtalarda solishtiradi.
"""
import numpy as np


def katsuura_func(x):
    x = np.asarray(x, dtype=float).ravel()
    ndim = x.size
    p = 2.0 ** np.arange(1, 33)
    px = p * x[:, None]
    temp = (np.abs(px - np.round(px)) / p).sum(axis=1)
    idx = np.arange(1, ndim + 1)
    return (np.prod((1.0 + idx * temp) ** (10.0 / ndim ** 1.2)) - 1.0) * 10.0 / ndim ** 2


REPLACEMENTS = {"katsuura_func": katsuura_func}
_originals = {}


def apply():
    """Almashtirishni o'rnatadi. Bir necha marta chaqirilishi xavfsiz."""
    from opfunu.utils import operator
    for name, fn in REPLACEMENTS.items():
        if name not in _originals:
            _originals[name] = getattr(operator, name)
        setattr(operator, name, fn)
    return _originals


def originals():
    """Asl variantlarni qaytaradi (tekshiruv uchun)."""
    if not _originals:
        from opfunu.utils import operator
        for name in REPLACEMENTS:
            _originals[name] = getattr(operator, name)
    return dict(_originals)
