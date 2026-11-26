"""Скрипт завантаження відкритих наборів даних для курсової роботи.

Джерела:
1. GitHub Innovation Graph (languages.csv, developers.csv, repositories.csv):
   https://github.com/github/innovationgraph
2. Programming Language DataBase (PLDB, pldb.csv):
   https://pldb.info / https://github.com/Programming-Language-DataBase/pldb
"""

from __future__ import annotations

import pathlib
import sys
import urllib.request

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

DATASETS: dict[str, str] = {
    "languages.csv": "https://raw.githubusercontent.com/github/innovationgraph/main/data/languages.csv",
    "developers.csv": "https://raw.githubusercontent.com/github/innovationgraph/main/data/developers.csv",
    "repositories.csv": "https://raw.githubusercontent.com/github/innovationgraph/main/data/repositories.csv",
    "pldb.csv": "https://pldb.info/pldb.csv",
}


def download_file(url: str, dest: pathlib.Path, timeout: int = 60) -> int:
    """Завантажує файл за URL у локальний шлях dest і повертає розмір у байтах."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Coursework-DataAnalysis/1.0 (Python urllib)"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        content = response.read()
    dest.write_bytes(content)
    return len(content)


def fetch_all(force: bool = False) -> dict[str, pathlib.Path]:
    """Завантажує всі чотири вихідні CSV-таблиці до каталогу data/raw/."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, pathlib.Path] = {}

    for filename, url in DATASETS.items():
        target = RAW_DIR / filename
        if target.exists() and target.stat().st_size > 0 and not force:
            size_mb = target.stat().st_size / (1024 * 1024)
            print(f"[SKIP] {filename} уже існує ({size_mb:.2f} МБ): {target}")
        else:
            print(f"[GET ] Завантаження {filename} з {url} ...")
            nbytes = download_file(url, target)
            if nbytes == 0:
                raise RuntimeError(f"Помилка: файл {filename} порожній (0 байт).")
            size_mb = nbytes / (1024 * 1024)
            print(f"[OK  ] Збережено {filename} ({size_mb:.2f} МБ, {nbytes:,} байт)")
        paths[filename] = target

    return paths


if __name__ == "__main__":
    force_flag = "--force" in sys.argv
    fetch_all(force=force_flag)
    print("Усі набори даних успішно підготовлено.")
