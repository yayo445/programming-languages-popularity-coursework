"""Повний аналітичний конвеєр курсової роботи за темою № 12:
«Дослідження популярності мов програмування».

Виконує:
1. Аудит і завантаження 4 відкритих таблиць (GitHub Innovation Graph + PLDB).
2. Попереднє опрацювання (виправлення коду Намібії 'NA', відокремлення 'EU',
   очищення дублікатів і некоректних значень, оцінка викидів за IQR).
3. Три обґрунтовані об'єднання (merge) пов'язаних таблиць.
4. Інженерію нових аналітичних ознак (6+ похідних змінних).
5. Статистичний аналіз, групування, ранжування, зведені таблиці, часову динаміку
   (2020 Q1 – 2026 Q2), аналіз України ('UA') та кореляційний аналіз (Пірсон / Спірмен).
6. Побудову 8 графіків засобами Matplotlib і Seaborn (300 DPI).
7. Експорт результатів до data/processed/, results/exports/ і tableau/.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIGURES_DIR = BASE_DIR / "results" / "figures"
EXPORTS_DIR = BASE_DIR / "results" / "exports"
TABLEAU_DIR = BASE_DIR / "tableau"

# Словник для відомих провідних ІТ-економік та України (для аналізу і Tableau)
COUNTRY_META: dict[str, tuple[str, str]] = {
    "US": ("США", "Північна Америка"),
    "IN": ("Індія", "Азія"),
    "CN": ("Китай", "Азія"),
    "BR": ("Бразилія", "Латинська Америка"),
    "DE": ("Німеччина", "Європа"),
    "GB": ("Велика Британія", "Європа"),
    "FR": ("Франція", "Європа"),
    "JP": ("Японія", "Азія"),
    "CA": ("Канада", "Північна Америка"),
    "KR": ("Південна Корея", "Азія"),
    "RU": ("Росія", "Східна Європа та Пн. Азія"),
    "ID": ("Індонезія", "Азія"),
    "VN": ("В'єтнам", "Азія"),
    "PL": ("Польща", "Європа"),
    "NL": ("Нідерланди", "Європа"),
    "ES": ("Іспанія", "Європа"),
    "IT": ("Італія", "Європа"),
    "AU": ("Австралія", "Океанія"),
    "TW": ("Тайвань", "Азія"),
    "UA": ("Україна", "Європа"),
    "MX": ("Мексика", "Латинська Америка"),
    "TR": ("Туреччина", "Європа / Зах. Азія"),
    "SE": ("Швеція", "Європа"),
    "CH": ("Швейцарія", "Європа"),
    "IL": ("Ізраїль", "Близький Схід"),
    "SG": ("Сінгапур", "Азія"),
    "AR": ("Аргентина", "Латинська Америка"),
    "CO": ("Колумбія", "Латинська Америка"),
    "NG": ("Нігерія", "Африка"),
    "PK": ("Пакистан", "Азія"),
    "BD": ("Бангладеш", "Азія"),
    "PH": ("Філіппіни", "Азія"),
    "TH": ("Таїланд", "Азія"),
    "CZ": ("Чехія", "Європа"),
    "RO": ("Румунія", "Європа"),
    "PT": ("Португалія", "Європа"),
    "BE": ("Бельгія", "Європа"),
    "AT": ("Австрія", "Європа"),
    "FI": ("Фінляндія", "Європа"),
    "DK": ("Данія", "Європа"),
    "NO": ("Норвегія", "Європа"),
    "IE": ("Ірландія", "Європа"),
    "NZ": ("Нова Зеландія", "Океанія"),
    "ZA": ("ПАР", "Африка"),
    "EG": ("Єгипет", "Африка"),
    "NA": ("Намібія", "Африка"),
}

# Узгодження назв мов GitHub Linguist -> PLDB (у нижньому регістрі)
LANGUAGE_ALIASES: dict[str, str] = {
    "shell": "bash",
    "makefile": "make",
    "assembly": "assembly language",
    "plpgsql": "pl/pgsql",
    "tsx": "typescript",
    "jsx": "javascript",
}

# Утиліти збірки, контейнеризації та командні оболонки в категорії GitHub "programming"
TOOLING_LANGUAGES: set[str] = {
    "Shell",
    "Dockerfile",
    "Makefile",
    "Batchfile",
    "PowerShell",
    "CMake",
    "Nix",
    "HCL",
    "Meson",
    "Starlark",
    "Bicep",
    "M4",
}

PLDB_COLS: list[str] = [
    "id",
    "name",
    "appeared",
    "primaryTag",
    "rank",
    "pldbScore",
    "foundationScore",
    "numberOfUsersEstimate",
    "numberOfJobsEstimate",
    "githubLanguage_repos",
    "wikipedia_dailyPageViews",
    "bookCount",
    "paperCount",
    "isOpenSource",
    "creators",
]


def setup_plot_style() -> None:
    """Налаштовує єдину візуальну тему Matplotlib і Seaborn із підтримкою кирилиці."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "figure.titlesize": 14,
            "figure.titleweight": "bold",
        }
    )


def ensure_dirs() -> None:
    """Створює вихідні директорії, якщо вони відсутні."""
    for d in (PROCESSED_DIR, FIGURES_DIR, EXPORTS_DIR, TABLEAU_DIR):
        d.mkdir(parents=True, exist_ok=True)


def run_pipeline() -> dict[str, Any]:
    """Запускає повний цикл аналізу та зберігає всі артефакти."""
    ensure_dirs()
    setup_plot_style()

    # 1. Аудит проблеми парсингу 'NA' (Намібія) як NaN
    lang_def = pd.read_csv(RAW_DIR / "languages.csv")
    dev_def = pd.read_csv(RAW_DIR / "developers.csv")
    repo_def = pd.read_csv(RAW_DIR / "repositories.csv")

    na_pitfall = {
        "languages_false_na": int(lang_def["iso2_code"].isna().sum()),
        "developers_false_na": int(dev_def["iso2_code"].isna().sum()),
        "repositories_false_na": int(repo_def["iso2_code"].isna().sum()),
    }
    del lang_def, dev_def, repo_def

    # 2. Коректне завантаження без перетворення 'NA' на NaN
    languages_raw = pd.read_csv(
        RAW_DIR / "languages.csv", keep_default_na=False, na_values=[""]
    )
    developers_raw = pd.read_csv(
        RAW_DIR / "developers.csv", keep_default_na=False, na_values=[""]
    )
    repositories_raw = pd.read_csv(
        RAW_DIR / "repositories.csv", keep_default_na=False, na_values=[""]
    )
    pldb_raw = pd.read_csv(RAW_DIR / "pldb.csv", low_memory=False)

    # Первинні характеристики таблиць
    raw_audit = {
        "languages": {
            "rows": int(languages_raw.shape[0]),
            "cols": int(languages_raw.shape[1]),
            "unique_languages": int(languages_raw["language"].nunique()),
            "unique_iso2": int(languages_raw["iso2_code"].nunique()),
            "periods": int(
                languages_raw[["year", "quarter"]].drop_duplicates().shape[0]
            ),
            "min_period": f"{languages_raw['year'].min()}-Q{languages_raw.loc[languages_raw['year']==languages_raw['year'].min(), 'quarter'].min()}",
            "max_period": f"{languages_raw['year'].max()}-Q{languages_raw.loc[languages_raw['year']==languages_raw['year'].max(), 'quarter'].max()}",
            "exact_duplicates": int(languages_raw.duplicated().sum()),
            "missing_total": int(languages_raw.isna().sum().sum()),
            "num_pushers_min": int(languages_raw["num_pushers"].min()),
            "num_pushers_max": int(languages_raw["num_pushers"].max()),
        },
        "developers": {
            "rows": int(developers_raw.shape[0]),
            "cols": int(developers_raw.shape[1]),
            "exact_duplicates": int(developers_raw.duplicated().sum()),
            "missing_total": int(developers_raw.isna().sum().sum()),
        },
        "repositories": {
            "rows": int(repositories_raw.shape[0]),
            "cols": int(repositories_raw.shape[1]),
            "exact_duplicates": int(repositories_raw.duplicated().sum()),
            "missing_total": int(repositories_raw.isna().sum().sum()),
        },
        "pldb": {
            "rows": int(pldb_raw.shape[0]),
            "cols": int(pldb_raw.shape[1]),
            "selected_cols": len(PLDB_COLS),
            "exact_duplicates": int(pldb_raw.duplicated().sum()),
        },
    }

    # 3. Очищення та підготовка даних
    # 3.1 Відокремлення агрегату 'EU' (Євросоюз) від суверенних економік
    eu_counts = {
        "languages_eu_rows": int((languages_raw["iso2_code"] == "EU").sum()),
        "developers_eu_rows": int((developers_raw["iso2_code"] == "EU").sum()),
        "repositories_eu_rows": int((repositories_raw["iso2_code"] == "EU").sum()),
    }

    languages_eu = languages_raw[languages_raw["iso2_code"] == "EU"].copy()
    languages_clean = (
        languages_raw[languages_raw["iso2_code"] != "EU"]
        .drop_duplicates()
        .copy()
    )
    developers_clean = (
        developers_raw[developers_raw["iso2_code"] != "EU"]
        .drop_duplicates()
        .copy()
    )
    repositories_clean = (
        repositories_raw[repositories_raw["iso2_code"] != "EU"]
        .drop_duplicates()
        .copy()
    )

    # Перевірка числових меж та приведення типів
    for col in ("num_pushers", "year", "quarter"):
        languages_clean[col] = pd.to_numeric(languages_clean[col], errors="coerce").astype("int64")
    developers_clean["developers"] = pd.to_numeric(developers_clean["developers"], errors="coerce").astype("int64")
    repositories_clean["repositories"] = pd.to_numeric(repositories_clean["repositories"], errors="coerce").astype("int64")

    # 3.2 Очищення PLDB
    pldb_sub = pldb_raw[PLDB_COLS].copy()
    pldb_nulls_before = {c: int(pldb_sub[c].isna().sum()) for c in PLDB_COLS}

    num_pldb_cols = [
        "appeared",
        "rank",
        "pldbScore",
        "foundationScore",
        "numberOfUsersEstimate",
        "numberOfJobsEstimate",
        "githubLanguage_repos",
        "wikipedia_dailyPageViews",
        "bookCount",
        "paperCount",
    ]
    for col in num_pldb_cols:
        pldb_sub[col] = pd.to_numeric(pldb_sub[col], errors="coerce")

    invalid_appeared_count = int(
        ((pldb_sub["appeared"] < 1940) | (pldb_sub["appeared"] > 2026)).sum()
    )
    pldb_sub.loc[
        (pldb_sub["appeared"] < 1940) | (pldb_sub["appeared"] > 2026),
        "appeared",
    ] = np.nan

    pldb_sub["language_key"] = (
        pldb_sub["name"].astype(str).str.strip().str.lower()
    )
    pldb_sub = pldb_sub.sort_values("rank", ascending=True, na_position="last")
    pldb_dup_keys = int(pldb_sub["language_key"].duplicated().sum())
    pldb_clean = pldb_sub.drop_duplicates(
        subset=["language_key"], keep="first"
    ).copy()

    # Таблиця аудиту очищення
    cleaning_audit_rows = [
        {
            "Таблиця": "languages.csv",
            "Записів до очищення": int(languages_raw.shape[0]),
            "Записів після очищення": int(languages_clean.shape[0]),
            "Хибні NaN ('NA') виправлено": na_pitfall["languages_false_na"],
            "Відокремлено записів 'EU'": eu_counts["languages_eu_rows"],
            "Видалено дублікатів": 0,
            "Аномальних значень (appeared < 1940)": 0,
        },
        {
            "Таблиця": "developers.csv",
            "Записів до очищення": int(developers_raw.shape[0]),
            "Записів після очищення": int(developers_clean.shape[0]),
            "Хибні NaN ('NA') виправлено": na_pitfall["developers_false_na"],
            "Відокремлено записів 'EU'": eu_counts["developers_eu_rows"],
            "Видалено дублікатів": 0,
            "Аномальних значень (appeared < 1940)": 0,
        },
        {
            "Таблиця": "repositories.csv",
            "Записів до очищення": int(repositories_raw.shape[0]),
            "Записів після очищення": int(repositories_clean.shape[0]),
            "Хибні NaN ('NA') виправлено": na_pitfall["repositories_false_na"],
            "Відокремлено записів 'EU'": eu_counts["repositories_eu_rows"],
            "Видалено дублікатів": 0,
            "Аномальних значень (appeared < 1940)": 0,
        },
        {
            "Таблиця": "pldb.csv (вибрані 15 полів)",
            "Записів до очищення": int(pldb_raw.shape[0]),
            "Записів після очищення": int(pldb_clean.shape[0]),
            "Хибні NaN ('NA') виправлено": 0,
            "Відокремлено записів 'EU'": 0,
            "Видалено дублікатів": pldb_dup_keys,
            "Аномальних значень (appeared < 1940)": invalid_appeared_count,
        },
    ]
    cleaning_audit_df = pd.DataFrame(cleaning_audit_rows)
    cleaning_audit_df.to_csv(EXPORTS_DIR / "cleaning_audit.csv", index=False)

    # 4. Три обґрунтовані об'єднання (merge)
    # Merge 1: languages + developers
    df_m1 = languages_clean.merge(
        developers_clean,
        on=["iso2_code", "year", "quarter"],
        how="left",
        validate="many_to_one",
    )
    merge1_matched = int(df_m1["developers"].notna().sum())

    # Merge 2: + repositories
    df_m2 = df_m1.merge(
        repositories_clean,
        on=["iso2_code", "year", "quarter"],
        how="left",
        validate="many_to_one",
    )
    merge2_matched = int(df_m2["repositories"].notna().sum())

    # Merge 3: нормалізація ключів мов + PLDB
    df_m2["language_key"] = (
        df_m2["language"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace(LANGUAGE_ALIASES)
    )
    pldb_merge_cols = [
        "language_key",
        "appeared",
        "primaryTag",
        "rank",
        "pldbScore",
        "foundationScore",
        "numberOfUsersEstimate",
        "numberOfJobsEstimate",
        "githubLanguage_repos",
        "wikipedia_dailyPageViews",
        "bookCount",
        "paperCount",
    ]
    df = df_m2.merge(
        pldb_clean[pldb_merge_cols],
        on="language_key",
        how="left",
        validate="many_to_one",
    )
    merge3_row_match_pct = float(df["appeared"].notna().mean() * 100.0)
    merge3_vol_match_pct = float(
        df.loc[df["appeared"].notna(), "num_pushers"].sum()
        / df["num_pushers"].sum()
        * 100.0
    )
    merge3_unique_matched = int(
        df.loc[df["appeared"].notna(), "language"].nunique()
    )

    # 5. Інженерія нових аналітичних ознак
    quarter_start_month = (df["quarter"] - 1) * 3 + 1
    df["period_date"] = pd.to_datetime(
        dict(year=df["year"], month=quarter_start_month, day=1)
    )
    df["period_label"] = (
        df["year"].astype(str) + "-Q" + df["quarter"].astype(str)
    )
    df["period_year_dt"] = df["period_date"].dt.year
    df["period_quarter_dt"] = df["period_date"].dt.quarter

    df["pushers_per_1000_devs"] = np.where(
        df["developers"] > 0,
        (df["num_pushers"] / df["developers"]) * 1000.0,
        np.nan,
    )
    df["repos_per_dev"] = np.where(
        df["developers"] > 0, df["repositories"] / df["developers"], np.nan
    )

    # Частка мови у межах типу та пари (країна, рік, квартал)
    group_totals = df.groupby(
        ["iso2_code", "year", "quarter", "language_type"]
    )["num_pushers"].transform("sum")
    df["language_share_pct"] = (df["num_pushers"] / group_totals) * 100.0

    # Вік мови та покоління
    df["language_age"] = df["year"] - df["appeared"]
    df["language_era"] = pd.cut(
        df["appeared"],
        bins=[1940, 1989, 2004, 2014, 2026],
        labels=[
            "До 1990 (класичні)",
            "1990–2004 (веб-епоха)",
            "2005–2014 (сучасні)",
            "2015+ (новітні)",
        ],
        include_lowest=True,
    )

    # Розмір ринку розробників економіки
    df["dev_market_tier"] = pd.cut(
        df["developers"],
        bins=[0, 50_000, 500_000, 2_000_000, 1e9],
        labels=[
            "Малі (<50 тис.)",
            "Середні (50–500 тис.)",
            "Великі (0.5–2 млн)",
            "Глобальні лідери (>2 млн)",
        ],
    )

    # Загальнопризначені мови vs інструменти збірки/контейнеризації
    df["is_general_purpose"] = (df["language_type"] == "programming") & (
        ~df["language"].isin(TOOLING_LANGUAGES)
    )

    # Метадані країн
    df["country_name_uk"] = df["iso2_code"].map(
        lambda c: COUNTRY_META[c][0] if c in COUNTRY_META else f"Інші ({c})"
    )
    df["macro_region"] = df["iso2_code"].map(
        lambda c: COUNTRY_META[c][1] if c in COUNTRY_META else "Інші регіони"
    )

    # Збереження очищеного збагаченого набору
    df.to_csv(PROCESSED_DIR / "languages_enriched.csv", index=False)

    # 6. Статистичний аналіз
    # 6.1 Описова статистика + аудит викидів за IQR
    stat_cols = [
        "num_pushers",
        "pushers_per_1000_devs",
        "developers",
        "repositories",
        "language_share_pct",
        "language_age",
    ]
    desc_rows = []
    outlier_audit = {}
    for col in stat_cols:
        s = df[col].dropna()
        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        upper_fence = q3 + 1.5 * iqr
        n_outliers = int((s > upper_fence).sum())
        outlier_audit[col] = {
            "upper_fence": round(upper_fence, 2),
            "outliers_count": n_outliers,
            "outliers_pct": round(n_outliers / len(s) * 100.0, 2),
        }
        desc_rows.append(
            {
                "Показник": col,
                "N": int(s.count()),
                "Середнє": round(float(s.mean()), 2),
                "Медіана": round(float(s.median()), 2),
                "Стд. відх.": round(float(s.std()), 2),
                "Мін.": round(float(s.min()), 2),
                "Q1 (25%)": round(q1, 2),
                "Q3 (75%)": round(q3, 2),
                "Макс.": round(float(s.max()), 2),
                "IQR": round(iqr, 2),
                "Асиметрія": round(float(s.skew()), 2),
            }
        )
    desc_df = pd.DataFrame(desc_rows)
    desc_df.to_csv(EXPORTS_DIR / "descriptive_stats.csv", index=False)

    # 6.2 Глобальний щоквартальний агрегат по мовах (без 'EU')
    global_q = (
        df.groupby(
            ["period_date", "period_label", "year", "quarter", "language", "language_type", "is_general_purpose"],
            as_index=False,
        )
        .agg(
            total_pushers=("num_pushers", "sum"),
            countries_count=("iso2_code", "nunique"),
            mean_intensity=("pushers_per_1000_devs", "mean"),
        )
        .sort_values(["language", "period_date"])
    )
    global_q["rolling_4q_pushers"] = (
        global_q.groupby("language")["total_pushers"]
        .transform(lambda s: s.rolling(4, min_periods=1).mean())
    )
    global_q["yoy_growth_pct"] = (
        global_q.groupby("language")["total_pushers"].pct_change(4) * 100.0
    )
    global_q.to_csv(PROCESSED_DIR / "languages_global_quarterly.csv", index=False)

    # 6.3 Топ-15 глобально у 2026 Q2 (усі programming) + порівняння з Україною (UA)
    q_latest = df[(df["year"] == 2026) & (df["quarter"] == 2)].copy()
    prog_latest = q_latest[q_latest["language_type"] == "programming"].copy()

    global_prog_2026 = (
        prog_latest.groupby(["language", "is_general_purpose"], as_index=False)
        .agg(
            global_pushers=("num_pushers", "sum"),
            countries=("iso2_code", "nunique"),
            mean_intensity=("pushers_per_1000_devs", "mean"),
        )
        .sort_values("global_pushers", ascending=False)
    )
    global_prog_2026["global_rank"] = (
        global_prog_2026["global_pushers"]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    global_prog_2026["global_share_pct"] = (
        global_prog_2026["global_pushers"]
        / global_prog_2026["global_pushers"].sum()
        * 100.0
    )
    top15_global = global_prog_2026.nlargest(15, "global_pushers").copy()
    top15_global.to_csv(EXPORTS_DIR / "top15_global_2026q2.csv", index=False)

    # Україна (UA) у 2026 Q2
    ua_latest = prog_latest[prog_latest["iso2_code"] == "UA"].copy()
    ua_latest["ua_rank"] = (
        ua_latest["num_pushers"].rank(ascending=False, method="min").astype(int)
    )
    ua_latest["ua_share_pct"] = (
        ua_latest["num_pushers"] / ua_latest["num_pushers"].sum() * 100.0
    )
    ua_merged = ua_latest.merge(
        global_prog_2026[["language", "global_rank", "global_share_pct"]],
        on="language",
        how="left",
    )
    ua_merged["rank_diff_vs_global"] = (
        ua_merged["global_rank"] - ua_merged["ua_rank"]
    )
    top12_ua = ua_merged.nlargest(12, "num_pushers")[
        [
            "ua_rank",
            "language",
            "num_pushers",
            "ua_share_pct",
            "pushers_per_1000_devs",
            "global_rank",
            "rank_diff_vs_global",
        ]
    ].copy()
    top12_ua.to_csv(EXPORTS_DIR / "top12_ukraine_2026q2.csv", index=False)

    # Динаміка розробників та репозиторіїв України (2020 Q1 vs 2026 Q2)
    ua_dev_ts = developers_clean[developers_clean["iso2_code"] == "UA"].sort_values(
        ["year", "quarter"]
    )
    ua_repo_ts = repositories_clean[repositories_clean["iso2_code"] == "UA"].sort_values(
        ["year", "quarter"]
    )

    # 6.4 Темпи зростання мов програмування (2020 Q1 -> 2026 Q2)
    q_start = global_q[
        (global_q["year"] == 2020)
        & (global_q["quarter"] == 1)
        & (global_q["language_type"] == "programming")
    ][["language", "is_general_purpose", "total_pushers"]].rename(
        columns={"total_pushers": "pushers_2020q1"}
    )
    q_end = global_q[
        (global_q["year"] == 2026)
        & (global_q["quarter"] == 2)
        & (global_q["language_type"] == "programming")
    ][["language", "total_pushers"]].rename(
        columns={"total_pushers": "pushers_2026q2"}
    )
    growth_df = q_start.merge(q_end, on="language", how="inner")
    growth_df["growth_multiple"] = (
        growth_df["pushers_2026q2"] / growth_df["pushers_2020q1"]
    )
    growth_df["growth_pct"] = (growth_df["growth_multiple"] - 1.0) * 100.0
    # Аутсайдери та лідери серед значущих мов (>=10 000 розробників у 2020 Q1)
    growth_major = growth_df[growth_df["pushers_2020q1"] >= 10_000].sort_values(
        "growth_multiple", ascending=False
    )
    growth_major.to_csv(EXPORTS_DIR / "growth_2020_2026.csv", index=False)

    # 6.5 Групування за language_type та language_era у 2026 Q2
    by_type_2026 = (
        q_latest.groupby("language_type", as_index=False)
        .agg(
            records=("num_pushers", "count"),
            unique_languages=("language", "nunique"),
            total_pushers=("num_pushers", "sum"),
            mean_pushers=("num_pushers", "mean"),
            median_pushers=("num_pushers", "median"),
            mean_intensity=("pushers_per_1000_devs", "mean"),
        )
        .sort_values("total_pushers", ascending=False)
    )
    by_type_2026["share_pct"] = (
        by_type_2026["total_pushers"] / by_type_2026["total_pushers"].sum() * 100.0
    )
    by_type_2026.to_csv(EXPORTS_DIR / "agg_by_language_type_2026q2.csv", index=False)

    by_era_2026 = (
        prog_latest.groupby("language_era", observed=True, as_index=False)
        .agg(
            unique_languages=("language", "nunique"),
            total_pushers=("num_pushers", "sum"),
            mean_intensity=("pushers_per_1000_devs", "mean"),
            median_intensity=("pushers_per_1000_devs", "median"),
        )
    )
    by_era_2026["share_pct"] = (
        by_era_2026["total_pushers"] / by_era_2026["total_pushers"].sum() * 100.0
    )
    by_era_2026.to_csv(EXPORTS_DIR / "agg_by_language_era_2026q2.csv", index=False)

    by_tier_2026 = (
        prog_latest.groupby("dev_market_tier", observed=True, as_index=False)
        .agg(
            economies=("iso2_code", "nunique"),
            unique_languages=("language", "nunique"),
            total_pushers=("num_pushers", "sum"),
            mean_intensity=("pushers_per_1000_devs", "mean"),
            median_intensity=("pushers_per_1000_devs", "median"),
        )
    )
    by_tier_2026.to_csv(EXPORTS_DIR / "agg_by_market_tier_2026q2.csv", index=False)

    # 6.6 Зведені таблиці (pivot_table та crosstab)
    top12_gp_names = (
        global_prog_2026[global_prog_2026["is_general_purpose"]]
        .nlargest(12, "global_pushers")["language"]
        .tolist()
    )
    q2_yearly = global_q[
        (global_q["quarter"] == 2) & (global_q["language"].isin(top12_gp_names))
    ]
    pivot_years = q2_yearly.pivot_table(
        index="language",
        columns="year",
        values="total_pushers",
        aggfunc="sum",
    )
    pivot_years = pivot_years.loc[top12_gp_names]
    pivot_years["Зростання 2020→2026 (разів)"] = (
        pivot_years[2026] / pivot_years[2020]
    ).round(2)
    pivot_years.to_csv(EXPORTS_DIR / "pivot_langs_by_year.csv")

    pivot_era_tier = prog_latest.pivot_table(
        index="language_era",
        columns="dev_market_tier",
        values="pushers_per_1000_devs",
        aggfunc="mean",
        observed=True,
    ).round(2)
    pivot_era_tier.to_csv(EXPORTS_DIR / "pivot_era_by_tier.csv")

    crosstab_type_era = pd.crosstab(
        df.drop_duplicates("language")["language_type"],
        df.drop_duplicates("language")["language_era"],
    )
    crosstab_type_era.to_csv(EXPORTS_DIR / "crosstab_type_era.csv")

    # 6.7 Рівень окремих мов: агрегація GitHub 2026 Q2 + PLDB для кореляції та Scatter
    lang_summary = (
        q_latest.groupby(["language", "language_type", "language_key", "is_general_purpose"], as_index=False)
        .agg(
            global_pushers_2026q2=("num_pushers", "sum"),
            countries_2026q2=("iso2_code", "nunique"),
            mean_intensity_2026q2=("pushers_per_1000_devs", "mean"),
        )
        .merge(
            pldb_clean[
                [
                    "language_key",
                    "appeared",
                    "rank",
                    "pldbScore",
                    "numberOfUsersEstimate",
                    "numberOfJobsEstimate",
                    "githubLanguage_repos",
                    "wikipedia_dailyPageViews",
                    "bookCount",
                ]
            ],
            on="language_key",
            how="inner",
        )
        .merge(
            growth_df[["language", "pushers_2020q1", "growth_multiple"]],
            on="language",
            how="left",
        )
    )
    lang_summary["language_age_2026"] = 2026 - lang_summary["appeared"]
    lang_summary["language_era"] = pd.cut(
        lang_summary["appeared"],
        bins=[1940, 1989, 2004, 2014, 2026],
        labels=[
            "До 1990 (класичні)",
            "1990–2004 (веб-епоха)",
            "2005–2014 (сучасні)",
            "2015+ (новітні)",
        ],
        include_lowest=True,
    )
    lang_summary.to_csv(PROCESSED_DIR / "languages_pldb_summary.csv", index=False)

    corr_cols = [
        "global_pushers_2026q2",
        "countries_2026q2",
        "pldbScore",
        "numberOfUsersEstimate",
        "numberOfJobsEstimate",
        "wikipedia_dailyPageViews",
        "bookCount",
        "language_age_2026",
    ]
    corr_df_clean = lang_summary[corr_cols].dropna()
    corr_spearman = corr_df_clean.corr(method="spearman").round(3)
    corr_pearson = corr_df_clean.corr(method="pearson").round(3)
    corr_spearman.to_csv(EXPORTS_DIR / "correlations_spearman.csv")
    corr_pearson.to_csv(EXPORTS_DIR / "correlations_pearson.csv")

    # Парні коефіцієнти та p-values для ключових пар
    corr_pairs = {}
    pairs_to_test = [
        ("global_pushers_2026q2", "countries_2026q2"),
        ("global_pushers_2026q2", "pldbScore"),
        ("global_pushers_2026q2", "numberOfUsersEstimate"),
        ("global_pushers_2026q2", "numberOfJobsEstimate"),
        ("global_pushers_2026q2", "bookCount"),
        ("global_pushers_2026q2", "wikipedia_dailyPageViews"),
        ("global_pushers_2026q2", "language_age_2026"),
    ]
    for a, b in pairs_to_test:
        sub = lang_summary[[a, b]].dropna()
        r_s, p_s = stats.spearmanr(sub[a], sub[b])
        r_p, p_p = stats.pearsonr(sub[a], sub[b])
        corr_pairs[f"{a}__vs__{b}"] = {
            "n": int(len(sub)),
            "spearman_r": round(float(r_s), 4),
            "spearman_p": float(p_s),
            "pearson_r": round(float(r_p), 4),
            "pearson_p": float(p_p),
        }

    # 7. Візуалізація (8 графіків у results/figures/)
    # Fig 01: Топ-12 мов у світі vs в Україні (2026 Q2)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2))
    g12 = global_prog_2026.nlargest(12, "global_pushers").iloc[::-1]
    colors_g = ["#1f77b4" if gp else "#7f8c8d" for gp in g12["is_general_purpose"]]
    bars_g = axes[0].barh(g12["language"], g12["global_pushers"] / 1e6, color=colors_g, edgecolor="none")
    axes[0].set_title("а) Світ: Топ-12 мов програмування (2026 Q2)")
    axes[0].set_xlabel("Активні розробники (pushers), млн")
    axes[0].set_xlim(0, g12["global_pushers"].max() / 1e6 * 1.18)
    for bar, val in zip(bars_g, g12["global_pushers"]):
        axes[0].text(
            bar.get_width() + 0.08,
            bar.get_y() + bar.get_height() / 2,
            f"{val/1e6:.2f} млн",
            va="center",
            fontsize=8.8,
        )
    # Легенда типів
    from matplotlib.patches import Patch
    axes[0].legend(
        handles=[
            Patch(facecolor="#1f77b4", label="Загальнопризначені мови"),
            Patch(facecolor="#7f8c8d", label="Інструменти (Shell / Dockerfile / Makefile)"),
        ],
        loc="lower right",
        frameon=True,
    )

    u12 = top12_ua.iloc[::-1]
    colors_u = [
        "#2ca02c" if d >= 3 else ("#d62728" if d <= -3 else "#1f77b4")
        for d in u12["rank_diff_vs_global"]
    ]
    bars_u = axes[1].barh(u12["language"], u12["num_pushers"] / 1e3, color=colors_u, edgecolor="none")
    axes[1].set_title("б) Україна (UA): Топ-12 мов програмування (2026 Q2)")
    axes[1].set_xlabel("Активні розробники (pushers), тис.")
    axes[1].set_xlim(0, u12["num_pushers"].max() / 1e3 * 1.24)
    for bar, (_, row) in zip(bars_u, u12.iterrows()):
        diff = int(row["rank_diff_vs_global"])
        diff_txt = f" (+{diff})" if diff > 0 else (f" ({diff})" if diff < 0 else " (=)")
        axes[1].text(
            bar.get_width() + 0.6,
            bar.get_y() + bar.get_height() / 2,
            f"{row['num_pushers']/1e3:.1f} тис.{diff_txt}",
            va="center",
            fontsize=8.8,
        )
    axes[1].legend(
        handles=[
            Patch(facecolor="#1f77b4", label="Ранг близький до світового"),
            Patch(facecolor="#2ca02c", label="Вищий попит в Україні (+3 і більше)"),
        ],
        loc="lower right",
        frameon=True,
    )
    fig.suptitle("Порівняння провідних мов програмування у світі та Україні (GitHub, 2026 Q2)", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig01_top_languages_global_vs_ua.png", bbox_inches="tight")
    plt.close(fig)

    # Fig 02: Квартальна динаміка Топ-8 загальнопризначених мов (2020 Q1 - 2026 Q2) з rolling(4)
    top8_gp = [
        "JavaScript",
        "Python",
        "TypeScript",
        "Java",
        "C++",
        "C",
        "PHP",
        "C#",
    ]
    ts_top8 = global_q[global_q["language"].isin(top8_gp)].copy()
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    palette_8 = sns.color_palette("tab10", n_colors=len(top8_gp))
    for idx, lang in enumerate(top8_gp):
        sub = ts_top8[ts_top8["language"] == lang].sort_values("period_date")
        ax.plot(
            sub["period_date"],
            sub["total_pushers"] / 1e6,
            color=palette_8[idx],
            alpha=0.35,
            linewidth=1.3,
        )
        ax.plot(
            sub["period_date"],
            sub["rolling_4q_pushers"] / 1e6,
            color=palette_8[idx],
            linewidth=2.5,
            label=f"{lang} ({sub['total_pushers'].iloc[-1]/1e6:.2f} млн у 2026 Q2)",
        )
    ax.set_title("Квартальна динаміка Топ-8 загальнопризначених мов програмування (2020–2026 рр.)")
    ax.set_xlabel("Період (напівпрозора лінія — квартальне значення, суцільна — 4-кв. ковзне середнє)")
    ax.set_ylabel("Активні розробники у світі, млн")
    ax.legend(loc="upper left", ncol=2, frameon=True)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig02_quarterly_dynamics_line.png", bbox_inches="tight")
    plt.close(fig)

    # Fig 03: Лідери та зрілі мови за темпом зростання (2020 Q1 -> 2026 Q2)
    focus_growth_langs = [
        "TypeScript",
        "Rust",
        "Dart",
        "Python",
        "Kotlin",
        "Swift",
        "JavaScript",
        "Go",
        "C++",
        "C#",
        "C",
        "PHP",
        "Java",
        "Ruby",
    ]
    g14 = (
        growth_df[growth_df["language"].isin(focus_growth_langs)]
        .sort_values("growth_multiple", ascending=True)
        .copy()
    )
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    bar_colors = [
        "#2ca02c" if m >= 5.0 else ("#1f77b4" if m >= 2.5 else "#e67e22")
        for m in g14["growth_multiple"]
    ]
    bars = ax.barh(g14["language"], g14["growth_multiple"], color=bar_colors)
    ax.axvline(1.0, color="#555555", linestyle="--", linewidth=1, label="Базовий рівень (1.0x)")
    ax.set_title("Темпи зростання активних розробників за період 2020 Q1 – 2026 Q2 (кратність)")
    ax.set_xlabel("Кратність зростання кількості розробників (2026 Q2 / 2020 Q1), разів")
    ax.set_xlim(0, g14["growth_multiple"].max() * 1.18)
    for bar, (_, r) in zip(bars, g14.iterrows()):
        ax.text(
            bar.get_width() + 0.15,
            bar.get_y() + bar.get_height() / 2,
            f"×{r['growth_multiple']:.2f} (+{r['growth_pct']:.0f}%)",
            va="center",
            fontsize=9,
        )
    ax.legend(
        handles=[
            Patch(facecolor="#2ca02c", label="Стрімке зростання (≥5×)"),
            Patch(facecolor="#1f77b4", label="Помірне зростання (2.5×–5×)"),
            Patch(facecolor="#e67e22", label="Зрілі / консервативні мови (<2.5×)"),
        ],
        loc="lower right",
        frameon=True,
    )
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig03_growth_leaders_bar.png", bbox_inches="tight")
    plt.close(fig)

    # Fig 04: Гістограма розподілу (log10(num_pushers) та pushers_per_1000_devs)
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4))
    log_p = np.log10(df["num_pushers"])
    sns.histplot(log_p, bins=35, kde=True, color="#1f77b4", ax=axes[0])
    axes[0].axvline(log_p.mean(), color="#d62728", linestyle="--", linewidth=1.8, label=f"Середнє: {log_p.mean():.2f} (~{10**log_p.mean():,.0f})")
    axes[0].axvline(log_p.median(), color="#2ca02c", linestyle="-.", linewidth=1.8, label=f"Медіана: {log_p.median():.2f} (~{10**log_p.median():,.0f})")
    axes[0].set_title("а) Логарифмічний розподіл активних розробників")
    axes[0].set_xlabel("log10(num_pushers) — десятковий логарифм кількості")
    axes[0].set_ylabel("Кількість спостережень")
    axes[0].legend(frameon=True)

    prog_int = df.loc[df["language_type"] == "programming", "pushers_per_1000_devs"].dropna()
    prog_int_clipped = prog_int[prog_int <= 40]
    sns.histplot(prog_int_clipped, bins=32, kde=True, color="#9467bd", ax=axes[1])
    axes[1].axvline(prog_int.mean(), color="#d62728", linestyle="--", linewidth=1.8, label=f"Середнє (усі): {prog_int.mean():.2f}")
    axes[1].axvline(prog_int.median(), color="#2ca02c", linestyle="-.", linewidth=1.8, label=f"Медіана (усі): {prog_int.median():.2f}")
    axes[1].set_title("б) Розподіл інтенсивності (programming, ≤ 40)")
    axes[1].set_xlabel("Розробників мови на 1000 активних розробників країни")
    axes[1].set_ylabel("Кількість спостережень")
    axes[1].legend(frameon=True)

    fig.suptitle("Аналіз розподілів ключових числових показників (2020–2026 рр.)", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig04_distribution_histogram.png", bbox_inches="tight")
    plt.close(fig)

    # Fig 05: Boxplot за language_type та language_era
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.8))
    df_plot = df.copy()
    df_plot["log10_pushers"] = np.log10(df_plot["num_pushers"])
    type_order = ["programming", "markup", "data", "prose"]
    sns.boxplot(
        data=df_plot,
        x="language_type",
        y="log10_pushers",
        order=type_order,
        hue="language_type",
        palette="Set2",
        legend=False,
        ax=axes[0],
        fliersize=2.2,
    )
    axes[0].set_title("а) Розподіл активності за типом мови (log10)")
    axes[0].set_xlabel("Тип мови (класифікація GitHub Linguist)")
    axes[0].set_ylabel("log10(num_pushers)")

    prog_era_df = prog_latest.dropna(subset=["language_era", "pushers_per_1000_devs"]).copy()
    prog_era_df = prog_era_df[prog_era_df["pushers_per_1000_devs"] > 0]
    sns.boxplot(
        data=prog_era_df,
        x="language_era",
        y="pushers_per_1000_devs",
        hue="language_era",
        palette="viridis",
        legend=False,
        ax=axes[1],
        fliersize=2.6,
    )
    axes[1].set_yscale("log")
    axes[1].set_title("б) Інтенсивність за поколінням мов (2026 Q2, лог-шкала)")
    axes[1].set_xlabel("Покоління появи мови програмування")
    axes[1].set_ylabel("Розробників на 1000 активних розробників (лог-шкала)")
    axes[1].tick_params(axis="x", rotation=14)

    fig.suptitle("Порівняння розподілів і виявлення викидів за типами та поколіннями мов", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig05_era_and_type_boxplot.png", bbox_inches="tight")
    plt.close(fig)

    # Fig 06: Кореляційна теплова карта Спірмена
    uk_labels = {
        "global_pushers_2026q2": "GitHub pushers (2026 Q2)",
        "countries_2026q2": "Кількість країн (2026 Q2)",
        "pldbScore": "Рейтинг PLDB Score",
        "numberOfUsersEstimate": "Оцінка користувачів (PLDB)",
        "numberOfJobsEstimate": "Оцінка вакансій (PLDB)",
        "wikipedia_dailyPageViews": "Перегляди Wikipedia/день",
        "bookCount": "Кількість книг (PLDB)",
        "language_age_2026": "Вік мови (років)",
    }
    corr_renamed = corr_spearman.rename(index=uk_labels, columns=uk_labels)
    fig, ax = plt.subplots(figsize=(10.6, 8.4))
    sns.heatmap(
        corr_renamed,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.6,
        square=True,
        ax=ax,
        cbar_kws={"label": "Коефіцієнт рангової кореляції Спірмена (ρ)", "shrink": 0.86},
    )
    ax.set_title(
        f"Кореляційна матриця Спірмена між показниками GitHub та PLDB\n(n = {len(corr_df_clean)} мов, 2026 Q2)",
        pad=16,
    )
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig06_correlation_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    # Fig 07: Теплова карта частки Топ-10 мов (%) у провідних 10 економіках включно з Україною
    selected_isos = ["US", "IN", "CN", "DE", "GB", "BR", "JP", "PL", "CA", "UA"]
    top10_gp = [
        "JavaScript",
        "Python",
        "TypeScript",
        "Java",
        "C++",
        "C",
        "PHP",
        "C#",
        "Kotlin",
        "Go",
    ]
    sub_hm = prog_latest[
        prog_latest["iso2_code"].isin(selected_isos)
        & prog_latest["language"].isin(top10_gp)
    ].copy()
    sub_hm["country_lbl"] = sub_hm["iso2_code"].map(
        lambda c: f"{COUNTRY_META[c][0]} ({c})"
    )
    ordered_country_lbls = [f"{COUNTRY_META[c][0]} ({c})" for c in selected_isos]
    pv_country = sub_hm.pivot_table(
        index="language",
        columns="country_lbl",
        values="language_share_pct",
        aggfunc="sum",
    ).reindex(index=top10_gp, columns=ordered_country_lbls)

    fig, ax = plt.subplots(figsize=(12.2, 6.4))
    sns.heatmap(
        pv_country,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Частка у категорії programming країни (%)"},
    )
    ax.set_title("Географічний профіль: частка Топ-10 мов програмування (%) у провідних економіках та Україні (2026 Q2)")
    ax.set_xlabel("Країна (код ISO-2)")
    ax.set_ylabel("Мова програмування")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig07_country_language_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    # Fig 08: Діаграма розсіювання (Scatter): GitHub pushers vs PLDB Jobs Estimate
    scatter_df = lang_summary[
        (lang_summary["language_type"] == "programming")
        & (lang_summary["numberOfJobsEstimate"] > 0)
        & (lang_summary["global_pushers_2026q2"] > 0)
    ].dropna(subset=["language_era"]).copy()

    fig, ax = plt.subplots(figsize=(11.8, 6.8))
    sns.scatterplot(
        data=scatter_df,
        x="global_pushers_2026q2",
        y="numberOfJobsEstimate",
        hue="language_era",
        size="countries_2026q2",
        sizes=(45, 360),
        alpha=0.82,
        palette="viridis",
        ax=ax,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(60, 2.2e7)
    ax.set_title(
        f"Зв'язок активності на GitHub (2026 Q2) та оцінки вакансій у PLDB (n = {len(scatter_df)} мов)"
    )
    ax.set_xlabel("Активні розробники на GitHub у 2026 Q2 (лог-шкала)")
    ax.set_ylabel("Оцінка кількості вакансій у PLDB (лог-шкала)")

    # Підписи для ключових мов зі зсувами для уникнення перекриття
    label_offsets = {
        "JavaScript": (-34, 11),
        "Python": (7, -11),
        "TypeScript": (7, -5),
        "Java": (8, 5),
        "C++": (8, -9),
        "C": (-18, 8),
        "C#": (-25, -13),
        "PHP": (10, -2),
        "Go": (-24, 6),
        "Rust": (8, -6),
        "Kotlin": (8, 3),
        "Swift": (-32, -10),
        "Ruby": (-34, -2),
        "Dart": (8, -2),
        "R": (-16, 7),
        "Scala": (7, -5),
        "Shell": (8, -4),
        "PowerShell": (-68, 5),
    }
    for _, row in scatter_df.iterrows():
        lang = row["language"]
        if lang in label_offsets:
            dx, dy = label_offsets[lang]
            ax.annotate(
                lang,
                (row["global_pushers_2026q2"], row["numberOfJobsEstimate"]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=8.5,
                fontweight="bold",
                color="#222222",
            )
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, title="Покоління та країни")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig08_scatter_github_vs_pldb.png", bbox_inches="tight")
    plt.close(fig)

    # 8. Підготовка файлів для Tableau Public
    top35_langs = (
        df.groupby("language")["num_pushers"].sum().nlargest(35).index.tolist()
    )
    tableau_main = df[df["language"].isin(top35_langs)][
        [
            "period_date",
            "period_label",
            "year",
            "quarter",
            "iso2_code",
            "country_name_uk",
            "macro_region",
            "dev_market_tier",
            "language",
            "language_type",
            "is_general_purpose",
            "language_era",
            "appeared",
            "language_age",
            "num_pushers",
            "developers",
            "repositories",
            "pushers_per_1000_devs",
            "language_share_pct",
            "repos_per_dev",
        ]
    ].copy()
    tableau_main["period_date"] = tableau_main["period_date"].dt.strftime("%Y-%m-%d")
    tableau_main.to_csv(TABLEAU_DIR / "tableau_languages_main.csv", index=False)

    tableau_summary = lang_summary.copy()
    tableau_summary.to_csv(TABLEAU_DIR / "tableau_language_summary.csv", index=False)

    with pd.ExcelWriter(
        TABLEAU_DIR / "tableau_coursework_data.xlsx", engine="openpyxl"
    ) as writer:
        tableau_main.to_excel(writer, sheet_name="Quarterly_By_Country", index=False)
        tableau_summary.to_excel(writer, sheet_name="Language_Summary_PLDB", index=False)

    # 9. Зведені метрики JSON (виключно обчислені значення)
    ua_2020q1_devs = int(
        ua_dev_ts.loc[(ua_dev_ts["year"] == 2020) & (ua_dev_ts["quarter"] == 1), "developers"].iloc[0]
    )
    ua_2026q2_devs = int(
        ua_dev_ts.loc[(ua_dev_ts["year"] == 2026) & (ua_dev_ts["quarter"] == 2), "developers"].iloc[0]
    )
    ua_2020q1_repos = int(
        ua_repo_ts.loc[(ua_repo_ts["year"] == 2020) & (ua_repo_ts["quarter"] == 1), "repositories"].iloc[0]
    )
    ua_2026q2_repos = int(
        ua_repo_ts.loc[(ua_repo_ts["year"] == 2026) & (ua_repo_ts["quarter"] == 2), "repositories"].iloc[0]
    )
    global_2026q2_devs = int(
        developers_clean.loc[
            (developers_clean["year"] == 2026) & (developers_clean["quarter"] == 2),
            "developers",
        ].sum()
    )
    global_2020q1_devs = int(
        developers_clean.loc[
            (developers_clean["year"] == 2020) & (developers_clean["quarter"] == 1),
            "developers",
        ].sum()
    )

    summary_metrics = {
        "na_pitfall": na_pitfall,
        "eu_counts": eu_counts,
        "raw_audit": raw_audit,
        "pldb_cleaning": {
            "nulls_before": pldb_nulls_before,
            "duplicate_keys_dropped": pldb_dup_keys,
            "invalid_appeared_nullified": invalid_appeared_count,
            "clean_rows": int(pldb_clean.shape[0]),
        },
        "clean_shapes": {
            "languages_clean": list(languages_clean.shape),
            "languages_eu": list(languages_eu.shape),
            "developers_clean": list(developers_clean.shape),
            "repositories_clean": list(repositories_clean.shape),
            "enriched_final": list(df.shape),
        },
        "merges": {
            "merge1_match_pct": round(merge1_matched / len(languages_clean) * 100.0, 2),
            "merge2_match_pct": round(merge2_matched / len(languages_clean) * 100.0, 2),
            "merge3_row_match_pct": round(merge3_row_match_pct, 2),
            "merge3_volume_match_pct": round(merge3_vol_match_pct, 2),
            "merge3_unique_matched_langs": merge3_unique_matched,
        },
        "outlier_audit": outlier_audit,
        "global_totals": {
            "devs_2020q1": global_2020q1_devs,
            "devs_2026q2": global_2026q2_devs,
            "devs_growth_x": round(global_2026q2_devs / global_2020q1_devs, 2),
            "ua_devs_2020q1": ua_2020q1_devs,
            "ua_devs_2026q2": ua_2026q2_devs,
            "ua_devs_growth_x": round(ua_2026q2_devs / ua_2020q1_devs, 2),
            "ua_repos_2020q1": ua_2020q1_repos,
            "ua_repos_2026q2": ua_2026q2_repos,
            "ua_repos_growth_x": round(ua_2026q2_repos / ua_2020q1_repos, 2),
            "ua_rows_total": int((df["iso2_code"] == "UA").sum()),
            "ua_unique_langs": int(df.loc[df["iso2_code"] == "UA", "language"].nunique()),
        },
        "corr_pairs": corr_pairs,
        "tableau_rows": {
            "tableau_languages_main": int(len(tableau_main)),
            "tableau_language_summary": int(len(tableau_summary)),
        },
    }
    (EXPORTS_DIR / "summary_metrics.json").write_text(
        json.dumps(summary_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary_metrics


if __name__ == "__main__":
    metrics = run_pipeline()
    print("Конвеєр аналізу успішно завершено.")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
