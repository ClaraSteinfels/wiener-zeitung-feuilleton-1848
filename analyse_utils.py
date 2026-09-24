"""Gemeinsame Hilfsfunktionen für die Feuilleton-Analyse."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RESULT_DIR = PROJECT_DIR / "ergebnisse"
TABLE_DIR = RESULT_DIR / "tabellen"
FIGURE_DIR = RESULT_DIR / "abbildungen"

CODING_FILE = DATA_DIR / "wiener_zeitung_feuilleton_1848_kodierung.csv"
ISSUE_FILE = DATA_DIR / "wiener_zeitung_feuilleton_1848_ausgabentabelle.csv"

TOPIC_ORDER = [
    "Kunst und Literatur",
    "Wissenschaft und Technik",
    "Gesellschaft und Zeitbetrachtung",
    "Industrie und Wirtschaft",
    "Politik und Recht",
    "unklar",
]

FORM_ORDER = [
    "Essay/Abhandlung",
    "Kritik/Rezension",
    "Bericht/Mitteilung/Protokoll",
    "Rede/Vortrag",
    "Literarischer Primärtext",
    "Brief/Korrespondenz",
    "Sonstiges/unklar",
]

SECONDARY_TOPIC_ORDER = [
    "Kunst und Literatur",
    "Wissenschaft und Technik",
    "Gesellschaft und Zeitbetrachtung",
    "Industrie und Wirtschaft",
    "Politik und Recht",
]

TOPIC_ALIASES = {
    "Gesellschaft/Zeitbetrachtung": "Gesellschaft und Zeitbetrachtung",
    "Politik/Recht": "Politik und Recht",
    "Wissenschaft/Technik": "Wissenschaft und Technik",
    "Industrie/Wirtschaft": "Industrie und Wirtschaft",
    "Kunst/Literatur": "Kunst und Literatur",
}

MONTH_LABELS = {
    1: "Januar",
    2: "Februar",
    3: "März",
    4: "April",
    5: "Mai",
    6: "Juni",
}


def ensure_output_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def _read_semicolon_csv(path: Path) -> pd.DataFrame:
    """Liest die aus Excel exportierten UTF-8-BOM/Strichpunkt-CSVs ein."""
    frame = pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype="string")
    frame.columns = frame.columns.str.strip()
    for column in frame.columns:
        if pd.api.types.is_string_dtype(frame[column]):
            frame[column] = frame[column].str.strip()
    return frame


def add_time_columns(frame: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    result = frame.copy()
    result[date_column] = pd.to_datetime(result[date_column], errors="raise")
    result["month_number"] = result[date_column].dt.month
    result["month"] = result["month_number"].map(MONTH_LABELS)
    result["revolution_period"] = result[date_column].ge(pd.Timestamp("1848-03-13")).map(
        {False: "Vor dem 13. März", True: "Ab dem 13. März"}
    )
    return result


def load_coding() -> pd.DataFrame:
    coding = _read_semicolon_csv(CODING_FILE)
    for column in ("primary_topic", "secondary_topic"):
        if column in coding.columns:
            coding[column] = coding[column].replace(TOPIC_ALIASES)
    coding = add_time_columns(coding)
    for column in ("page_start", "page_end"):
        coding[column] = pd.to_numeric(coding[column], errors="coerce").astype("Int64")
    return coding.sort_values(["date", "contribution_id"]).reset_index(drop=True)


def load_issues() -> pd.DataFrame:
    issues = _read_semicolon_csv(ISSUE_FILE)
    issues = add_time_columns(issues)
    issues["issue_number"] = pd.to_numeric(issues["issue_number"], errors="raise").astype(int)
    issues["feuilleton_pages"] = pd.to_numeric(issues["feuilleton_pages"], errors="coerce")
    issues["digitized_bool"] = issues["digitized"].eq("ja")
    issues["has_feuilleton_bool"] = issues["has_feuilleton"].eq("ja")
    return issues.sort_values("date").reset_index(drop=True)


WORK_FIELDS = [
    "primary_topic",
    "secondary_topic",
    "text_form",
    "author_normalized",
    "author_identity_status",
    "series_id",
    "series_type",
]

def make_work_table(coding: pd.DataFrame) -> pd.DataFrame:
    """Eine Zeile pro Werk; zeitliche Zuordnung zum ersten Segment."""
    ordered = coding.sort_values(
        ["date", "contribution_id"]
    ).copy()

    if ordered["work_id"].isna().any():
        raise ValueError(
            "Mindestens einem Segment fehlt die work_id."
        )

    conflicts = (
        ordered.groupby("work_id")[WORK_FIELDS]
        .nunique(dropna=False)
    )
    bad = conflicts.index[
        conflicts.gt(1).any(axis=1)
    ].tolist()

    if bad:
        raise ValueError(
            "Widersprüchliche Werkmetadaten: "
            + ", ".join(bad)
        )

    # Tatsächliche erste Zeile statt erster nichtleerer Wert je Spalte.
    works = ordered.drop_duplicates(
        "work_id", keep="first"
    ).copy()

    metadata = ordered.groupby(
        "work_id", as_index=False
    ).agg(
        last_date=("date", "max"),
        segment_count=("contribution_id", "nunique"),
        issue_count=("anno_issue_id", "nunique"),
        author_signatures=(
            "author_as_printed",
            lambda s: " | ".join(
                sorted(set(s.dropna().astype(str)) - {""})
            ),
        ),
    )

    works = works.merge(
        metadata,
        on="work_id",
        validate="one_to_one",
    )

    return (
        add_time_columns(works)
        .sort_values(["date", "work_id"])
        .reset_index(drop=True)
    )

def frequency_table(
    frame: pd.DataFrame,
    column: str,
    order: list[str] | None = None,
) -> pd.DataFrame:
    counts = frame[column].fillna("<leer>").value_counts(dropna=False)
    if order is not None:
        extra = [value for value in counts.index if value not in order]
        counts = counts.reindex(order + extra, fill_value=0)
    result = counts.rename_axis(column).reset_index(name="n")
    result["anteil_prozent"] = result["n"].div(len(frame)).mul(100).round(1)
    return result


def crosstab_counts(
    frame: pd.DataFrame,
    index: str,
    columns: str,
    index_order: list[str] | None = None,
    column_order: list[str] | None = None,
) -> pd.DataFrame:
    table = pd.crosstab(frame[index], frame[columns], dropna=False)
    if index_order is not None:
        table = table.reindex(index_order, fill_value=0)
    if column_order is not None:
        table = table.reindex(columns=column_order, fill_value=0)
    return table.reset_index()


def save_table(frame: pd.DataFrame, filename: str) -> Path:
    path = TABLE_DIR / filename
    export = frame.copy()
    for column in export.select_dtypes(include=["datetime", "datetimetz"]).columns:
        export[column] = export[column].dt.strftime("%Y-%m-%d")
    export.to_csv(path, sep=";", encoding="utf-8-sig", index=False)
    return path

