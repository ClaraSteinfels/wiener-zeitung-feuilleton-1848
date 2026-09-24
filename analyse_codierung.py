"""Deskriptive Statistiken für die Kodierungstabelle."""

from __future__ import annotations

import pandas as pd

from analyse_utils import (
    FORM_ORDER,
    MONTH_LABELS,
    SECONDARY_TOPIC_ORDER,
    TOPIC_ORDER,
    crosstab_counts,
    ensure_output_dirs,
    frequency_table,
    load_coding,
    make_work_table,
    save_table,
)


NO_SECONDARY = "kein Zweitthema"


def detailed_combinations(frame: pd.DataFrame, dimensions: list[str], level: str) -> pd.DataFrame:
    """Beobachtete Kombinationen; n ist der Zähler sämtlicher Prozentwerte.

    Nenner beziehen sich auf alle übergebenen Segmente bzw. Werke. Fehlende
    Zweitthemen bilden eine Gruppe, aber keine Sachkategorie. Ihre Anteile an
    tatsächlichen Zweitthemen bleiben leer. Vertauschte Themenrollen bleiben
    getrennt. Es werden keine unbeobachteten Nullkombinationen ausgegeben.
    """
    data = frame[dimensions].copy()
    if "secondary_topic" in dimensions:
        data["secondary_topic"] = data["secondary_topic"].astype("string").str.strip().replace("", pd.NA)
        secondary_total = int(data["secondary_topic"].notna().sum())
        data["secondary_topic"] = data["secondary_topic"].fillna(NO_SECONDARY)
    result = data.groupby(dimensions, observed=True, dropna=False).size().reset_index(name="n")
    result.insert(0, "ebene", level)

    def add_share(label, denominator, valid=None):
        denominator = pd.Series(denominator, index=result.index, dtype="Float64")
        if valid is not None:
            denominator = denominator.where(valid)
        result["bezugszahl_" + label] = denominator.astype("Int64")
        result["anteil_" + label + "_prozent"] = (
            result["n"].div(denominator.where(denominator.gt(0))).mul(100).round(1)
        )

    add_share("gesamt", len(frame))
    for field, label in [("primary_topic", "hauptthema"), ("text_form", "textform")]:
        if field in dimensions:
            denominator = result.groupby(field, observed=True, dropna=False)["n"].transform("sum")
            add_share(label, denominator)
    if "secondary_topic" in dimensions:
        valid = result["secondary_topic"].ne(NO_SECONDARY)
        add_share("mit_zweitthema", secondary_total, valid)
        denominator = result.groupby("secondary_topic", observed=True, dropna=False)["n"].transform("sum")
        add_share("zweitthema", denominator, valid)
        if "primary_topic" in dimensions and "text_form" in dimensions:
            denominator = result.groupby(
                ["primary_topic", "secondary_topic"], observed=True, dropna=False
            )["n"].transform("sum")
            # Auch für 'kein Zweitthema': Formenverteilung dieser Gruppe.
            add_share("haupt_und_zweitthema_kombination", denominator)
    orders = {"primary_topic": TOPIC_ORDER, "text_form": FORM_ORDER,
              "secondary_topic": SECONDARY_TOPIC_ORDER + [NO_SECONDARY]}
    for field in dimensions:
        unknown = set(result[field].dropna()) - set(orders[field])
        if unknown or result[field].isna().any():
            raise ValueError(f"Ungültige Kategorien in {field}: {unknown}")
        result[field] = pd.Categorical(result[field], categories=orders[field], ordered=True)
    return result.sort_values(dimensions).reset_index(drop=True)


def category_role_table(works: pd.DataFrame) -> pd.DataFrame:
    """Kategorienanteile nur unter Werken mit Zweitthema."""
    paired = works.loc[works["secondary_topic"].fillna("").str.strip().ne("")]
    result = pd.DataFrame({"kategorie": TOPIC_ORDER})
    for field, label in [("primary_topic", "hauptthema"), ("secondary_topic", "zweitthema")]:
        result[label + "_n"] = result["kategorie"].map(paired[field].value_counts()).fillna(0).astype(int)
        result[label + "_anteil_prozent"] = (
            result[label + "_n"].div(len(paired)).mul(100).round(1)
            if len(paired) else float("nan")
        )
    result["bezugszahl_werke_mit_zweitthema"] = len(paired)
    return result


def run() -> dict[str, pd.DataFrame]:
    ensure_output_dirs()
    segments = load_coding()
    works = make_work_table(segments)

    overview = pd.DataFrame(
        [
            ("Textsegmente", len(segments), "contribution_id"),
            ("Zusammengehörige Werke", len(works), "work_id"),
            ("Ausgaben mit mindestens einem Segment", segments["anno_issue_id"].nunique(), "anno_issue_id"),
            ("Fortsetzungswerke", int((works["segment_count"] > 1).sum()), "work_id"),
            ("Segmente in Artikelreihen", int(segments["series_id"].notna().sum()), "contribution_id"),
            ("Eindeutige Artikelreihen", int(segments["series_id"].nunique()), "series_id"),
            ("Final kodierte Segmente", int(segments["coding_status"].eq("final").sum()), "contribution_id"),
        ],
        columns=["kennzahl", "wert", "einheit"],
    )

    topic_segment = frequency_table(segments, "primary_topic", TOPIC_ORDER)
    topic_work = frequency_table(works, "primary_topic", TOPIC_ORDER)
    form_segment = frequency_table(segments, "text_form", FORM_ORDER)
    form_work = frequency_table(works, "text_form", FORM_ORDER)
    secondary = frequency_table(segments, "secondary_topic")
    combination_details = segments.loc[
        segments["secondary_topic"].fillna("").str.strip().ne(""),
        ["contribution_id", "work_id", "date", "anno_issue_id", "page_start",
         "page_end", "title", "primary_topic", "secondary_topic", "text_form", "coding_grounds"],
    ].sort_values(["primary_topic", "secondary_topic", "date", "contribution_id"])
    series = frequency_table(segments, "series_type", ["keine", "Fortsetzungsbeitrag", "Artikelreihe"])
    author_status = frequency_table(segments, "author_identity_status")

    topic_month_n = crosstab_counts(
        segments, "month", "primary_topic", list(MONTH_LABELS.values()), TOPIC_ORDER
    )
    topic_month_pct = topic_month_n.set_index("month")
    topic_month_pct = topic_month_pct.div(topic_month_pct.sum(axis=1), axis=0).mul(100).round(1).reset_index()

    topic_period_n = crosstab_counts(
        segments,
        "revolution_period",
        "primary_topic",
        ["Vor dem 13. März", "Ab dem 13. März"],
        TOPIC_ORDER,
    )
    topic_period_pct = topic_period_n.set_index("revolution_period")
    topic_period_pct = topic_period_pct.div(topic_period_pct.sum(axis=1), axis=0).mul(100).round(1).reset_index()

    form_topic = crosstab_counts(
        segments, "text_form", "primary_topic", FORM_ORDER, TOPIC_ORDER
    )

    monthly_segments = (
        segments.groupby(["month_number", "month"], observed=True)
        .agg(
            segmente=("contribution_id", "nunique"),
            ausgaben_mit_segment=("anno_issue_id", "nunique"),
        )
        .reset_index()
    )
    monthly_works = (
        works.groupby(["month_number", "month"], observed=True)
        .agg(werke=("work_id", "nunique"))
        .reset_index()
    )
    monthly = monthly_segments.merge(
        monthly_works, on=["month_number", "month"], how="outer", validate="one_to_one"
    ).sort_values("month_number")

    work_spans = works.loc[works["segment_count"].gt(1), [
        "work_id", "title", "date", "last_date", "segment_count", "issue_count",
        "primary_topic", "text_form", "series_id", "series_type"
    ]].rename(columns={"date": "first_date"}).sort_values(
        ["segment_count", "work_id"], ascending=[False, True]
    )

    tables = {
        "00_korpus_uebersicht.csv": overview,
        "01_hauptthemen_segmentebene.csv": topic_segment,
        "02_hauptthemen_werkebene.csv": topic_work,
        "03_textformen_segmentebene.csv": form_segment,
        "04_textformen_werkebene.csv": form_work,
        "05_hauptthemen_nach_monat_absolut.csv": topic_month_n,
        "06_hauptthemen_nach_monat_prozent.csv": topic_month_pct,
        "07_hauptthemen_vor_nach_13_maerz_absolut.csv": topic_period_n,
        "08_hauptthemen_vor_nach_13_maerz_prozent.csv": topic_period_pct,
        "09b_textform_x_hauptthema_segmentebene.csv": form_topic,
        "10_zweitthemen_segmentebene.csv": secondary,
        "10f_themenkombinationen_belegliste.csv": combination_details,
        "10g_kategorienrollen_werke_mit_zweitthema.csv": category_role_table(works),
        "11_serienstruktur_segmentebene.csv": series,
        "12_autoridentitaet_segmentebene.csv": author_status,
        "13_monatliche_korpusstruktur.csv": monthly,
        "14_fortsetzungswerke.csv": work_spans,
    }
    for level, frame in [("segmentebene", segments), ("werkebene", works)]:
        for number, label, dimensions in [
            ("15", "hauptthema_textform", ["primary_topic", "text_form"]),
            ("16", "hauptthema_zweitthema", ["primary_topic", "secondary_topic"]),
            ("17", "hauptthema_zweitthema_textform", ["primary_topic", "secondary_topic", "text_form"]),
        ]:
            tables[f"{number}_{label}_{level}.csv"] = detailed_combinations(frame, dimensions, level)
    for filename, table in tables.items():
        save_table(table, filename)
    return {"segments": segments, "works": works, **tables}


if __name__ == "__main__":
    run()
    print("Kodierungsanalyse abgeschlossen.")