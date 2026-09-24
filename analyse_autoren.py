"""Deskriptive Analyse der Autor- und Signaturangaben.

Die Hauptauswertung erfolgt auf Werkebene, damit Fortsetzungen die Präsenz
einzelner Zuschreibungen nicht künstlich erhöhen. Segmentwerte werden als
ergänzende Sichtbarkeitsperspektive ausgegeben.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyse_utils import (
    FIGURE_DIR,
    FORM_ORDER,
    MONTH_LABELS,
    TOPIC_ORDER,
    ensure_output_dirs,
    frequency_table,
    load_coding,
    make_work_table,
    save_table,
)


IDENTIFIED_STATUSES = ["gesichert", "wahrscheinlich", "unsicher"]
STATUS_ORDER = ["gesichert", "wahrscheinlich", "unsicher", "nicht identifiziert", "ohne Autor"]
LABEL_STATUS_ORDER = ["gesichert", "wahrscheinlich", "unsicher", "nicht identifiziert"]
STATUS_COLORS = {
    "gesichert": "#376795",
    "wahrscheinlich": "#55A3A3",
    "unsicher": "#D39A36",
    "nicht identifiziert": "#B95756",
    "ohne Autor": "#A5A5A5",
}
TOPIC_COLORS = {
    "Kunst und Literatur": "#376795",
    "Wissenschaft und Technik": "#55A3A3",
    "Gesellschaft und Zeitbetrachtung": "#D39A36",
    "Industrie und Wirtschaft": "#8D6A9F",
    "Politik und Recht": "#B95756",
    "unklar": "#9A9A9A",
}


def _join_unique(series: pd.Series) -> str:
    values = sorted({str(value) for value in series.dropna() if str(value).strip()})
    return " | ".join(values)


def _save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURE_DIR / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _create_figures(works: pd.DataFrame, author_labels: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    status = frequency_table(works, "author_identity_status", STATUS_ORDER)
    status = status.loc[status["n"].gt(0)].iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.4, 4.7))
    bars = ax.barh(
        status["author_identity_status"], status["n"],
        color=[STATUS_COLORS.get(value, "#777777") for value in status["author_identity_status"]],
    )
    for bar, n, pct in zip(bars, status["n"], status["anteil_prozent"]):
        ax.text(bar.get_width() + 0.7, bar.get_y() + bar.get_height()/2, f"{int(n)} ({pct:.1f} %)", va="center")
    ax.set_title(f"Status der Autoridentifikation auf Werkebene (n = {len(works)})", loc="left", fontweight="bold")
    ax.set_xlabel("Anzahl der Werke")
    ax.set_ylabel("")
    ax.set_xlim(0, status["n"].max() * 1.28)
    ax.grid(axis="x", alpha=0.22)
    fig.tight_layout()
    _save_figure(fig, "08_autorstatus_werkebene")

    author_status = pd.crosstab(
        author_labels["author_normalized"], author_labels["author_identity_status"]
    ).reindex(columns=LABEL_STATUS_ORDER, fill_value=0)
    totals = author_status.sum(axis=1).sort_values(ascending=False)
    top_names = totals.loc[totals.ge(2)].index
    author_status = author_status.loc[top_names].iloc[::-1]
    fig, ax = plt.subplots(figsize=(9.0, 6.2))
    left = np.zeros(len(author_status))
    for status_name in LABEL_STATUS_ORDER:
        values = author_status[status_name].to_numpy()
        ax.barh(
            author_status.index, values, left=left,
            label=status_name, color=STATUS_COLORS[status_name],
        )
        left += values
    for y, total in enumerate(left):
        ax.text(total + 0.25, y, f"{int(total)}", va="center")
    ax.set_title("Mehrfach vertretene normalisierte Autorenbezeichnungen", loc="left", fontweight="bold")
    ax.set_xlabel("Anzahl der Werke")
    ax.set_ylabel("")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    _save_figure(fig, "09_haeufigste_autoren_werkebene")

    monthly = pd.crosstab(works["month"], works["author_identity_status"])
    monthly = monthly.reindex(index=list(MONTH_LABELS.values()), columns=STATUS_ORDER, fill_value=0)
    monthly_pct = monthly.div(monthly.sum(axis=1), axis=0).mul(100)
    fig, ax = plt.subplots(figsize=(9.2, 5.5))
    x = np.arange(len(monthly_pct))
    bottom = np.zeros(len(monthly_pct))
    for status_name in STATUS_ORDER:
        values = monthly_pct[status_name].to_numpy()
        ax.bar(x, values, bottom=bottom, width=0.72, label=status_name, color=STATUS_COLORS[status_name])
        bottom += values
    ax.set_title("Autoridentifikationsstatus nach Monat auf Werkebene", loc="left", fontweight="bold")
    ax.set_ylabel("Anteil der erstmals erschienenen Werke (%)")
    ax.set_xticks(x, monthly_pct.index)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    _save_figure(fig, "10_autorstatus_nach_monat_werkebene")


def run() -> dict[str, pd.DataFrame]:
    ensure_output_dirs()
    segments = load_coding()
    works = make_work_table(segments)

    # author_normalized ist die gemeinsame Vergleichsbezeichnung: ein
    # identifizierter Name oder, wenn die Person unbekannt bleibt, ein stabiles
    # Alias/eine Chiffre wie "Quidam". Der Identifikationsstatus bleibt separat.
    author_label_mask = works["author_normalized"].notna()
    author_labels = works.loc[author_label_mask].copy()
    identified = author_labels.loc[author_labels["author_identity_status"].isin(IDENTIFIED_STATUSES)].copy()
    strong_mask = works["author_identity_status"].isin(["gesichert", "wahrscheinlich"])

    author_counts = (
        author_labels.groupby("author_normalized", as_index=False)
        .agg(
            werke=("work_id", "nunique"),
            segmente=("segment_count", "sum"),
            erstes_erscheinen=("date", "min"),
            letztes_erscheinen=("last_date", "max"),
            identifikationsstatus=("author_identity_status", _join_unique),
            gedruckte_signaturen=(
                "author_signatures",
                lambda s: " | ".join(
                    sorted({
                        signature.strip()
                        for value in s.dropna()
                        for signature in str(value).split(" | ")
                        if signature.strip()
                    })
                ),
            ),            hauptthemen=("primary_topic", _join_unique),
            textformen=("text_form", _join_unique),
        )
        .sort_values(["werke", "segmente", "author_normalized"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    author_counts["anteil_an_allen_werken_prozent"] = author_counts["werke"].div(len(works)).mul(100).round(1)
    author_counts["anteil_an_werken_mit_autorenbezeichnung_prozent"] = (
        author_counts["werke"].div(len(author_labels)).mul(100).round(1)
    )

    top_three_works = int(author_counts.head(3)["werke"].sum())
    overview = pd.DataFrame(
        [
            ("Werke insgesamt", len(works), "work_id"),
            ("Werke ohne Autor", int(works["author_identity_status"].eq("ohne Autor").sum()), "work_id"),
            ("Werke mit nicht identifizierter Signatur", int(works["author_identity_status"].eq("nicht identifiziert").sum()), "work_id"),
            ("Werke mit normalisierter Autorenbezeichnung", len(author_labels), "identifizierter Name oder stabiles Alias"),
            ("Davon nicht identifizierte Alias/Chiffren", int(author_labels["author_identity_status"].eq("nicht identifiziert").sum()), "z. B. Quidam"),
            ("Werke mit identifizierender Zuschreibung", len(identified), "gesichert, wahrscheinlich oder unsicher"),
            ("Davon gesichert oder wahrscheinlich", int(strong_mask.sum()), "work_id"),
            ("Verschiedene normalisierte Autorenbezeichnungen", int(author_labels["author_normalized"].nunique()), "Namen, institutionelle Angaben und Aliasse"),
            ("Werke der drei häufigsten Autorenbezeichnungen", top_three_works, f"{top_three_works / len(author_labels) * 100:.1f} % der Werke mit Autorenbezeichnung"),
        ],
        columns=["kennzahl", "wert", "hinweis"],
    )

    unidentified = works.loc[works["author_identity_status"].eq("nicht identifiziert")].copy()
    unidentified["signatur"] = unidentified["author_as_printed"].fillna(unidentified["author_normalized"]).fillna("[nicht lesbar]")
    unidentified_counts = (
        unidentified.groupby("signatur", as_index=False)
        .agg(
            werke=("work_id", "nunique"),
            segmente=("segment_count", "sum"),
            hauptthemen=("primary_topic", _join_unique),
            titel=("title", _join_unique),
        )
        .sort_values(["werke", "signatur"], ascending=[False, True])
    )

    signature_map = (
        segments.loc[
            segments["author_normalized"].notna()
            & segments["author_as_printed"].notna()
        ]
        .groupby(
            [
                "author_normalized",
                "author_as_printed",
                "author_identity_status",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            werke=("work_id", "nunique"),
            segmente=("contribution_id", "nunique"),
        )
        .sort_values(
            ["author_normalized", "werke", "author_as_printed"],
            ascending=[True, False, True],
        )
    )

    author_topic = pd.crosstab(author_labels["author_normalized"], author_labels["primary_topic"])
    author_topic = author_topic.reindex(columns=TOPIC_ORDER, fill_value=0)
    author_topic["werke_gesamt"] = author_topic.sum(axis=1)
    author_topic = author_topic.sort_values(["werke_gesamt", "author_normalized"], ascending=[False, True]).reset_index()

    author_form = pd.crosstab(author_labels["author_normalized"], author_labels["text_form"])
    author_form = author_form.reindex(columns=FORM_ORDER, fill_value=0)
    author_form["werke_gesamt"] = author_form.sum(axis=1)
    author_form = author_form.sort_values(["werke_gesamt", "author_normalized"], ascending=[False, True]).reset_index()

    monthly_status = pd.crosstab(works["month"], works["author_identity_status"])
    monthly_status = monthly_status.reindex(index=list(MONTH_LABELS.values()), columns=STATUS_ORDER, fill_value=0)
    monthly_status["werke_gesamt"] = monthly_status.sum(axis=1)
    monthly_status = monthly_status.reset_index()

      # Autorenverteilungen auf Werkebene mit mehreren Bezugsgrößen.
    distribution_works = works.copy()
    distribution_works["author_normalized"] = (
        distribution_works["author_normalized"]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )

    total_works = len(distribution_works)

    topic_totals = (
        distribution_works.groupby("primary_topic", observed=True)
        .size()
    )
    form_totals = (
        distribution_works.groupby("text_form", observed=True)
        .size()
    )

    def author_distribution_by(group_columns):
        result = (
            distribution_works
            .groupby(
                group_columns + ["author_normalized"],
                dropna=False,
                observed=True,
            )
            .size()
            .reset_index(name="werke")
        )

        # Anteil an sämtlichen Werken des Korpus.
        result["werke_gesamt"] = total_works
        result["anteil_an_allen_werken_prozent"] = (
            result["werke"]
            .div(total_works if total_works else float("nan"))
            .mul(100)
            .round(1)
        )

        # Anteil an allen Werken des jeweiligen Hauptthemas.
        if "primary_topic" in group_columns:
            result["werke_im_hauptthema"] = (
                result["primary_topic"].map(topic_totals)
            )
            result["anteil_am_hauptthema_prozent"] = (
                result["werke"]
                .div(result["werke_im_hauptthema"])
                .mul(100)
                .round(1)
            )

        # Anteil an allen Werken der jeweiligen Textform.
        if "text_form" in group_columns:
            result["werke_in_textform"] = (
                result["text_form"].map(form_totals)
            )
            result["anteil_an_textform_prozent"] = (
                result["werke"]
                .div(result["werke_in_textform"])
                .mul(100)
                .round(1)
            )

        # Anteil an allen Werken derselben Hauptthema-Textform-Kombination.
        if {"primary_topic", "text_form"}.issubset(group_columns):
            result["werke_in_kombination"] = (
                result.groupby(
                    ["primary_topic", "text_form"],
                    dropna=False,
                    observed=True,
                )["werke"]
                .transform("sum")
            )
            result["anteil_an_kombination_prozent"] = (
                result["werke"]
                .div(result["werke_in_kombination"])
                .mul(100)
                .round(1)
            )

        result["author_normalized"] = (
            result["author_normalized"].fillna("ohne Autor")
        )

        for column, order in [
            ("primary_topic", TOPIC_ORDER),
            ("text_form", FORM_ORDER),
        ]:
            if column in group_columns:
                result[column] = pd.Categorical(
                    result[column],
                    categories=order,
                    ordered=True,
                )

        return result.sort_values(
            group_columns + ["werke", "author_normalized"],
            ascending=[True] * len(group_columns) + [False, True],
        ).reset_index(drop=True)

    authors_by_topic = author_distribution_by(
        ["primary_topic"]
    )
    authors_by_form = author_distribution_by(
        ["text_form"]
    )
    authors_by_topic_and_form = author_distribution_by(
        ["primary_topic", "text_form"]
    )

    tables = {
        "30_autoren_uebersicht_werkebene.csv": overview,
        "31_autorstatus_werkebene.csv": frequency_table(works, "author_identity_status", STATUS_ORDER),
        "32_autorstatus_segmentebene.csv": frequency_table(segments, "author_identity_status", STATUS_ORDER),
        "33_normalisierte_autoren_werkebene.csv": author_counts,
        "34_nicht_identifizierte_signaturen_werkebene.csv": unidentified_counts,
        "35_signaturvarianten_und_zuschreibungen.csv": signature_map,
        "36_autoren_x_hauptthema_werkebene.csv": author_topic,
        "37_autoren_x_textform_werkebene.csv": author_form,
        "38_autorstatus_nach_monat_werkebene.csv": monthly_status,
        "39_autoren_nach_hauptthema_werkebene.csv": authors_by_topic,
        "40_autoren_nach_textform_werkebene.csv": authors_by_form,
        "41_autoren_nach_hauptthema_und_textform_werkebene.csv": authors_by_topic_and_form,
    }
    
    for filename, table in tables.items():
        save_table(table, filename)

    _create_figures(works, author_labels)
    return {
        "segments": segments,
        "works": works,
        "author_labels": author_labels,
        "identified": identified,
        **tables,
    }


if __name__ == "__main__":
    run()
    print("Autorenanalyse abgeschlossen.")
