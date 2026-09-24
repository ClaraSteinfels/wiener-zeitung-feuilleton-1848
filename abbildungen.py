"""Erzeugt publikationsfähige Abbildungen aus den deskriptiven Tabellen."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyse_utils import FIGURE_DIR, FORM_ORDER, TOPIC_ORDER, ensure_output_dirs


TOPIC_COLORS = {
    "Kunst und Literatur": "#376795",
    "Wissenschaft und Technik": "#55A3A3",
    "Gesellschaft und Zeitbetrachtung": "#D39A36",
    "Industrie und Wirtschaft": "#8D6A9F",
    "Politik und Recht": "#B95756",
    "unklar": "#9A9A9A",
}

BAR_BLUE = "#376795"
BAR_GOLD = "#D39A36"


def _read_table(filename: str) -> pd.DataFrame:
    return pd.read_csv(
        FIGURE_DIR.parent / "tabellen" / filename,
        sep=";",
        encoding="utf-8-sig",
    )


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 130,
        }
    )


def _save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(
        FIGURE_DIR / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        FIGURE_DIR / f"{stem}.svg",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def _horizontal_frequency(
    table: pd.DataFrame,
    category: str,
    title: str,
    stem: str,
    color: str,
) -> None:
    shown = table.loc[table["n"].gt(0)].copy().iloc[::-1]

    fig, ax = plt.subplots(figsize=(8.2, 4.8))

    bars = ax.barh(
        shown[category],
        shown["n"],
        color=color,
    )

    for bar, n, pct in zip(
        bars,
        shown["n"],
        shown["anteil_prozent"],
    ):
        ax.text(
            bar.get_width() + max(shown["n"]) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{int(n)} ({pct:.1f} %)",
            va="center",
        )

    ax.set_title(
        title,
        loc="left",
        fontweight="bold",
    )
    ax.set_xlabel("Anzahl")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.22)
    ax.set_xlim(0, max(shown["n"]) * 1.28)

    fig.tight_layout()
    _save(fig, stem)

def _topic_levels_side_by_side(
    topic_segment: pd.DataFrame,
    topic_work: pd.DataFrame,
    stem: str,
) -> None:
    """Zeigt die Hauptthemenverteilung auf Segment- und Werkebene
    in zwei nebeneinanderstehenden Teilabbildungen.
    """

    # Einheitliche Themenreihenfolge herstellen.
    segment = (
        topic_segment
        .set_index("primary_topic")
        .reindex(TOPIC_ORDER)
        .reset_index()
    )

    work = (
        topic_work
        .set_index("primary_topic")
        .reindex(TOPIC_ORDER)
        .reset_index()
    )

    # Fehlende Werte gegebenenfalls als 0 behandeln.
    segment["n"] = segment["n"].fillna(0)
    segment["anteil_prozent"] = segment["anteil_prozent"].fillna(0)

    work["n"] = work["n"].fillna(0)
    work["anteil_prozent"] = work["anteil_prozent"].fillna(0)

    # Prozentwerte verwenden, damit beide Ebenen direkt
    # miteinander vergleichbar sind.
    max_pct = max(
        segment["anteil_prozent"].max(),
        work["anteil_prozent"].max(),
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.2, 5.2),
        sharex=True,
        sharey=True,
    )

    datasets = [
        (
            axes[0],
            segment,
            "Segmentebene",
            int(segment["n"].sum()),
        ),
        (
            axes[1],
            work,
            "Werkebene",
            int(work["n"].sum()),
        ),
    ]

    y = np.arange(len(TOPIC_ORDER))

    for ax, table, title, total_n in datasets:

        values = table["anteil_prozent"].to_numpy(dtype=float)
        counts = table["n"].to_numpy(dtype=int)

        colors = [
            TOPIC_COLORS[topic]
            for topic in table["primary_topic"]
        ]

        bars = ax.barh(
            y,
            values,
            color=colors,
            height=0.68,
        )

        # Prozentwert + absolute Häufigkeit am Balkenende.
        for bar, pct, n in zip(
            bars,
            values,
            counts,
        ):
            ax.text(
                bar.get_width() + max_pct * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f} %\n(n = {n})",
                va="center",
                ha="left",
                fontsize=8.5,
                linespacing=1.0,
            )

        ax.set_title(
            f"{title} (n = {total_n})",
            loc="left",
            fontweight="bold",
        )

        ax.set_xlabel("Anteil (%)")

        ax.set_xlim(
            0,
            max_pct * 1.35,
        )

        ax.grid(
            axis="x",
            alpha=0.22,
        )

    # Themenbeschriftungen nur links.
    axes[0].set_yticks(
        y,
        TOPIC_ORDER,
    )

    axes[0].set_ylabel("")
    axes[1].set_ylabel("")

    # TOPIC_ORDER von oben nach unten darstellen.
    axes[0].invert_yaxis()

    fig.suptitle(
        "Hauptthemenverteilung auf Segment- und Werkebene",
        x=0.07,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.94),
    )

    _save(
        fig,
        stem,
    )

def run() -> None:
    ensure_output_dirs()
    _style()

    # ------------------------------------------------------------
    # Tabellen einlesen
    # ------------------------------------------------------------

    topic_segment = _read_table(
        "01_hauptthemen_segmentebene.csv"
    )

    topic_work = _read_table(
        "02_hauptthemen_werkebene.csv"
    )


    _topic_levels_side_by_side(
        topic_segment,
        topic_work,
        "02b_hauptthemen_segment_und_werkebene",
    )

    form_segment = _read_table(
        "03_textformen_segmentebene.csv"
    )

    form_topic = _read_table(
        "09b_textform_x_hauptthema_segmentebene.csv"
    )

    topic_month_abs = _read_table(
        "05_hauptthemen_nach_monat_absolut.csv"
    )

    topic_month_pct = _read_table(
        "06_hauptthemen_nach_monat_prozent.csv"
    )

    monthly_issues = _read_table(
        "21_ausgaben_nach_monat.csv"
    )

    monthly_corpus = _read_table(
        "13_monatliche_korpusstruktur.csv"
    )

    # ------------------------------------------------------------
    # 01: Hauptthemen auf Segmentebene
    # ------------------------------------------------------------

    _horizontal_frequency(
        topic_segment,
        "primary_topic",
        f"Hauptthemen auf Segmentebene "
        f"(n = {int(topic_segment['n'].sum())})",
        "01_hauptthemen_segmentebene",
        BAR_BLUE,
    )

    # ------------------------------------------------------------
    # 02: Hauptthemen auf Werkebene
    # ------------------------------------------------------------

    _horizontal_frequency(
        topic_work,
        "primary_topic",
        f"Hauptthemen auf Werkebene "
        f"(n = {int(topic_work['n'].sum())})",
        "02_hauptthemen_werkebene",
        BAR_GOLD,
    )

    # ------------------------------------------------------------
    # 03: Textformen auf Segmentebene
    # ------------------------------------------------------------

    _horizontal_frequency(
        form_segment,
        "text_form",
        f"Textformen auf Segmentebene "
        f"(n = {int(form_segment['n'].sum())})",
        "03_textformen_segmentebene",
        BAR_BLUE,
    )

     # ------------------------------------------------------------
    # 03b: Textformen nach thematischer Zusammensetzung
    #
    # Die Balkenlänge entspricht der absoluten Zahl der Segmente
    # einer Textform. Die gestapelten Bereiche zeigen die
    # Hauptthemen. Prozentwerte werden nur bei Textformen mit
    # mindestens 10 Segmenten angezeigt.
    # ------------------------------------------------------------

    # Nur tatsächlich vorkommende Textformen darstellen
    # und nach Häufigkeit sortieren.
    form_topic["gesamt"] = form_topic[TOPIC_ORDER].sum(axis=1)

    form_topic = (
        form_topic.loc[form_topic["gesamt"].gt(0)]
        .sort_values("gesamt", ascending=True)
        .reset_index(drop=True)
    )

    y = np.arange(len(form_topic))
    left = np.zeros(len(form_topic))

    totals = (
        form_topic[TOPIC_ORDER]
        .sum(axis=1)
        .to_numpy(dtype=float)
    )

    fig, ax = plt.subplots(figsize=(9.4, 5.4))

    for topic in TOPIC_ORDER:
        counts = form_topic[topic].to_numpy(dtype=float)

        percentages = np.divide(
            counts,
            totals,
            out=np.zeros_like(counts),
            where=totals != 0,
        ) * 100

        ax.barh(
            y,
            counts,
            left=left,
            label=topic,
            color=TOPIC_COLORS[topic],
            height=0.68,
        )

        # Prozentwerte nur bei ausreichend großen Textformen
        # und ausreichend großen Balkensegmenten anzeigen.
        for i, (count, pct, total) in enumerate(
            zip(counts, percentages, totals)
        ):
            if total >= 10 and count >= 2 and pct >= 8:
                ax.text(
                    left[i] + count / 2,
                    y[i],
                    f"{pct:.1f} %",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    fontweight="bold",
                )

        left += counts

    # Absolute Gesamtzahl rechts neben jedem Balken anzeigen.
    for i, total in enumerate(totals):
        ax.text(
            total + max(totals) * 0.015,
            y[i],
            f"n = {int(total)}",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_title(
        "Textformen und ihre thematische Zusammensetzung",
        loc="left",
        fontweight="bold",
    )

    ax.set_xlabel("Anzahl der Segmente")
    ax.set_ylabel("")

    ax.set_yticks(
        y,
        form_topic["text_form"],
    )

    # Platz für die n-Angaben rechts neben den Balken.
    ax.set_xlim(
        0,
        max(totals) * 1.18,
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
    )

    ax.grid(
        axis="x",
        alpha=0.18,
    )

    fig.tight_layout()

    _save(
        fig,
        "03b_textformen_nach_hauptthemen",
    )

    # ------------------------------------------------------------
    # 04: Prozentuale Zusammensetzung der Hauptthemen nach Monat
    # ------------------------------------------------------------

    month_labels = topic_month_pct["month"].tolist()
    x = np.arange(len(month_labels))
    bottom = np.zeros(len(month_labels))

    fig, ax = plt.subplots(figsize=(9.2, 5.6))

    for topic in TOPIC_ORDER:
        values = topic_month_pct[topic].to_numpy(dtype=float)

        ax.bar(
            x,
            values,
            bottom=bottom,
            label=topic,
            color=TOPIC_COLORS[topic],
            width=0.72,
        )

        bottom += values

    ax.set_title(
        "Zusammensetzung der Hauptthemen nach Monat",
        loc="left",
        fontweight="bold",
    )

    ax.set_ylabel("Anteil der Segmente (%)")
    ax.set_xlabel("")

    ax.set_xticks(
        x,
        month_labels,
    )

    ax.set_ylim(0, 100)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        frameon=False,
    )

    ax.grid(
        axis="y",
        alpha=0.18,
    )

    fig.tight_layout()

    _save(
        fig,
        "04_hauptthemen_nach_monat_prozent",
    )

    # ------------------------------------------------------------
    # 04b:
    # Absolute Zahl der Segmente pro Monat,
    # gestapelt nach Hauptthemen.
    #
    # Die Balkenhöhe zeigt den Rückgang der Segmentzahl.
    # Innerhalb der Balken werden die prozentualen
    # Themenanteile des jeweiligen Monats angegeben.
    # ------------------------------------------------------------

    month_labels = topic_month_abs["month"].tolist()
    x = np.arange(len(month_labels))
    bottom = np.zeros(len(month_labels))

    fig, ax = plt.subplots(figsize=(9.4, 5.8))

    for topic in TOPIC_ORDER:

        counts = topic_month_abs[topic].to_numpy(dtype=float)
        percentages = topic_month_pct[topic].to_numpy(dtype=float)

        ax.bar(
            x,
            counts,
            bottom=bottom,
            label=topic,
            color=TOPIC_COLORS[topic],
            width=0.72,
        )

        # Prozentwerte innerhalb der Balkensegmente.
        #
        # Nur Segmente mit mindestens 2 Beiträgen werden
        # beschriftet, damit sehr kleine Flächen lesbar bleiben.
        for i, (count, pct) in enumerate(
            zip(counts, percentages)
        ):

            if count >= 2:
                ax.text(
                    x[i],
                    bottom[i] + count / 2,
                    f"{pct:.1f} %",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    fontweight="bold",
                )

        bottom += counts

    # Gesamtzahl der Segmente pro Monat
    monthly_totals = (
        topic_month_abs[TOPIC_ORDER]
        .sum(axis=1)
        .to_numpy(dtype=float)
    )

    # Gesamtzahl oberhalb der Balken anzeigen
    for i, total in enumerate(monthly_totals):
        ax.text(
            x[i],
            total + 0.8,
            f"n = {int(total)}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_title(
        "Feuilletonbeiträge und thematische Zusammensetzung nach Monat",
        loc="left",
        fontweight="bold",
    )

    ax.set_ylabel("Anzahl der Segmente")
    ax.set_xlabel("")

    ax.set_xticks(
        x,
        month_labels,
    )

    # Etwas Platz über dem höchsten Balken für n-Beschriftungen
    ax.set_ylim(
        0,
        max(monthly_totals) * 1.14,
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        frameon=False,
    )

    ax.grid(
        axis="y",
        alpha=0.18,
    )

    fig.tight_layout()

    _save(
        fig,
        "04b_themen_und_segmentzahl_nach_monat",
    )

    # ------------------------------------------------------------
    # 05: Anteil der digitalisierten Ausgaben mit Feuilleton
    # ------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    bars = ax.bar(
        monthly_issues["month"],
        monthly_issues[
            "anteil_feuilleton_an_digitalisierten_prozent"
        ],
        color=BAR_BLUE,
    )

    for bar, value in zip(
        bars,
        monthly_issues[
            "anteil_feuilleton_an_digitalisierten_prozent"
        ],
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.2,
            f"{value:.1f} %",
            ha="center",
        )

    ax.set_title(
        "Anteil der digitalisierten Ausgaben mit Feuilleton",
        loc="left",
        fontweight="bold",
    )

    ax.set_ylabel("Anteil (%)")

    ax.set_ylim(
        0,
        max(
            monthly_issues[
                "anteil_feuilleton_an_digitalisierten_prozent"
            ]
        )
        * 1.18,
    )

    ax.grid(
        axis="y",
        alpha=0.22,
    )

    fig.tight_layout()

    _save(
        fig,
        "05_feuilletonausgaben_nach_monat",
    )

    # ------------------------------------------------------------
    # 06: Segmente und Werke nach Monat
    # ------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8.8, 4.8))

    x = np.arange(len(monthly_corpus))
    width = 0.36

    ax.bar(
        x - width / 2,
        monthly_corpus["segmente"],
        width,
        label="Segmente",
        color=BAR_BLUE,
    )

    ax.bar(
        x + width / 2,
        monthly_corpus["werke"],
        width,
        label="Werke",
        color=BAR_GOLD,
    )

    ax.set_xticks(
        x,
        monthly_corpus["month"],
    )

    ax.set_title(
        "Erfasste Segmente und Werke nach Monat",
        loc="left",
        fontweight="bold",
    )

    ax.set_ylabel("Anzahl")

    ax.legend(
        frameon=False,
    )

    ax.grid(
        axis="y",
        alpha=0.22,
    )

    fig.tight_layout()

    _save(
        fig,
        "06_segmente_und_werke_nach_monat",
    )

    # ------------------------------------------------------------
    # 07: Nachgewiesene Feuilletonseiten nach Monat
    # ------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8.8, 4.8))

    ax.plot(
        monthly_issues["month"],
        monthly_issues["feuilletonseiten_summe"],
        marker="o",
        linewidth=2.2,
        color=BAR_BLUE,
    )

    for x_label, value in zip(
        monthly_issues["month"],
        monthly_issues["feuilletonseiten_summe"],
    ):
        ax.annotate(
            f"{value:.0f}",
            (x_label, value),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
        )

    ax.set_title(
        "Nachgewiesene Feuilletonseiten nach Monat",
        loc="left",
        fontweight="bold",
    )

    ax.set_ylabel(
        "Summe der Seitenangaben"
    )

    ax.grid(
        axis="y",
        alpha=0.22,
    )

    fig.tight_layout()

    _save(
        fig,
        "07_feuilletonseiten_nach_monat",
    )


if __name__ == "__main__":
    run()
    print("Abbildungen erstellt.")