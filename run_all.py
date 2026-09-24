"""Führt alle Analyseschritte aus"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import abbildungen
import analyse_ausgaben
import analyse_autoren
import analyse_codierung
from analyse_utils import (
    FORM_ORDER,
    RESULT_DIR,
    SECONDARY_TOPIC_ORDER,
    TOPIC_ORDER,
    WORK_FIELDS,
    ensure_output_dirs,
    load_coding,
    load_issues,
    save_table,
)

def _pct(numerator: int | float, denominator: int | float) -> float:
    return round(float(numerator) / float(denominator) * 100, 1)


def run_quality_checks(
    segments: pd.DataFrame,
    issues: pd.DataFrame,
) -> pd.DataFrame:
    """Prüft Eingaben und bricht bei widersprüchlichen Daten ab."""
    ensure_output_dirs()
    checks = []

    def check(name, passed, note=""):
        checks.append(
            (
                name,
                "bestanden" if bool(passed) else "FEHLER",
                note,
            )
        )

    required = [
        "date", "anno_issue_id", "contribution_id", "work_id",
        "title", "text_form", "primary_topic",
        "author_identity_status", "series_type",
        "coding_grounds", "coding_status",
    ]

    check(
        "Pflichtfelder vollständig",
        segments[required]
        .fillna("")
        .astype(str)
        .apply(lambda s: s.str.strip().ne(""))
        .all()
        .all(),
    )
    check(
        "Eingaben nicht leer",
        not segments.empty and not issues.empty,
    )
    check(
        "Segment-IDs eindeutig",
        segments["contribution_id"].is_unique,
    )
    check(
        "Ausgaben-IDs vollständig und eindeutig",
        issues["anno_issue_id"].notna().all()
        and issues["anno_issue_id"].is_unique,
    )
    check(
        "Kodierung abgeschlossen",
        segments["coding_status"].eq("final").all(),
    )
    check(
        "Hauptthemen zulässig",
        segments["primary_topic"].isin(TOPIC_ORDER).all(),
    )
    check(
        "Zweitthemen zulässig",
        segments["secondary_topic"]
        .dropna()
        .isin(SECONDARY_TOPIC_ORDER)
        .all(),
    )
    check(
        "Textformen zulässig",
        segments["text_form"].isin(FORM_ORDER).all(),
    )
    check(
        "Haupt- und Zweitthema verschieden",
        not segments["primary_topic"]
        .eq(segments["secondary_topic"])
        .fillna(False)
        .any(),
    )

    status_order = [
        "gesichert",
        "wahrscheinlich",
        "unsicher",
        "nicht identifiziert",
        "ohne Autor",
    ]
    check(
        "Autorenstatus zulässig",
        segments["author_identity_status"]
        .isin(status_order)
        .all(),
    )
    check(
        "Ohne Autor entspricht fehlender Normalisierung",
        segments["author_normalized"]
        .isna()
        .eq(
            segments["author_identity_status"]
            .eq("ohne Autor")
            .fillna(False)
        )
        .all(),
    )

    conflicts = (
        segments.groupby("work_id")[WORK_FIELDS]
        .nunique(dropna=False)
    )
    bad = conflicts.index[
        conflicts.gt(1).any(axis=1)
    ].tolist()

    check(
        "Werkmetadaten konsistent",
        not bad,
        ", ".join(bad),
    )
    check(
        "Reihentyp zulässig",
        segments["series_type"].isin(
            ["keine", "Fortsetzungsbeitrag", "Artikelreihe"]
        ).all(),
    )
    check(
        "Reihenkennungen konsistent",
        segments["series_id"].notna().eq(
            segments["series_type"]
            .eq("Artikelreihe")
            .fillna(False)
        ).all(),
    )

    for label, frame in [
        ("Kodierung", segments),
        ("Ausgaben", issues),
    ]:
        check(
            label + ": Datum im Zeitraum",
            frame["date"].between(
                pd.Timestamp("1848-01-01"),
                pd.Timestamp("1848-06-30"),
            ).all(),
        )
        check(
            label + ": Datum und Ausgaben-ID passen",
            frame["anno_issue_id"].eq(
                "wrz" + frame["date"].dt.strftime("%Y%m%d")
            ).fillna(False).all(),
        )

    check(
        "Digitalisierungsstatus zulässig",
        issues["digitized"].isin(["ja", "nein"]).all(),
    )

    digitized = issues["digitized"].eq("ja")
    check(
        "Feuilletonstatus dokumentiert",
        issues.loc[digitized, "has_feuilleton"]
        .isin(["ja", "nein"]).all()
        and issues.loc[~digitized, "has_feuilleton"]
        .isna().all(),
    )

    coded = set(segments["anno_issue_id"].dropna())
    marked = set(
        issues.loc[
            issues["has_feuilleton"].eq("ja"),
            "anno_issue_id",
        ]
    )

    check(
        "Alle kodierten Ausgaben im Protokoll",
        coded <= set(issues["anno_issue_id"]),
    )
    check(
        "Feuilleton-Ausgaben stimmen überein",
        coded == marked,
        "Abweichende IDs: "
        + ", ".join(sorted(coded ^ marked)),
    )

    pages = segments[["page_start", "page_end"]]
    valid_pages = (
        pages.notna().all().all()
        and pages.ge(1).fillna(False).all().all()
        and pages.mod(1).eq(0).fillna(False).all().all()
        and segments["page_end"]
        .ge(segments["page_start"])
        .fillna(False)
        .all()
    )
    check("Segmentseiten gültig", valid_pages)

    fp = issues["feuilleton_pages"]
    has = issues["has_feuilleton"].eq("ja").fillna(False)
    no = issues["has_feuilleton"].eq("nein").fillna(False)

    check(
        "Seitenzahlen im Protokoll gültig",
        (
            fp.loc[has].ge(1)
            & fp.loc[has].mod(1).eq(0)
        ).all()
        and fp.loc[no].eq(0).all()
        and fp.loc[~digitized].isna().all(),
    )

    if valid_pages and issues["anno_issue_id"].is_unique:
        page_sets = {}

        for row in segments.itertuples():
            page_sets.setdefault(
                row.anno_issue_id, set()
            ).update(
                range(
                    int(row.page_start),
                    int(row.page_end) + 1,
                )
            )

        protocol = issues.set_index(
            "anno_issue_id"
        )["feuilleton_pages"]

        differences = [
            issue
            for issue, values in page_sets.items()
            if issue not in protocol.index
            or pd.isna(protocol[issue])
            or len(values) != protocol[issue]
        ]

        check(
            "Seitenabgleich beider Tabellen",
            not differences,
            ", ".join(sorted(differences)),
        )

    quality = pd.DataFrame(
        checks,
        columns=["pruefung", "status", "hinweis"],
    )
    save_table(quality, "99_datenqualitaet.csv")

    failed = quality.loc[
        quality["status"].eq("FEHLER")
    ]

    if not failed.empty:
        raise ValueError(
            "Analyse abgebrochen. "
            "Siehe 99_datenqualitaet.csv:\n"
            + failed.to_string(index=False)
        )

    return quality

def main() -> None:
    # Zuerst Eingaben prüfen, noch keine Analysen erzeugen.
    segments = load_coding()
    issues = load_issues()
    run_quality_checks(segments, issues)

    # Nur nach erfolgreicher Prüfung auswerten.
    coding_results = analyse_codierung.run()
    issue_results = analyse_ausgaben.run()
    analyse_autoren.run()
    abbildungen.run()

    print(
        "Analyse abgeschlossen. "
        "Tabellen und Abbildungen liegen im Ordner ergebnisse."
    )

if __name__ == "__main__":
    main()
