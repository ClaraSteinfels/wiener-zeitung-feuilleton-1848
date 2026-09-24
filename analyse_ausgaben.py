"""Deskriptive Statistiken für das Ausgabenprotokoll."""

from __future__ import annotations

import pandas as pd

from analyse_utils import ensure_output_dirs, load_coding, load_issues, save_table


def run() -> dict[str, pd.DataFrame]:
    ensure_output_dirs()
    issues = load_issues()
    coding = load_coding()

    coded_by_issue = (
        coding.groupby("anno_issue_id", as_index=False)
        .agg(
            segmente=("contribution_id", "nunique"),
            werke=("work_id", "nunique"),
        )
    )
    issue_detail = issues.merge(coded_by_issue, on="anno_issue_id", how="left", validate="one_to_one")
    issue_detail[["segmente", "werke"]] = issue_detail[["segmente", "werke"]].fillna(0).astype(int)

    digitized = issues.loc[issues["digitized_bool"]]
    feuilleton = digitized.loc[digitized["has_feuilleton_bool"]]
    overview = pd.DataFrame(
        [
            ("Ausgabennummern im Zeitraum", len(issues), "alle 179 Ausgabennummern"),
            ("Digitalisierte Ausgaben", int(issues["digitized_bool"].sum()), "Nenner: alle Ausgabennummern"),
            ("Nicht digitalisierte Ausgaben", int((~issues["digitized_bool"]).sum()), "Nenner: alle Ausgabennummern"),
            ("Digitalisierte Ausgaben mit Feuilleton", len(feuilleton), "Nenner: digitalisierte Ausgaben"),
            ("Anteil digitalisierter Ausgaben mit Feuilleton (%)", round(len(feuilleton) / len(digitized) * 100, 1), "Nenner: digitalisierte Ausgaben"),
            ("Summe nachgewiesener Feuilletonseiten", float(feuilleton["feuilleton_pages"].sum()), "nur Ausgaben mit Feuilleton"),
            ("Mittlere Feuilletonseiten je Feuilleton-Ausgabe", round(float(feuilleton["feuilleton_pages"].mean()), 2), "nur Ausgaben mit Feuilleton"),
            ("Mittlere Segmente je Feuilleton-Ausgabe", round(float(issue_detail.loc[issue_detail["has_feuilleton_bool"], "segmente"].mean()), 2), "nur Ausgaben mit Feuilleton"),
        ],
        columns=["kennzahl", "wert", "hinweis"],
    )

    monthly = (
        issue_detail.groupby(["month_number", "month"], observed=True)
        .agg(
            ausgabennummern=("anno_issue_id", "size"),
            digitalisierte_ausgaben=("digitized_bool", "sum"),
            ausgaben_mit_feuilleton=("has_feuilleton_bool", "sum"),
            feuilletonseiten_summe=("feuilleton_pages", "sum"),
            segmente=("segmente", "sum"),
            werke_in_ausgaben=("werke", "sum"),
        )
        .reset_index()
        .sort_values("month_number")
    )
    monthly["anteil_feuilleton_an_digitalisierten_prozent"] = (
        monthly["ausgaben_mit_feuilleton"]
        .div(monthly["digitalisierte_ausgaben"])
        .mul(100)
        .round(1)
    )
    monthly["mittlere_feuilletonseiten_je_feuilletonausgabe"] = (
        monthly["feuilletonseiten_summe"]
        .div(monthly["ausgaben_mit_feuilleton"])
        .round(2)
    )
    monthly["mittlere_segmente_je_feuilletonausgabe"] = (
        monthly["segmente"]
        .div(monthly["ausgaben_mit_feuilleton"])
        .round(2)
    )

    period = (
        issue_detail.groupby("revolution_period", observed=True)
        .agg(
            ausgabennummern=("anno_issue_id", "size"),
            digitalisierte_ausgaben=("digitized_bool", "sum"),
            ausgaben_mit_feuilleton=("has_feuilleton_bool", "sum"),
            feuilletonseiten_summe=("feuilleton_pages", "sum"),
            segmente=("segmente", "sum"),
        )
        .reindex(["Vor dem 13. März", "Ab dem 13. März"])
        .reset_index()
    )
    period["anteil_feuilleton_an_digitalisierten_prozent"] = (
        period["ausgaben_mit_feuilleton"].div(period["digitalisierte_ausgaben"]).mul(100).round(1)
    )

    tables = {
        "20_ausgaben_uebersicht.csv": overview,
        "21_ausgaben_nach_monat.csv": monthly,
        "22_ausgaben_vor_nach_13_maerz.csv": period,
        "23_ausgaben_detail_mit_korpuszaehlung.csv": issue_detail,
    }
    for filename, table in tables.items():
        save_table(table, filename)
    return {"issues": issues, "issue_detail": issue_detail, **tables}


if __name__ == "__main__":
    run()
    print("Ausgabenanalyse abgeschlossen.")

