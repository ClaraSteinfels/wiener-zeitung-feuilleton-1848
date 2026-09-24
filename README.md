# Deskriptive Analyse des frühen Feuilletons der Wiener Zeitung

Dieses Paket wertet die Kodierung und das Ausgabenprotokoll für den Zeitraum
1. Januar bis 30. Juni 1848 reproduzierbar mit Python aus.

Die Quelldateien werden von den Skripten nicht verändert. 
Alle Ergebnisse können durch erneutes Ausführen reproduziert und überschrieben werden.

## Dateien

- `analyse_codierung.py`: Themen, Textformen, Werk- und Serienstruktur
- `analyse_ausgaben.py`: Präsenz und Umfang des Feuilletons nach Ausgaben und Monaten
- `analyse_autoren.py`: Autorstatus, normalisierte Zuschreibungen und Signaturvarianten
- `abbildungen.py`: Abbildungen als PNG (300 dpi) und SVG
- `analyse_utils.py`: Einlesen, Zeitvariablen und gemeinsame Hilfsfunktionen
- `run_all.py`: führt alle Schritte aus
- `data/`: Eingabe-CSVs, Kodierungstabelle und Ausgabenprotokoll 
- `documentation/`: Kodierhandbuch
- `ergebnisse/tabellen/`: erzeugte Ergebnistabellen als UTF-8-BOM/Strichpunkt-CSV
- `ergebnisse/abbildungen/`: erzeugte Abbildungen

## Ausführen

Im Projektordner:

```bash
python -m pip install -r requirements.txt
python run_all.py
```

Benötigt werden Python 3.10 oder neuer sowie `pandas`, `numpy` und `matplotlib`.

## Zentrale Auswertungsentscheidungen

- Segmentebene: Jede `contribution_id` zählt einmal (final: 168 Segmente).
- Werkebene: Jede `work_id` zählt einmal (final: 150 Werke); zeitlich wird das Werk dem ersten Segment zugeordnet.
- Autorenhäufigkeiten werden primär auf Werkebene berechnet; die Segmentebene wird nur ergänzend ausgegeben.
- `author_normalized` ist die gemeinsame Vergleichsbezeichnung. Sie enthält entweder den identifizierten Namen oder bei unbekannter Identität ein stabiles Alias/eine Chiffre wie `Quidam`.
- Gesicherte, wahrscheinliche, unsichere und nicht identifizierte Bezeichnungen bleiben über `author_identity_status` unterscheidbar. Aliasse werden in Häufigkeitsvergleichen berücksichtigt, aber nicht als gesicherte bürgerliche Namen behandelt.
- `coding_grounds` dokumentiert für jedes Segment knapp die thematische Kodierentscheidung.
- Beide Ebenen werden getrennt berichtet und nie in demselben Nenner vermischt.
- Für Monatsvergleiche werden absolute Häufigkeiten und Prozentwerte gemeinsam ausgegeben.
- Der politische Einschnitt wird deskriptiv als „vor dem 13. März“ und „ab dem 13. März“ operationalisiert.
- Nicht digitalisierte Ausgaben werden beim Anteil der Ausgaben mit Feuilleton nicht in den Nenner aufgenommen.

## Methodische Hinweise

1. Segment- und Werkebene werden strikt getrennt. Fortsetzungen erhalten auf Segmentebene ein größeres Gewicht; auf Werkebene wird jedes `work_id` einmal gezählt.
2. Für monatliche Werkstatistiken wird ein Werk dem Monat seines ersten sichtbaren Segments zugeordnet.
3. Prozentwerte nennen stets ihren Nenner. 
