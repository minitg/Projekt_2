"""Create the three-class dataset view from existing audited precipitation_v2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from classification_data import build_classification_dataset


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "data/processed/precipitation_v2")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/precipitation_classification_v1")
    args = parser.parse_args()
    manifest = build_classification_dataset(args.source_dir, args.output_dir)
    balance = json.loads((args.output_dir / "class_balance.json").read_text())
    rows = "\n".join(
        f"| {r['horizon_h']} | {r['split']} | {r['n']} | {r['class_counts']['0']} | {r['class_counts']['1']} | {r['class_counts']['2']} |"
        for r in balance
    )
    threshold = manifest["config"]["upper_class_threshold_mm"]
    (args.output_dir / "DATASET.md").write_text(f"""# Niederschlagsklassifikation

Ziel ist `target_class`: **0 = 0 mm**, **1 = >0 bis <{threshold:g} mm**, **2 = >={threshold:g} mm** je zukünftigem Dreistundenblock. Dies sind Projektklassen, keine amtlichen Warnstufen.

Der Datensatz ergänzt `precipitation_v2`. Er enthält unverändert **{manifest['cases']:,}** zulässige Fälle über alle Vorläufe; keine Über-/Unterabtastung und keine ergänzten Zielwerte. Die Fälle sind zeitlich abhängig. Die Quelle, UTC-Prüfung, Qualitätsauswahl und hypothetische Einstundenverzögerung werden aus v2 übernommen. 2024 bleibt rückblickende Evaluierung.

## Dateien

- `cases.parquet`: Fall-ID, Vorlauf, Abschnitt, Eingabefenster und Zielzeitraum; keine Zielklasse oder Zielmenge.
- `labels.parquet`: Fall-ID, kategoriale Zielklasse und beobachtete Zielsumme nur für Training/Auswertung.
- `hourly_features.parquet`: dieselben skalierten Eingaben wie v2; Sequenzen oder flache XGBoost-Ansicht.
- `hourly_context.parquet`: dieselbe Historie in Originaleinheiten sowie beobachteter Niederschlag mit NaN bei Messlücken für Forecasting-Adapter.
- `schema.json`: Klassen, Reihenfolge der drei Wahrscheinlichkeiten und Eingaberollen.
- `class_balance.json`: Häufigkeiten getrennt nach Vorlauf und Abschnitt; kein automatisches Klassenbalancing.
- `preprocessing.json`, `manifest.json`: Trainingsstatistiken, Herkunft und Dateiprüfsummen.

## Vergleichsfälle

| Vorlauf h | Abschnitt | Fälle | Klasse 0 | Klasse 1 | Klasse 2 |
| --- | --- | ---: | ---: | ---: | ---: |
{rows}

## Modellnutzung

`load_classification_bundle(...).arrays(h, split)` liefert `X, y, cases`. Für XGBoost `flatten=True` setzen. Labels und Reihenfolge sind identisch. Die Standardhistorie hat 48 Stunden und 20 Merkmale.

`foundation_records(h, split)` liefert modellneutrale Forecasting-Eingaben, keine fertige Chronos-/Timestral-API: historische R1-Messungen in mm, historische Zusatzmerkmale und ausschließlich bekannte Kalendermerkmale in der Zukunft. Labels und spätere Wetterbeobachtungen werden nicht mitgegeben. Die zukünftige Stundenachse enthält auch die zurückgehaltene Beobachtungslücke; der Klassenblock wird durch `class_target_hour_indices` markiert. Bei h=3 und einer Stunde Verzögerung sind dies die letzten drei von vier vorherzusagenden Stunden.

Ein vortrainierter numerischer Forecaster ist dadurch noch kein Dreiklassenmodell. Für Klassenwahrscheinlichkeiten ist ein im Training angepasster Klassifikationsadapter/-kopf oder eine begründete Verteilungsumrechnung erforderlich. Die IDs 0/1/2 dürfen nicht als Niederschlagsmengen prognostiziert werden. Einzelne Mittelwerte oder die Summe marginaler Stundenquantile ergeben keine belastbaren Klassenwahrscheinlichkeiten. Gemeinsame Stichprobenpfade müssen je Pfad über den Zielblock summiert werden. Es wurden hier keine Modelle trainiert oder XAI-Erklärungen berechnet.

Die geschätzten physikalischen Niederschlagsintervallenden liegen weiterhin beim jeweiligen DWD-Referenzstempel plus dem konfigurierten Minutenversatz. Historische tatsächliche Empfangszeiten sind nicht bekannt. Zukunftskalender beziehen sich auf UTC-Referenzzeiten, wie die bisherigen Kalenderinputs.
""")
    print(f"Saved {manifest['cases']:,} three-class cases to {args.output_dir}")
    for row in balance:
        print(f"h={row['horizon_h']:2d} {row['split']:10s} n={row['n']:5d} classes={row['class_counts']}")


if __name__ == "__main__":
    main()
