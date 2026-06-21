# OncoShape3D

Versione aggiornata con **Morphometric Infiltration Score (MIS)**.

## Novità

- Calcolo automatico del MIS
- MIS inserito come primo risultato evidenziato
- MIS esportato nel file Excel
- Categorie esplorative:
  - < 4.0 = Basso
  - 4.0–6.5 = Intermedio
  - ≥ 6.5 = Alto

## Formula

MIS = z(Sfericità) + z(Compattezza) - z(Irregolarità superficie)

Lo score è normalizzato in scala 0–10 usando le costanti del dataset OncoShape3D attuale.

## Nota

Il MIS è uno score esplorativo di ricerca, non un dispositivo medico e non sostituisce la diagnosi istopatologica.
