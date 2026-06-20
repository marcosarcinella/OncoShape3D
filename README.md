# OncoShape3D

OncoShape3D è una piattaforma web locale per calcolare parametri morfometrici 3D da file STL.

## Funzioni

- Upload multiplo di STL
- Calcolo automatico di:
  - Volume mm3
  - Superficie mm2
  - Sfericita
  - S/V mm-1
  - Compattezza
  - Diametro max 3D mm
  - Asse maggiore mm
  - Asse intermedio mm
  - Asse minore mm
  - Elongazione
  - Irregolarita superficie
  - Euler
  - Faces
  - Vertices
- Download Excel
- Download CSV

## Installazione

Apri il terminale nella cartella del progetto ed esegui:

```bash
pip install -r requirements.txt
```

## Avvio

```bash
streamlit run app.py
```

Poi apri:

```text
http://localhost:8501
```

## Nota

Gli STL devono essere in millimetri e preferibilmente mesh chiuse/watertight.