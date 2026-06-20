# OncoShape3D

OncoShape3D è una piattaforma web per la morfometria tridimensionale di tumori solidi da file STL.

## Funzioni attuali

- Upload multiplo di file STL
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
- Sezione metodo
- Disclaimer
- Pagina contatti

## Avvio locale

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Uso online

Il progetto può essere pubblicato su Streamlit Community Cloud collegando questo repository GitHub.

## Nota

OncoShape3D è destinato a uso di ricerca. Non è un dispositivo medico.