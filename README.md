# OncoShape3D

Versione con modulo clinico e upload database.

## Funzioni

- Analisi STL
- Visualizzatore 3D
- Export Excel con colonne cliniche vuote
- Nuova pagina Upload
- Invio del file Excel compilato a oncoshape3d@gmail.com tramite SMTP configurato nei Secrets Streamlit

## Secrets necessari su Streamlit Cloud

Nel menu della tua app Streamlit Cloud vai su:

Settings → Secrets

e inserisci:

```toml
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = "oncoshape3d@gmail.com"
SMTP_PASSWORD = "INSERISCI_APP_PASSWORD_GMAIL"
EMAIL_TO = "oncoshape3d@gmail.com"
```

Per Gmail serve una App Password, non la password normale dell'account.

## Avvio locale

```bash
pip install -r requirements.txt
streamlit run app.py
```