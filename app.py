
import io
import math
import struct
import numpy as np
import pandas as pd
import streamlit as st
from scipy.spatial import ConvexHull, distance

st.set_page_config(
    page_title="OncoShape3D",
    page_icon="🧬",
    layout="wide"
)

CUSTOM_CSS = """
<style>
.main-title {
    font-size: 52px;
    font-weight: 900;
    color: #0F766E;
    margin-bottom: 0px;
}
.subtitle {
    font-size: 20px;
    color: #475569;
    margin-top: 0px;
    margin-bottom: 18px;
}
.hero-box {
    padding: 28px;
    border-radius: 22px;
    background: linear-gradient(135deg, #ECFDF5 0%, #F8FAFC 100%);
    border: 1px solid #CCFBF1;
    margin-bottom: 25px;
}
.section-title {
    font-size: 28px;
    font-weight: 800;
    color: #0F172A;
    margin-top: 25px;
    margin-bottom: 10px;
}
.subsection-title {
    font-size: 21px;
    font-weight: 700;
    color: #0F766E;
    margin-top: 18px;
}
.small-note {
    font-size: 13px;
    color: #64748B;
}
.card {
    padding: 18px;
    border-radius: 16px;
    background-color: #F8FAFC;
    border: 1px solid #E2E8F0;
    margin-bottom: 12px;
}
.warning-box {
    padding: 18px;
    border-radius: 16px;
    background-color: #FFFBEB;
    border: 1px solid #FDE68A;
    color: #92400E;
}
.footer {
    font-size: 13px;
    color: #64748B;
    text-align: center;
    margin-top: 40px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def parse_stl(file_bytes: bytes) -> np.ndarray:
    if len(file_bytes) >= 84:
        n = struct.unpack("<I", file_bytes[80:84])[0]
        if 84 + n * 50 == len(file_bytes):
            tris = np.empty((n, 3, 3), dtype=np.float64)
            offset = 84
            for i in range(n):
                values = struct.unpack("<12f", file_bytes[offset:offset + 48])
                tris[i, 0] = values[3:6]
                tris[i, 1] = values[6:9]
                tris[i, 2] = values[9:12]
                offset += 50
            return tris

    text = file_bytes.decode("utf-8", errors="ignore")
    vertices = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("vertex"):
            _, x, y, z = line.split()[:4]
            vertices.append([float(x), float(y), float(z)])
    if not vertices:
        raise ValueError("STL non leggibile o formato non valido.")
    return np.array(vertices, dtype=float).reshape((-1, 3, 3))

def mesh_topology(tris: np.ndarray):
    points_all = tris.reshape(-1, 3)
    points_unique, inverse = np.unique(np.round(points_all, 6), axis=0, return_inverse=True)
    faces = tris.shape[0]
    vertices = len(points_unique)
    face_idx = inverse.reshape(-1, 3)
    edges = np.vstack([
        np.sort(face_idx[:, [0, 1]], axis=1),
        np.sort(face_idx[:, [1, 2]], axis=1),
        np.sort(face_idx[:, [2, 0]], axis=1),
    ])
    unique_edges = np.unique(edges, axis=0)
    euler = int(vertices - len(unique_edges) + faces)
    return points_unique, vertices, faces, euler

def compute_metrics(filename: str, file_bytes: bytes) -> dict:
    tris = parse_stl(file_bytes)

    cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    surface = float(0.5 * np.linalg.norm(cross, axis=1).sum())

    signed_volume = np.einsum(
        "ij,ij->i",
        tris[:, 0],
        np.cross(tris[:, 1], tris[:, 2])
    ) / 6.0
    volume = abs(float(signed_volume.sum()))

    if surface <= 0 or volume <= 0:
        raise ValueError("Volume o superficie non validi. Controllare che la mesh sia chiusa.")

    equivalent_sphere_surface = math.pi ** (1 / 3) * (6 * volume) ** (2 / 3)
    sphericity = equivalent_sphere_surface / surface
    sv_ratio = surface / volume
    compactness = sphericity ** 3
    irregularity = surface / equivalent_sphere_surface

    points_unique, vertices, faces, euler = mesh_topology(tris)

    hull = ConvexHull(points_unique)
    hull_points = points_unique[hull.vertices]
    if len(hull_points) > 5000:
        idx = np.linspace(0, len(hull_points) - 1, 5000).astype(int)
        hull_points = hull_points[idx]
    max_diameter = float(distance.pdist(hull_points).max())

    centered = points_unique - points_unique.mean(axis=0)
    eigvals, eigvecs = np.linalg.eigh(np.cov(centered.T))
    eigvecs = eigvecs[:, np.argsort(eigvals)[::-1]]
    projected = centered @ eigvecs
    extents = projected.max(axis=0) - projected.min(axis=0)

    major_axis = float(extents[0])
    intermediate_axis = float(extents[1])
    minor_axis = float(extents[2])
    elongation = major_axis / minor_axis if minor_axis > 0 else np.nan

    return {
        "File": filename,
        "Volume mm3": round(volume, 2),
        "Superficie mm2": round(surface, 2),
        "Sfericita": round(sphericity, 4),
        "S/V mm-1": round(sv_ratio, 4),
        "Compattezza": round(compactness, 4),
        "Diametro max 3D mm": round(max_diameter, 2),
        "Asse maggiore mm": round(major_axis, 2),
        "Asse intermedio mm": round(intermediate_axis, 2),
        "Asse minore mm": round(minor_axis, 2),
        "Elongazione": round(elongation, 4),
        "Irregolarita superficie": round(irregularity, 4),
        "Euler": euler,
        "Faces": faces,
        "Vertices": vertices,
    }

with st.sidebar:
    st.markdown("## 🧬 OncoShape3D")
    st.markdown("**3D Tumor Morphometry Platform**")
    st.divider()
    st.markdown("### Moduli")
    st.markdown("- Analisi STL")
    st.markdown("- Parametri morfometrici")
    st.markdown("- Esportazione Excel/CSV")
    st.markdown("- Metodo scientifico")
    st.divider()
    st.markdown("### Nota")
    st.caption("Gli STL devono essere anonimizzati e preferibilmente espressi in millimetri.")

st.markdown('<div class="main-title">OncoShape3D</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Piattaforma per la morfometria tridimensionale dei tumori solidi da file STL</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero-box">
    <b>OncoShape3D</b> nasce per trasformare la geometria tridimensionale del tumore in dati quantitativi.
    L'obiettivo è affiancare ai parametri clinico-patologici tradizionali una descrizione numerica della forma,
    della superficie e della complessità spaziale della neoplasia.
    </div>
    """,
    unsafe_allow_html=True
)

tab_home, tab_analysis, tab_method, tab_disclaimer, tab_contacts = st.tabs(
    ["Home", "Analisi STL", "Metodo", "Disclaimer", "Contatti"]
)

with tab_home:
    st.markdown('<div class="section-title">Obiettivo della piattaforma</div>', unsafe_allow_html=True)

    st.write(
        """
        OncoShape3D consente di caricare modelli STL tridimensionali di tumori solidi e di ottenere
        automaticamente una serie di indici morfometrici. Questi parametri possono essere utilizzati
        per studi di ricerca traslazionale, analisi retrospettive, correlazioni con dati istopatologici
        e costruzione di database multicentrici.
        """
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="card">
            <div class="subsection-title">Dimensione tumorale</div>
            Volume, superficie, diametro massimo e assi principali descrivono l'estensione spaziale della lesione.
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """
            <div class="card">
            <div class="subsection-title">Forma 3D</div>
            Sfericità, compattezza, elongazione e irregolarità descrivono la geometria della crescita tumorale.
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            """
            <div class="card">
            <div class="subsection-title">Ricerca oncologica</div>
            Gli indici possono essere correlati a DOI, WPOI, grading, pT, pN, ENE e outcome clinici.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('<div class="section-title">Parametri calcolati</div>', unsafe_allow_html=True)
    st.markdown(
        """
        - Volume mm3
        - Superficie mm2
        - Sfericita
        - Rapporto superficie/volume
        - Compattezza
        - Diametro massimo 3D
        - Assi principali maggiore, intermedio e minore
        - Elongazione
        - Irregolarita di superficie
        - Euler, faces e vertices
        """
    )

with tab_analysis:
    st.markdown('<div class="section-title">Caricamento file STL</div>', unsafe_allow_html=True)

    st.markdown(
        """
        Carica uno o più file STL. I file dovrebbero essere anonimizzati prima dell'upload.
        """
    )

    uploaded_files = st.file_uploader(
        "Trascina qui uno o più file STL",
        type=["stl"],
        accept_multiple_files=True
    )

    if uploaded_files:
        results = []
        errors = []

        with st.spinner("Calcolo dei parametri morfometrici in corso..."):
            for uploaded in uploaded_files:
                try:
                    results.append(compute_metrics(uploaded.name, uploaded.read()))
                except Exception as e:
                    errors.append({"File": uploaded.name, "Errore": str(e)})

        if results:
            df = pd.DataFrame(results)

            st.success(f"Analisi completata: {len(df)} file elaborati correttamente.")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("STL analizzati", len(df))
            c2.metric("Volume medio", f"{df['Volume mm3'].mean():.2f} mm³")
            c3.metric("Sfericità media", f"{df['Sfericita'].mean():.4f}")
            c4.metric("Irregolarità media", f"{df['Irregolarita superficie'].mean():.4f}")

            st.markdown('<div class="section-title">Tabella risultati</div>', unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Scarica CSV",
                data=csv,
                file_name="OncoShape3D_parametri_STL.csv",
                mime="text/csv"
            )

            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="Parametri STL")

            st.download_button(
                "Scarica Excel",
                data=excel_buffer.getvalue(),
                file_name="OncoShape3D_parametri_STL.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        if errors:
            st.warning("Alcuni file non sono stati elaborati.")
            st.dataframe(pd.DataFrame(errors), use_container_width=True)

with tab_method:
    st.markdown('<div class="section-title">Metodo di calcolo</div>', unsafe_allow_html=True)

    st.markdown(
        """
        La piattaforma analizza la mesh triangolare contenuta nel file STL. A partire dai vertici e dalle facce
        della mesh vengono calcolati volume, superficie, assi principali e indici di forma.
        """
    )

    st.markdown("### Interpretazione generale")
    st.markdown(
        """
        **Parametri dimensionali**
        - Volume
        - Superficie
        - Diametro massimo 3D
        - Assi principali

        Questi parametri descrivono prevalentemente quanto è estesa la lesione.

        **Parametri morfologici**
        - Sfericita
        - Compattezza
        - Rapporto superficie/volume
        - Elongazione
        - Irregolarita di superficie

        Questi parametri descrivono come la lesione è organizzata nello spazio.
        """
    )

    st.markdown("### Requisiti tecnici")
    st.markdown(
        """
        Per una corretta interpretazione:
        - il file STL dovrebbe essere in millimetri;
        - la mesh dovrebbe essere chiusa;
        - il modello dovrebbe rappresentare esclusivamente il volume di interesse;
        - i file caricati dovrebbero essere privi di dati identificativi del paziente.
        """
    )

with tab_disclaimer:
    st.markdown('<div class="section-title">Disclaimer</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="warning-box">
        OncoShape3D è uno strumento di ricerca e analisi morfometrica.
        Non deve essere utilizzato come dispositivo medico, né come strumento autonomo per diagnosi,
        stadiazione, decisioni terapeutiche o valutazioni prognostiche individuali.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        I risultati devono essere interpretati da personale qualificato e sempre integrati con dati clinici,
        radiologici e istopatologici. L'accuratezza dei parametri dipende dalla qualità della segmentazione,
        dalla correttezza del file STL e dalla chiusura della mesh.
        """
    )

    st.markdown("### Privacy")
    st.markdown(
        """
        Prima del caricamento, i file devono essere anonimizzati. Evitare nomi file contenenti nome,
        cognome, data di nascita o altri identificativi del paziente.
        """
    )

with tab_contacts:
    st.markdown('<div class="section-title">Contatti e progetto</div>', unsafe_allow_html=True)

    st.write(
        """
        OncoShape3D è una piattaforma in sviluppo per la quantificazione della morfologia tridimensionale
        dei tumori solidi da modelli STL.
        """
    )

    st.markdown("### Possibili sviluppi")
    st.markdown(
        """
        - Visualizzatore 3D interattivo
        - Report PDF automatico
        - Modulo clinico-patologico
        - Database multicentrico anonimizzato
        - Score morfometrico di rischio
        """
    )

    st.markdown("### Contatto")
    st.write("Inserire qui email istituzionale o riferimento del gruppo di ricerca.")

st.markdown(
    '<div class="footer">OncoShape3D · 3D Tumor Morphometry Platform · Research use only</div>',
    unsafe_allow_html=True
)
