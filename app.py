
import io
import math
import struct
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.spatial import ConvexHull, distance

st.set_page_config(
    page_title="OncoShape3D",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

CUSTOM_CSS = """
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1250px;
}
.hero {
    display: flex;
    align-items: center;
    gap: 24px;
    margin-bottom: 12px;
}
.logo-mark {
    width: 96px;
    height: 96px;
    border-radius: 24px;
    background: linear-gradient(135deg, #0F766E, #14B8A6);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 14px 34px rgba(15, 118, 110, 0.24);
    color: white;
    font-weight: 900;
    letter-spacing: -2px;
}
.logo-main {
    font-size: 42px;
    line-height: 1;
}
.logo-sub {
    font-size: 20px;
    vertical-align: super;
    margin-left: 1px;
}
.title-wrap h1 {
    font-size: 56px;
    line-height: 1.0;
    margin: 0;
    color: #0F172A;
    font-weight: 900;
    letter-spacing: -2px;
}
.title-wrap h1 span {
    color: #0F766E;
}
.title-wrap p {
    margin-top: 8px;
    font-size: 19px;
    color: #475569;
}
.intro-box {
    padding: 22px 26px;
    border-radius: 24px;
    background: linear-gradient(135deg, #ECFDF5 0%, #F8FAFC 100%);
    border: 1px solid #CCFBF1;
    margin-top: 16px;
    margin-bottom: 24px;
    color: #334155;
    font-size: 16px;
}
.nav-grid {
    display: grid;
    grid-template-columns: repeat(5, minmax(130px, 1fr));
    gap: 14px;
    margin: 20px 0 26px 0;
}
div[data-testid="column"] .stButton button {
    width: 100%;
    min-height: 132px;
    border-radius: 18px;
    border: 1px solid #E2E8F0;
    background: white;
    color: #0F172A;
    font-weight: 800;
    font-size: 16px;
    box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
    transition: all 0.15s ease-in-out;
}
div[data-testid="column"] .stButton button:hover {
    border-color: #0F766E;
    box-shadow: 0 12px 26px rgba(15, 118, 110, 0.14);
    transform: translateY(-1px);
}
.panel {
    padding: 26px;
    border-radius: 24px;
    background: white;
    border: 1px solid #E2E8F0;
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.05);
    margin-bottom: 18px;
}
.panel h2 {
    color: #0F766E;
    font-size: 24px;
    font-weight: 800;
    margin-top: 0;
}
.result-row {
    display: grid;
    grid-template-columns: 48px 1fr 160px;
    align-items: center;
    gap: 10px;
    padding: 12px 8px;
    border-bottom: 1px solid #E2E8F0;
}
.result-icon {
    width: 34px;
    height: 34px;
    border-radius: 10px;
    background: #ECFDF5;
    color: #0F766E;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
}
.result-name {
    font-weight: 650;
    color: #0F172A;
}
.result-value {
    text-align: right;
    font-weight: 800;
    color: #0F172A;
}
.note-box {
    padding: 20px 24px;
    border-radius: 20px;
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    color: #475569;
}
.info-card {
    padding: 22px;
    border-radius: 20px;
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    margin-bottom: 16px;
}
.info-card h3 {
    color: #0F766E;
    margin-top: 0;
}
.footer {
    margin-top: 35px;
    padding-top: 18px;
    border-top: 1px solid #E2E8F0;
    display: flex;
    justify-content: space-between;
    color: #64748B;
    font-size: 13px;
}
@media (max-width: 900px) {
    .hero {
        flex-direction: column;
        align-items: flex-start;
    }
    .title-wrap h1 {
        font-size: 42px;
    }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Navigation state
if "page" not in st.session_state:
    st.session_state.page = "Analisi STL"

def set_page(page_name):
    st.session_state.page = page_name

def parse_stl(file_bytes: bytes):
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

def mesh_topology(tris):
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
    return points_unique, face_idx, vertices, faces, euler

def compute_metrics(filename, file_bytes):
    tris = parse_stl(file_bytes)

    cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    surface = float(0.5 * np.linalg.norm(cross, axis=1).sum())
    signed_volume = np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])) / 6.0
    volume = abs(float(signed_volume.sum()))

    if surface <= 0 or volume <= 0:
        raise ValueError("Volume o superficie non validi. Controllare che la mesh sia chiusa.")

    equivalent_sphere_surface = math.pi ** (1 / 3) * (6 * volume) ** (2 / 3)
    sphericity = equivalent_sphere_surface / surface
    sv_ratio = surface / volume
    compactness = sphericity ** 3
    irregularity = surface / equivalent_sphere_surface

    points_unique, face_idx, vertices, faces, euler = mesh_topology(tris)

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

    metrics = {
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
    return metrics, points_unique, face_idx

def make_3d_plot(points, faces, max_faces=15000):
    if len(faces) > max_faces:
        idx = np.linspace(0, len(faces) - 1, max_faces).astype(int)
        faces_viz = faces[idx]
    else:
        faces_viz = faces

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                i=faces_viz[:, 0],
                j=faces_viz[:, 1],
                k=faces_viz[:, 2],
                opacity=0.72,
                color="#14B8A6",
                flatshading=False,
                lighting=dict(ambient=0.35, diffuse=0.7, specular=0.25, roughness=0.7),
                lightposition=dict(x=100, y=200, z=300),
            )
        ]
    )
    fig.update_layout(
        height=520,
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            bgcolor="rgba(248,250,252,1)",
        ),
        paper_bgcolor="rgba(255,255,255,0)",
    )
    return fig

def result_vertical_table(row):
    icons = {
        "Volume mm3": "▣",
        "Superficie mm2": "△",
        "Sfericita": "◉",
        "S/V mm-1": "↔",
        "Compattezza": "⬢",
        "Diametro max 3D mm": "⟷",
        "Asse maggiore mm": "↦",
        "Asse intermedio mm": "↔",
        "Asse minore mm": "↤",
        "Elongazione": "⤢",
        "Irregolarita superficie": "≈",
        "Euler": "χ",
        "Faces": "F",
        "Vertices": "V",
    }
    units = {
        "Volume mm3": "mm³",
        "Superficie mm2": "mm²",
        "S/V mm-1": "mm⁻¹",
        "Diametro max 3D mm": "mm",
        "Asse maggiore mm": "mm",
        "Asse intermedio mm": "mm",
        "Asse minore mm": "mm",
    }
    order = [
        "Volume mm3", "Superficie mm2", "Sfericita", "S/V mm-1", "Compattezza",
        "Diametro max 3D mm", "Asse maggiore mm", "Asse intermedio mm",
        "Asse minore mm", "Elongazione", "Irregolarita superficie",
        "Euler", "Faces", "Vertices"
    ]
    html = ""
    for key in order:
        val = row[key]
        unit = units.get(key, "")
        value_text = f"{val} {unit}".strip()
        html += f"""
        <div class="result-row">
            <div class="result-icon">{icons.get(key, "•")}</div>
            <div class="result-name">{key}</div>
            <div class="result-value">{value_text}</div>
        </div>
        """
    return html

# Header
st.markdown(
    """
    <div class="hero">
        <div class="logo-mark">
            <span class="logo-main">OS</span><span class="logo-sub">3D</span>
        </div>
        <div class="title-wrap">
            <h1>Onco<span>Shape3D</span></h1>
            <p>3D Tumor Morphometry Platform</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="intro-box">
    <b>Piattaforma per la morfometria tridimensionale dei tumori solidi da file STL.</b><br>
    OncoShape3D trasforma la geometria tridimensionale del tumore in dati quantitativi,
    utili per ricerca, confronto morfologico e integrazione con parametri clinico-patologici.
    </div>
    """,
    unsafe_allow_html=True
)

# Real clickable navigation
cols = st.columns(5)
nav_items = [
    ("🏠\n\nHome", "Home"),
    ("☁️\n\nAnalisi STL", "Analisi STL"),
    ("🔬\n\nMetodo", "Metodo"),
    ("🛡️\n\nDisclaimer", "Disclaimer"),
    ("✉️\n\nContatti", "Contatti"),
]
for col, (label, page_name) in zip(cols, nav_items):
    with col:
        if st.button(label, key=f"nav_{page_name}", use_container_width=True):
            set_page(page_name)

st.divider()

# Pages
if st.session_state.page == "Home":
    st.markdown('<div class="panel"><h2>Home</h2>', unsafe_allow_html=True)
    st.write(
        """
        **OncoShape3D** è una piattaforma di morfometria tridimensionale pensata per trasformare
        i modelli STL di tumori solidi in parametri numerici oggettivi e riproducibili.

        L'obiettivo è affiancare ai dati clinico-patologici tradizionali una descrizione quantitativa
        della forma 3D, della superficie e della complessità morfologica della neoplasia.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="info-card"><h3>Dimensione</h3>Volume, superficie, diametro massimo e assi principali descrivono quanto è estesa la lesione.</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="info-card"><h3>Forma</h3>Sfericità, compattezza, elongazione e irregolarità descrivono come il tumore cresce nello spazio.</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="info-card"><h3>Ricerca</h3>Gli indici possono essere correlati a DOI, WPOI, grading, pT, pN, ENE e outcome clinici.</div>', unsafe_allow_html=True)

elif st.session_state.page == "Analisi STL":
    left, right = st.columns([0.95, 1.05], gap="large")

    with left:
        st.markdown('<div class="panel"><h2>1. Caricamento file STL</h2>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Trascina qui uno o più file STL",
            type=["stl"],
            accept_multiple_files=True,
            label_visibility="visible"
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="panel"><h2>2. Visualizzatore 3D</h2>', unsafe_allow_html=True)
        viewer_placeholder = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel"><h2>3. Risultati morfometrici</h2>', unsafe_allow_html=True)
        results_placeholder = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

    results = []
    errors = []
    first_viewer = None

    if uploaded_files:
        with st.spinner("Analisi dei file STL in corso..."):
            for uploaded in uploaded_files:
                try:
                    raw = uploaded.read()
                    metrics, points, faces = compute_metrics(uploaded.name, raw)
                    results.append(metrics)
                    if first_viewer is None:
                        first_viewer = (points, faces, uploaded.name)
                except Exception as e:
                    errors.append({"File": uploaded.name, "Errore": str(e)})

    if first_viewer:
        points, faces, filename = first_viewer
        with viewer_placeholder.container():
            st.caption(f"Anteprima 3D: {filename}")
            st.plotly_chart(make_3d_plot(points, faces), use_container_width=True)
    else:
        with viewer_placeholder.container():
            st.info("Carica un file STL per visualizzare il modello 3D.")

    if results:
        df = pd.DataFrame(results)
        with results_placeholder.container():
            st.success(f"Analisi completata: {len(df)} file elaborati correttamente.")

            if len(df) == 1:
                st.markdown(result_vertical_table(df.iloc[0]), unsafe_allow_html=True)
            else:
                selected_file = st.selectbox("Seleziona file da visualizzare", df["File"].tolist())
                row = df[df["File"] == selected_file].iloc[0]
                st.markdown(result_vertical_table(row), unsafe_allow_html=True)
                with st.expander("Tabella completa per tutti i file"):
                    st.dataframe(df, use_container_width=True)

            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="Parametri STL")

            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    "📗 Esporta Excel",
                    data=excel_buffer.getvalue(),
                    file_name="OncoShape3D_parametri_STL.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            with c2:
                st.download_button(
                    "📘 Esporta CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="OncoShape3D_parametri_STL.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    elif not uploaded_files:
        with results_placeholder.container():
            st.info("I risultati compariranno qui dopo il caricamento di uno o più STL.")

    if errors:
        st.warning("Alcuni file non sono stati elaborati.")
        st.dataframe(pd.DataFrame(errors), use_container_width=True)

elif st.session_state.page == "Metodo":
    st.markdown('<div class="panel"><h2>Metodo</h2>', unsafe_allow_html=True)
    st.write(
        """
        La piattaforma analizza la mesh triangolare contenuta nel file STL.
        A partire da vertici e facce vengono calcolati volume, superficie,
        assi principali e indici di forma.
        """
    )
    st.markdown("### Parametri dimensionali")
    st.write("- Volume\n- Superficie\n- Diametro massimo 3D\n- Assi principali")
    st.markdown("### Parametri morfologici")
    st.write("- Sfericità\n- Compattezza\n- Rapporto superficie/volume\n- Elongazione\n- Irregolarità di superficie")
    st.markdown("### Requisiti tecnici")
    st.write(
        """
        Gli STL dovrebbero essere in millimetri, rappresentare esclusivamente il volume di interesse
        ed essere preferibilmente mesh chiuse/watertight.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.page == "Disclaimer":
    st.markdown('<div class="panel"><h2>Disclaimer</h2>', unsafe_allow_html=True)
    st.warning(
        """
        OncoShape3D è uno strumento di ricerca e analisi morfometrica.
        Non è un dispositivo medico e non deve essere utilizzato come strumento autonomo
        per diagnosi, stadiazione, decisioni terapeutiche o valutazioni prognostiche individuali.
        """
    )
    st.write(
        """
        I risultati devono essere interpretati da personale qualificato e sempre integrati con dati
        clinici, radiologici e istopatologici. L'accuratezza dei parametri dipende dalla qualità
        della segmentazione, dalla correttezza del file STL e dalla chiusura della mesh.
        """
    )
    st.markdown("### Privacy")
    st.write(
        """
        Prima del caricamento, i file devono essere anonimizzati. Evitare nomi file contenenti nome,
        cognome, data di nascita o altri identificativi del paziente.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.page == "Contatti":
    st.markdown('<div class="panel"><h2>Contatti</h2>', unsafe_allow_html=True)
    st.write(
        """
        OncoShape3D è una piattaforma in sviluppo per la quantificazione della morfologia
        tridimensionale dei tumori solidi da modelli STL.
        """
    )
    st.markdown("### Sviluppi futuri")
    st.write(
        """
        - Report PDF automatico
        - Modulo clinico-patologico
        - Database multicentrico anonimizzato
        - Score morfometrico di rischio
        """
    )
    st.markdown("### Email")
    st.write("Inserire qui email istituzionale o riferimento del gruppo di ricerca.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="note-box">
        <b>Nota importante.</b><br>
        Gli STL devono essere anonimizzati e preferibilmente espressi in millimetri.
        OncoShape3D è destinato a uso di ricerca.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="footer">
        <div><b>OncoShape3D</b><br>3D Tumor Morphometry Platform</div>
        <div>Research Use Only</div>
        <div>Contatti: inserire email istituzionale</div>
    </div>
    """,
    unsafe_allow_html=True
)
