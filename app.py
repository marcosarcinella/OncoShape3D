
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
    font-size: 44px;
    font-weight: 800;
    color: #0F766E;
    margin-bottom: 0px;
}
.subtitle {
    font-size: 18px;
    color: #475569;
    margin-top: 0px;
}
.metric-card {
    padding: 18px;
    border-radius: 16px;
    background-color: #F8FAFC;
    border: 1px solid #E2E8F0;
}
.section-title {
    font-size: 26px;
    font-weight: 700;
    color: #0F172A;
    margin-top: 20px;
}
.small-note {
    font-size: 13px;
    color: #64748B;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown('<div class="main-title">OncoShape3D</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Piattaforma di morfometria 3D per analisi STL in oncologia del cavo orale</div>',
    unsafe_allow_html=True
)

st.divider()

with st.sidebar:
    st.header("OncoShape3D")
    st.write("Carica file STL in millimetri.")
    st.write("La mesh dovrebbe essere chiusa/watertight per garantire volume e sfericità affidabili.")
    st.divider()
    st.caption("Versione MVP locale")

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
    signed_volume = np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])) / 6.0
    volume = abs(float(signed_volume.sum()))

    if surface <= 0 or volume <= 0:
        raise ValueError("Volume o superficie non validi. Controllare la mesh.")

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

tab_upload, tab_about = st.tabs(["Analisi STL", "Metodo e interpretazione"])

with tab_upload:
    st.markdown('<div class="section-title">Caricamento file STL</div>', unsafe_allow_html=True)
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

with tab_about:
    st.markdown('<div class="section-title">Cosa calcola OncoShape3D</div>', unsafe_allow_html=True)
    st.write(
        """
        OncoShape3D calcola parametri quantitativi dalla geometria tridimensionale del tumore.
        I parametri dimensionali descrivono l'estensione della massa tumorale, mentre gli indici
        morfologici descrivono la forma e la complessità della superficie.
        """
    )

    st.markdown("### Parametri principali")
    st.markdown(
        """
        - **Volume mm3**: volume della mesh STL.
        - **Superficie mm2**: area superficiale totale della mesh.
        - **Sfericita**: quanto la forma si avvicina a una sfera perfetta.
        - **S/V mm-1**: rapporto superficie/volume.
        - **Compattezza**: indice derivato dalla sfericità.
        - **Diametro max 3D mm**: massima distanza tra due punti del tumore.
        - **Assi principali**: estensione lungo le tre direzioni principali.
        - **Elongazione**: asse maggiore / asse minore.
        - **Irregolarita superficie**: superficie reale / superficie della sfera equivalente.
        - **Euler, Faces, Vertices**: parametri topologici/tecnici della mesh.
        """
    )

    st.markdown("### Nota metodologica")
    st.info(
        """
        I risultati sono affidabili se lo STL è espresso in millimetri e la mesh è chiusa.
        Mesh aperte, non manifold o con errori di segmentazione possono produrre valori di volume
        e superficie non accurati.
        """
    )
