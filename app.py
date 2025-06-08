import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Calculadora de Rigidez", layout="wide")

# ENCABEZADO PRINCIPAL
st.markdown("""
    <div style="text-align: center;">
        <h1 style="color:#4F8BF9;">🧮 Calculadora de Matriz de Rigidez</h1>
        <h3>Ingeniería Aeronautica - UTN</h3>
        <p style="font-size:18px;">Materia: Estructuras Aeronáuticas III</p>
        <p style="font-size:16px;">Alumno: <strong>Lahan, Alberto Nicolas</strong></p>
        <hr style="border:1px solid #4F8BF9">
    </div>
""", unsafe_allow_html=True)
st.markdown("Esta app te permite calcular matrices de rigidez de barras estructurales 2D de forma automática.")

st.sidebar.header("🔧 Parámetros del problema")

# Entrar nodos
n_nodos = st.sidebar.number_input("Cantidad de nodos", min_value=2, max_value=20, value=4)
st.subheader("📍 Coordenadas de los Nodos")

coords = []
for i in range(n_nodos):
    x = st.number_input(f"X Nodo {i}", key=f"x{i}")
    y = st.number_input(f"Y Nodo {i}", key=f"y{i}")
    coords.append([x, y])
coords = np.array(coords)

df_coords = pd.DataFrame(coords, columns=["X", "Y"])
st.dataframe(df_coords)

# Entrar elementos
n_elem = st.sidebar.number_input("Cantidad de barras", min_value=1, max_value=30, value=2)
st.subheader("🔩 Propiedades de las Barras")

elementos = []

for i in range(n_elem):
    with st.container():
        st.markdown(f"<h4 style='color:#4F8BF9;'>Barra {i}</h4>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            ni = st.number_input(f"Nodo inicial", min_value=0, max_value=n_nodos-1, key=f"ni{i}")
            E = st.number_input(f"Módulo de Young E (Pa)", value=2.1e11, key=f"E{i}", format="%.2e")
        with col2:
            nf = st.number_input(f"Nodo final", min_value=0, max_value=n_nodos-1, key=f"nf{i}")
            A = st.number_input(f"Área A (m²)", value=0.005, key=f"A{i}", format="%.4f")
        elementos.append([ni, nf, E, A])

# Función de matriz de rigidez global
def matriz_rigidez_global_barra(xi, yi, xj, yj, E, A):
    L = np.sqrt((xj - xi)**2 + (yj - yi)**2)
    C = (xj - xi) / L
    S = (yj - yi) / L
    k = (E * A / L) * np.array([
        [ C*C,  C*S, -C*C, -C*S],
        [ C*S,  S*S, -C*S, -S*S],
        [-C*C, -C*S,  C*C,  C*S],
        [-C*S, -S*S,  C*S,  S*S]
    ])
    return k, L

# Ensamblar matriz global
K_global = np.zeros((2 * n_nodos, 2 * n_nodos))
st.subheader("🧩 Matrices de rigidez por barra")

for idx, (ni, nf, E, A) in enumerate(elementos):
    xi, yi = coords[int(ni)]
    xj, yj = coords[int(nf)]
    k_elem, L = matriz_rigidez_global_barra(xi, yi, xj, yj, E, A)
    
    st.markdown(f"**Matriz Barra {idx}** (L = {L:.2f} m)")
    st.dataframe(pd.DataFrame(k_elem))
    
    dofs = [2*int(ni), 2*int(ni)+1, 2*int(nf), 2*int(nf)+1]
    for i in range(4):
        for j in range(4):
            K_global[dofs[i], dofs[j]] += k_elem[i, j]

st.markdown("""
    <div style="text-align:center; padding-top:20px;">
        <h2 style="color:#4F8BF9;">📐 Matriz de Rigidez Global</h2>
    </div>
""", unsafe_allow_html=True)

df_Kglobal = pd.DataFrame(np.round(K_global, decimals=2))
st.dataframe(df_Kglobal.style.background_gradient(cmap="Blues").format(precision=2))

# Gráfico
st.markdown("""
    <div style="text-align:center; padding-top:20px;">
        <h2 style="color:#4F8BF9;">📊 Visualización de la Estructura</h2>
    </div>
""", unsafe_allow_html=True)

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_facecolor("white")
ax.set_aspect("equal")

# Dibujar barras
for idx, (ni, nf, _, _) in enumerate(elementos):
    xi, yi = coords[int(ni)]
    xj, yj = coords[int(nf)]
    ax.plot([xi, xj], [yi, yj], color='steelblue', linewidth=2)

# Dibujar nodos
for i, (x, y) in enumerate(coords):
    ax.plot(x, y, 'o', markersize=8, color="#FF4B4B")
    ax.text(x + 0.05, y + 0.05, f"N{i}", fontsize=11, color="black", ha="center", va="center")

# Limpiar márgenes y ajustar eje
ax.set_title("Vista 2D de la estructura", fontsize=14, color="#333333")
ax.set_xlabel("X [m]", fontsize=12)
ax.set_ylabel("Y [m]", fontsize=12)
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_xlim(min(coords[:,0]) - 1, max(coords[:,0]) + 1)
ax.set_ylim(min(coords[:,1]) - 1, max(coords[:,1]) + 1)

st.pyplot(fig)

# Botón para descargar la matriz de rigidez global
csv = df_Kglobal.to_csv(index=False).encode('utf-8')

st.download_button(
    label="⬇️ Descargar matriz global en CSV",
    data=csv,
    file_name='matriz_rigidez_global.csv',
    mime='text/csv',
    help="Hace clic para descargar la matriz ensamblada"
)

