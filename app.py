import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

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

# Entrada de nodos
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

# Entrada de elementos
n_elem = st.sidebar.number_input("Cantidad de barras", min_value=1, max_value=30, value=2)
# Selección de grados de libertad fijos (condiciones de borde)
st.sidebar.markdown("### 🧷 Condiciones de contorno")
st.sidebar.markdown("Seleccioná los grados de libertad fijos.\n\nEj: 0=X del Nodo 0, 1=Y del Nodo 0, 2=X del Nodo 1, etc.")
gdl_fijos = st.sidebar.multiselect(
    "Grados de libertad fijos (nodos empotrados o apoyos)",
    options=list(range(2 * n_nodos)),
    default=[]
)

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

# Funcion de matriz de rigidez global
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

# Se ensambla la matriz global
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

st.subheader("🎯 Vector de Cargas")

# Se Ingresa el vector de fuerzas
F_global = np.zeros((2 * n_nodos, 1))
for i in range(2 * n_nodos):
    F_global[i, 0] = st.number_input(f"Fuerza en GDL {i}", value=0.0, format="%.2f", key=f"F{i}")

# Se Aplican condiciones de borde
todos_gdl = np.arange(2 * n_nodos)
gdl_libres = np.setdiff1d(todos_gdl, gdl_fijos)

# Se reducen las matrices
K_reducida = K_global[np.ix_(gdl_libres, gdl_libres)]
F_reducida = F_global[gdl_libres]

# Resolución del sistema
if K_reducida.size > 0:
    u_reducida = np.linalg.solve(K_reducida, F_reducida)
else:
    u_reducida = np.array([])

# Se reconstruye el vector de desplazamientos completo
u = np.zeros((2 * n_nodos, 1))
for idx, gdl in enumerate(gdl_libres):
    u[gdl] = u_reducida[idx]

# Calculo de reacciones: R = K_global * u - F_global
reacciones = np.dot(K_global, u) - F_global

# Se muestran los resultados
st.subheader("📌 Desplazamientos nodales")
df_u = pd.DataFrame(u, columns=["Desplazamiento [m]"])
st.dataframe(df_u.style.format(precision=4))

st.subheader("💥 Reacciones en apoyos")
df_reacciones = pd.DataFrame(reacciones, columns=["Reacción [N]"])
st.dataframe(df_reacciones.style.format(precision=2))

# BOTONES PARA DESCARGAR RESULTADOS 

# Se organizan los desplazamientos en formato por nodo
df_u_nodal = pd.DataFrame(u.reshape(-1, 2), columns=["Ux [m]", "Uy [m]"])
df_u_nodal.index = [f"Nodo {i}" for i in range(len(df_u_nodal))]

# Se organizan las reacciones en formato por nodo
df_R_nodal = pd.DataFrame(reacciones.reshape(-1, 2), columns=["Rx [N]", "Ry [N]"])
df_R_nodal.index = [f"Nodo {i}" for i in range(len(df_R_nodal))]

st.markdown("### 📤 Descargar Resultados Nodalizados")

col1, col2 = st.columns(2)

with col1:
    csv_u = df_u_nodal.to_csv().encode('utf-8')
    st.download_button(
        label="⬇️ Descargar desplazamientos",
        data=csv_u,
        file_name='desplazamientos.csv',
        mime='text/csv'
    )

with col2:
    csv_R = df_R_nodal.to_csv().encode('utf-8')
    st.download_button(
        label="⬇️ Descargar reacciones",
        data=csv_R,
        file_name='reacciones.csv',
        mime='text/csv'
    )

# Gráfico
st.markdown("""
    <div style="text-align:center; padding-top:20px;">
        <h2 style="color:#4F8BF9;">📊 Visualización de la Estructura</h2>
    </div>
""", unsafe_allow_html=True)

factor_amplificacion = st.slider("🔍 Factor de amplificación de desplazamientos", min_value=1, max_value=-500, value=-100, step=10)

# Calculo de coordenadas deformadas
coords_deformados = coords + factor_amplificacion * u.reshape(-1, 2)

fig = go.Figure()

# Estructura original
for idx, (ni, nf, _, _) in enumerate(elementos):
    x = [coords[int(ni)][0], coords[int(nf)][0]]
    y = [coords[int(ni)][1], coords[int(nf)][1]]
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name='Original', line=dict(color='red')))

# Estructura deformada
for idx, (ni, nf, _, _) in enumerate(elementos):
    x = [coords_deformados[int(ni)][0], coords_deformados[int(nf)][0]]
    y = [coords_deformados[int(ni)][1], coords_deformados[int(nf)][1]]
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name='Deformada', line=dict(color='blue', dash='dash')))

# Etiquetas de nodos originales y deformados
for i in range(n_nodos):
    x_orig, y_orig = coords[i]
    x_def, y_def = coords_deformados[i]
    fig.add_trace(go.Scatter(x=[x_orig], y=[y_orig], mode='text', text=[f"N{i}"], textposition="top center", showlegend=False))
    fig.add_trace(go.Scatter(x=[x_def], y=[y_def], mode='text', text=[f"N{i}'"], textposition="top center", showlegend=False))

fig.update_layout(
    title="Vista 2D de la estructura (interactiva)",
    xaxis_title="X [m]",
    yaxis_title="Y [m]",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    height=600
)

fig.update_yaxes(scaleanchor="x", scaleratio=1)  # Mantiene proporción 1:1
st.plotly_chart(fig, use_container_width=True)

# Botón para descargar la matriz de rigidez global
csv = df_Kglobal.to_csv(index=False).encode('utf-8')

st.download_button(
    label="⬇️ Descargar matriz global en CSV",
    data=csv,
    file_name='matriz_rigidez_global.csv',
    mime='text/csv',
    help="Hace clic para descargar la matriz ensamblada"
)

