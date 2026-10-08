"""
Hierarchical PCA en los mercados del mundo — app de Streamlit.

Recrea el análisis de Avellaneda (2019), "Hierarchical PCA and Applications to Portfolio
Management", sobre los principales índices bursátiles. Los datos se generan con
scripts/construir_datos.py y se leen de data/.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import hpca_core as hc

st.set_page_config(page_title="HPCA · Mercados del mundo", page_icon="📊", layout="wide")

DATA = Path(__file__).resolve().parent / "data"
COLOR_PCA = "#eb6834"
COLOR_HPCA = "#2a78d6"
N_EV = 5
MAX_N_HEATMAP_DECIMAL = 120  # por encima, el heatmap se envía comprimido
PAPER = "https://arxiv.org/abs/1910.02310"


# ----------------------------------------------------------------------------
# Carga y cálculo (cacheados)
# ----------------------------------------------------------------------------
@st.cache_data
def cargar_meta():
    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
    resumen = json.loads((DATA / "resumen.json").read_text(encoding="utf-8"))
    disponibles = {k: v for k, v in meta["indices"].items() if (DATA / f"{k}_precios.parquet").exists()}
    return meta, resumen, disponibles


# cache_resource devuelve el mismo objeto sin copiarlo en cada interacción (cache_data lo
# des-serializa cada vez, y para el S&P 500 son decenas de MB). Ninguna función los modifica.
@st.cache_resource(max_entries=20)
def cargar_indice(clave):
    precios = pd.read_parquet(DATA / f"{clave}_precios.parquet").astype(float)
    sectores = json.loads((DATA / f"{clave}_sectores.json").read_text(encoding="utf-8"))
    archivo_fx = DATA / f"{clave}_fx.parquet"
    fx = pd.read_parquet(archivo_fx).iloc[:, 0] if archivo_fx.exists() else None
    return precios, sectores, fx


@st.cache_resource(show_spinner="Calculando PCA y HPCA…", max_entries=40)
def calcular(clave, anios):
    precios, sectores, fx = cargar_indice(clave)
    r, etiquetas, descartadas = hc.preparar_retornos(precios, sectores, anios, fx)
    a = hc.analizar(r, etiquetas, N_EV)
    a.update(tickers=list(r.columns), etiquetas=etiquetas, descartadas=descartadas,
             inicio=r.index[0].date(), fin=r.index[-1].date())
    return a


@st.cache_resource(show_spinner="Simulando el rebalanceo mensual de la cartera de mínima varianza…",
                   max_entries=40)
def calcular_rotacion(clave, anios):
    precios, sectores, fx = cargar_indice(clave)
    r, etiquetas, _ = hc.preparar_retornos(precios, sectores, anios, fx)
    return hc.rotacion_min_var(r, etiquetas)


def bloques(etiquetas):
    """[(sector, inicio, fin)] de cada bloque contiguo de acciones del mismo sector."""
    out, inicio = [], 0
    for i in range(1, len(etiquetas) + 1):
        if i == len(etiquetas) or etiquetas[i] != etiquetas[inicio]:
            out.append((etiquetas[inicio], inicio, i))
            inicio = i
    return out


# ----------------------------------------------------------------------------
# Gráficos
# ----------------------------------------------------------------------------
def fig_heatmap(A, a, modo):
    n = a["n"]
    etiquetas_ejes = [f"{t} · {s}" for t, s in zip(a["tickers"], a["etiquetas"])]
    if modo == "Diferencia":
        escala, zmin, zmax, titulo_barra = "RdBu_r", -0.3, 0.3, "R − R̃"
    else:
        escala, zmin, zmax, titulo_barra = "Blues", 0, 1, "ρ"
    if n > MAX_N_HEATMAP_DECIMAL:
        # Matrices grandes: se envían en centésimos como enteros de 1 byte (8 veces menos datos
        # que floats). Para el S&P 500 pasa de ~3 MB a ~0,4 MB, clave para que cargue en el celular.
        z = np.clip(np.round(A * 100), -127, 127).astype(np.int8)
        zmin, zmax = zmin * 100, zmax * 100
        ticks = np.linspace(zmin, zmax, 5)
        colorbar = dict(title=titulo_barra, thickness=12, tickvals=ticks, ticktext=[f"{t / 100:.2f}" for t in ticks])
        hover = "%{y}<br>%{x}<br><b>%{z}</b> (centésimos)<extra></extra>"
    else:
        z, colorbar = np.round(A, 3).astype(np.float32), dict(title=titulo_barra, thickness=12)
        hover = "%{y}<br>%{x}<br><b>%{z:.2f}</b><extra></extra>"
    fig = go.Figure(go.Heatmap(
        z=z, x=etiquetas_ejes, y=etiquetas_ejes, colorscale=escala, zmin=zmin, zmax=zmax,
        zmid=0 if modo == "Diferencia" else None, colorbar=colorbar, hovertemplate=hover,
    ))
    linea = dict(type="line", line=dict(color="rgba(20,20,20,0.55)", width=0.8))
    for _, ini, fin in bloques(a["etiquetas"])[:-1]:
        fig.add_shape(**linea, x0=fin - 0.5, x1=fin - 0.5, y0=-0.5, y1=n - 0.5)
        fig.add_shape(**linea, y0=fin - 0.5, y1=fin - 0.5, x0=-0.5, x1=n - 0.5)
    mostrar = n <= 60
    fig.update_xaxes(showticklabels=mostrar, tickfont_size=9, showgrid=False)
    fig.update_yaxes(showticklabels=mostrar, tickfont_size=9, autorange="reversed", showgrid=False,
                     scaleanchor="x")
    fig.update_layout(height=720, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def fig_rho_bar(a):
    s = a["sectores"]
    fig = go.Figure(go.Heatmap(
        z=np.round(a["rho_bar"], 2), x=s, y=s, colorscale="Blues", zmin=0, zmax=1,
        text=np.round(a["rho_bar"], 2), texttemplate="%{text:.2f}" if len(s) <= 14 else None,
        colorbar=dict(title="ρ̄", thickness=12),
        hovertemplate="%{y} × %{x}<br><b>ρ̄ = %{z:.2f}</b><extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", tickfont_size=10)
    fig.update_xaxes(tickangle=-45, tickfont_size=10)
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def fig_autovectores(a):
    n, tickers, etiq = a["n"], a["tickers"], a["etiquetas"]
    titulos = [f"<b>EV{j + 1}</b> · similitud coseno {a['similitud'][j]:.2f} · HPCA: {a['descripcion'][j]}"
               for j in range(N_EV)]
    fig = make_subplots(rows=N_EV, cols=1, shared_xaxes=True, vertical_spacing=0.045,
                        subplot_titles=titulos)
    x = np.arange(n)
    custom = np.column_stack([tickers, etiq])
    modo = "lines+markers" if n <= 80 else "lines"
    for j in range(N_EV):
        for nombre, V, color in [("PCA", a["V_pca"], COLOR_PCA), ("HPCA", a["V_hpca"], COLOR_HPCA)]:
            fig.add_trace(go.Scatter(
                x=x, y=V[:, j], mode=modo, name=nombre, legendgroup=nombre, showlegend=j == 0,
                line=dict(color=color, width=2 if nombre == "HPCA" else 1.5),
                marker=dict(size=5), customdata=custom,
                hovertemplate=f"<b>%{{customdata[0]}}</b> · %{{customdata[1]}}<br>{nombre}: %{{y:.3f}}<extra></extra>",
            ), row=j + 1, col=1)
        fig.add_hline(y=0, line=dict(color="rgba(120,120,120,0.6)", width=1), row=j + 1, col=1)
    centros, nombres = [], []
    for k, (sector, ini, fin) in enumerate(bloques(etiq)):
        if k % 2 == 0:
            fig.add_vrect(x0=ini - 0.5, x1=fin - 0.5, fillcolor="rgba(128,128,128,0.10)",
                          line_width=0, layer="below", row="all", col=1)
        centros.append((ini + fin - 1) / 2)
        nombres.append(sector)
    fig.update_xaxes(tickvals=centros, ticktext=nombres, tickangle=-40, tickfont_size=10,
                     showgrid=False, row=N_EV, col=1)
    fig.update_annotations(font_size=12, x=0, xanchor="left")
    fig.update_layout(height=230 * N_EV, hovermode="closest",
                      legend=dict(orientation="h", y=1.04, x=1, xanchor="right"),
                      margin=dict(l=10, r=10, t=60, b=10))
    return fig


def fig_varianza(a):
    k = np.arange(1, a["n"] + 1)
    fig = go.Figure()
    for nombre, lam, color in [("PCA", a["lam_pca"], COLOR_PCA), ("HPCA", a["lam_hpca"], COLOR_HPCA)]:
        fig.add_trace(go.Scatter(x=k, y=np.cumsum(lam) / a["n"], name=nombre, line=dict(color=color, width=2.5),
                                 hovertemplate=f"{nombre}: %{{y:.1%}} con %{{x}} factores<extra></extra>"))
    fig.update_yaxes(tickformat=".0%", range=[0, 1.02], title="Varianza explicada acumulada")
    fig.update_xaxes(title="Cantidad de factores")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", y=1.08, x=1, xanchor="right"))
    return fig


def fig_rotacion_tiempo(mensual):
    fig = go.Figure()
    for nombre, color in [("PCA", COLOR_PCA), ("HPCA", COLOR_HPCA)]:
        fig.add_trace(go.Scatter(x=mensual.index, y=mensual[f"Rotación {nombre}"], name=nombre,
                                 mode="lines+markers", line=dict(color=color, width=2), marker=dict(size=4),
                                 hovertemplate=f"{nombre}: %{{y:.2f}}<br>%{{x|%b %Y}}<extra></extra>"))
    fig.update_yaxes(title="Rotación mensual Σ|Δw|", rangemode="tozero")
    fig.update_layout(height=380, hovermode="x unified", margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", y=1.08, x=1, xanchor="right"))
    return fig


def fig_rotacion_mercados(resumen, disponibles, actual, log):
    claves = [k for k in disponibles if k in resumen]
    claves.sort(key=lambda k: resumen[k]["rotacion_hpca"])
    nombres = [disponibles[k]["nombre"].split(" · ")[0] for k in claves]
    fig = go.Figure()
    for nombre, campo, color in [("PCA", "rotacion_pca", COLOR_PCA), ("HPCA", "rotacion_hpca", COLOR_HPCA)]:
        fig.add_trace(go.Bar(
            x=nombres, y=[resumen[k][campo] for k in claves], name=nombre, marker_color=color,
            marker_line=dict(width=[3 if k == actual else 0 for k in claves], color="#111"),
            customdata=[[resumen[k]["n"], resumen[k]["sectores"]] for k in claves],
            hovertemplate=f"<b>%{{x}}</b><br>{nombre}: %{{y:.2f}}"
                          "<br>%{customdata[0]} acciones · %{customdata[1]} sectores<extra></extra>",
        ))
    fig.update_yaxes(title="Rotación mensual Σ|Δw|", type="log" if log else "linear")
    fig.update_layout(barmode="group", bargap=0.25, height=420, margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", y=1.08, x=1, xanchor="right"))
    return fig


# ----------------------------------------------------------------------------
# Interfaz
# ----------------------------------------------------------------------------
meta, resumen, disponibles = cargar_meta()

with st.sidebar:
    st.header("Configuración")
    clave = st.selectbox("Índice", list(disponibles), format_func=lambda k: disponibles[k]["nombre"],
                         index=list(disponibles).index("sp500") if "sp500" in disponibles else 0)
    anios = st.slider("Años de historia", min_value=3, max_value=10, value=5,
                      help="Ventana de retornos diarios usada para estimar las correlaciones.")
    st.caption(f"Moneda: **{disponibles[clave]['moneda']}**")
    st.divider()
    st.caption(
        f"Datos al {meta['actualizado']} (Yahoo Finance). Sectores: tablas de Wikipedia (GICS cuando "
        "está disponible); Merval, Nikkei y FTSE usan la partición sectorial del trabajo práctico."
    )
    st.caption(f"Método: [Avellaneda (2019), *Hierarchical PCA and Applications to Portfolio Management*]({PAPER}).")

st.title("Hierarchical PCA en los mercados del mundo")
st.markdown(
    "¿Se puede describir el riesgo de un mercado con una estructura **por sectores** sin perder información "
    "respecto del PCA tradicional? Elegí un índice y compará **PCA** "
    f"<span style='color:{COLOR_PCA}'>■</span> contra **HPCA** <span style='color:{COLOR_HPCA}'>■</span>.",
    unsafe_allow_html=True,
)

with st.expander("¿Qué es PCA y qué es HPCA?"):
    st.markdown(
        "- **PCA** resume los retornos de muchas acciones en pocos portfolios no correlacionados, ordenados "
        "por el riesgo que explican. Cada autovector es un portfolio: pesos positivos se compran y negativos "
        "se venden. El primero suele ser *el mercado*; los siguientes mezclan sectores y cuesta interpretarlos.\n"
        "- **HPCA** hace primero un PCA dentro de cada sector y después un segundo PCA entre sectores, suponiendo "
        "que dos acciones de sectores distintos sólo se relacionan a través de sus sectores. Cada portfolio "
        "resultante se lee como *mercado*, *un sector contra otro* o *acciones de un mismo sector entre sí*.\n"
        "- **Similitud coseno**: 1 indica que el autovector de PCA y el de HPCA son el mismo portfolio; "
        "0, que no se parecen."
    )

a = calcular(clave, anios)

c = st.columns(6)
c[0].metric("Acciones", a["n"])
c[1].metric("Sectores", len(a["sectores"]))
c[2].metric("Días de datos", a["T"], help=f"{a['inicio']} a {a['fin']}")
c[3].metric("EV1 · PCA", f"{a['lam_pca'][0] / a['n']:.1%}", help="Varianza explicada por el primer portfolio.")
c[4].metric("EV1 · HPCA", f"{a['lam_hpca'][0] / a['n']:.1%}",
            delta=f"{(a['lam_hpca'][0] - a['lam_pca'][0]) / a['n'] * 100:.2f} pp", delta_color="off")
c[5].metric("Factores sobre ruido", a["factores_mp"],
            help=f"Autovalores de la matriz empírica por encima de la cota de Marchenko-Pastur λ⁺ = {a['lam_mp']:.2f}.")

# Selector en vez de st.tabs: las pestañas calculan y envían todo su contenido aunque no se vean;
# así sólo se procesa la sección visible (mucho más liviano en el celular).
SECCIONES = ["📈 Autovectores", "🟦 Correlaciones", "🔄 Rotación", "🗂️ Sectores"]
seccion = st.segmented_control("Sección", SECCIONES, default=SECCIONES[0], key="seccion",
                               label_visibility="collapsed") or SECCIONES[0]

if seccion == SECCIONES[0]:
    st.markdown(
        "Cada punto es una acción (eje x, agrupadas por sector) y su altura es el **peso** de esa acción en el "
        "autovector. HPCA forma *escalones* por sector; PCA suele ser una versión ruidosa del mismo patrón. "
        "Pasá el mouse para ver el ticker y hacé clic en la leyenda para ocultar una serie."
    )
    st.plotly_chart(fig_autovectores(a), width="stretch", theme="streamlit")
    tabla = pd.DataFrame({
        "Varianza PCA": a["lam_pca"][:N_EV] / a["n"],
        "Varianza HPCA": a["lam_hpca"][:N_EV] / a["n"],
        "Similitud coseno": a["similitud"],
        "Interpretación HPCA": a["descripcion"],
    }, index=[f"EV{j + 1}" for j in range(N_EV)])
    st.dataframe(tabla, width="stretch", column_config={
        "Varianza PCA": st.column_config.NumberColumn(format="percent"),
        "Varianza HPCA": st.column_config.NumberColumn(format="percent"),
        "Similitud coseno": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
    })
    with st.expander("Varianza explicada acumulada"):
        st.plotly_chart(fig_varianza(a), width="stretch", theme="streamlit")
        st.caption("PCA siempre queda por encima: por construcción es el óptimo. "
                   "La pregunta es cuánto se pierde al imponer la estructura sectorial.")

if seccion == SECCIONES[1]:
    modo = st.segmented_control("Matriz", ["Empírica", "HPCA", "Diferencia"], default="Empírica")
    modo = modo or "Empírica"
    fuera = a["etiquetas"][:, None] != a["etiquetas"][None, :]
    col_izq, col_der = st.columns([3, 2])
    with col_izq:
        A = {"Empírica": a["R"], "HPCA": a["R_tilde"], "Diferencia": a["R"] - a["R_tilde"]}[modo]
        st.plotly_chart(fig_heatmap(A, a, modo), width="stretch", theme="streamlit")
    with col_der:
        st.markdown("**Correlación entre sectores (ρ̄)**")
        st.plotly_chart(fig_rho_bar(a), width="stretch", theme="streamlit")
        m1, m2 = st.columns(2)
        m1.metric("Correlación media entre sectores", f"{a['R'][fuera].mean():.2f}")
        m2.metric("Error medio de HPCA fuera de bloque", f"{np.abs(a['R'] - a['R_tilde'])[fuera].mean():.3f}")
    st.caption(
        "Las líneas separan sectores. Dentro de cada bloque HPCA usa la correlación empírica; entre sectores la "
        "reconstruye como βᵢ·βⱼ·ρ̄ (ec. 11 del paper). En *Diferencia*, lo que no es blanco es la correlación "
        "que HPCA asume nula."
    )

if seccion == SECCIONES[2]:
    st.markdown(
        "Cartera de **mínima varianza** rebalanceada mes a mes: se estima la matriz de riesgo con el último año "
        "de datos y se mantiene la cartera durante el mes siguiente. La **rotación** Σ|Δw| mide cuánto cambian "
        "los pesos en cada rebalanceo: más rotación implica más costos de transacción y una estimación más inestable."
    )
    mensual, diarios = calcular_rotacion(clave, anios)
    k = st.columns(4)
    k[0].metric("Rotación media · PCA", f"{mensual['Rotación PCA'].mean():.2f}")
    k[1].metric("Rotación media · HPCA", f"{mensual['Rotación HPCA'].mean():.2f}",
                delta=f"{mensual['Rotación HPCA'].mean() / mensual['Rotación PCA'].mean() - 1:+.0%} vs PCA",
                delta_color="inverse")
    k[2].metric("Volatilidad anual · PCA", f"{diarios['PCA'].std() * np.sqrt(252):.1%}")
    k[3].metric("Volatilidad anual · HPCA", f"{diarios['HPCA'].std() * np.sqrt(252):.1%}")
    st.plotly_chart(fig_rotacion_tiempo(mensual), width="stretch", theme="streamlit")

    st.subheader("Comparación entre mercados")
    log = st.toggle("Escala logarítmica", value=False)
    st.plotly_chart(fig_rotacion_mercados(resumen, disponibles, clave, log), width="stretch", theme="streamlit")
    st.caption(
        f"Rotación media con los últimos {meta['anios_resumen']} años de cada índice, calculada al "
        f"{meta['actualizado']}. El índice seleccionado aparece con borde. PCA usa tantos factores como sectores "
        "tiene el índice, más una varianza propia por acción."
    )

if seccion == SECCIONES[3]:
    sec = pd.Series(a["etiquetas"], index=a["tickers"])
    tabla_sec = (sec.groupby(sec, sort=False).apply(lambda s: ", ".join(s.index))
                 .rename("Acciones").to_frame())
    tabla_sec.insert(0, "Cantidad", sec.value_counts().reindex(tabla_sec.index))
    st.dataframe(tabla_sec, width="stretch")
    if a["descartadas"]:
        st.caption(f"Descartadas por iliquidez o datos faltantes en la ventana ({len(a['descartadas'])}): "
                   + ", ".join(a["descartadas"]))
    st.caption("Los sectores con menos de 3 acciones se agrupan en *Otros* para que el PCA por sector tenga sentido.")
