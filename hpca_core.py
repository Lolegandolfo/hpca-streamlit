"""
Núcleo matemático de PCA y HPCA (Avellaneda, 2019).

Lo usan la app (app.py) y el script de datos (scripts/construir_datos.py).
Las funciones son las mismas del trabajo práctico (HPCA_Merval.py / HPCA_Nikkei.py / HPCA_FTSE.py).
"""
import numpy as np
import pandas as pd

MAX_FRAC_CEROS = 0.10  # descarta acciones con >10% de días sin variación (iliquidez)
MAX_FALTANTES = 0.05   # descarta acciones con >5% de días sin precio en la ventana
WINSOR_SIGMA = 5.0     # recorta retornos diarios a ±5 desvíos (errores de precio / outliers)
MAX_CORR_DUPLICADO = 0.95  # dos tickers con correlación mayor son la misma empresa (distinta clase)
MIN_POR_SECTOR = 3     # sectores más chicos se agrupan en "Otros"
VENTANA = 252          # 1 año hábil de estimación (análisis rolling)
PASO = 21              # rebalanceo mensual


# ----------------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------------
def preparar_retornos(precios, sectores, anios, fx=None):
    """
    Log-retornos limpios y ordenados por sector para los últimos `anios` años.

    precios: DataFrame de precios en moneda local (fechas x tickers).
    sectores: dict {ticker: sector}.
    fx: serie opcional para convertir precios (precio / fx), p. ej. el CCL del Merval.
    Devuelve (retornos, etiquetas, descartadas).
    """
    inicio = precios.index.max() - pd.DateOffset(years=anios)
    p = precios.loc[precios.index >= inicio, [t for t in precios.columns if t in sectores]]
    p = p.dropna(how="all")

    # La iliquidez se mide en moneda local: convertida, un precio sin operar igual "se mueve".
    r_local = np.log(p).diff().iloc[1:]
    frac_ceros = (r_local == 0).sum() / r_local.notna().sum().clip(lower=1)
    faltantes = r_local.isna().mean()
    validas = [t for t in p.columns if frac_ceros[t] <= MAX_FRAC_CEROS and faltantes[t] < MAX_FALTANTES]
    descartadas = sorted(set(p.columns) - set(validas))

    if fx is not None:
        fechas = p.index.intersection(fx.dropna().index)
        p = p.loc[fechas].div(fx.loc[fechas], axis=0)
    r = np.log(p[validas]).diff().iloc[1:].dropna()

    # Clases de acciones de una misma empresa (GOOG/GOOGL, preferidas vs ordinarias) son casi
    # idénticas y dejan la matriz de correlación casi singular: se conserva sólo la primera.
    C = np.corrcoef(r.to_numpy(), rowvar=False)
    i, j = np.where(np.triu(C, k=1) > MAX_CORR_DUPLICADO)
    duplicadas = sorted(set(r.columns[j]))
    r = r.drop(columns=duplicadas)
    validas = list(r.columns)
    descartadas = sorted(set(descartadas) | set(duplicadas))

    # Sectores con pocas acciones se agrupan en "Otros"
    sec = pd.Series({t: sectores[t] for t in validas})
    chicos = sec.value_counts()[lambda c: c < MIN_POR_SECTOR].index
    sec = sec.where(~sec.isin(chicos), "Otros")
    if (sec == "Otros").sum() < 2:
        sec = sec[sec != "Otros"]
    # Orden: sectores de más grande a más chico, "Otros" al final
    tamanio = sec.value_counts()
    orden = sorted(sec.index, key=lambda t: (sec[t] == "Otros", -tamanio[sec[t]], sec[t]))
    r = r[orden]

    mu, sd = r.mean(), r.std()
    r = r.clip(mu - WINSOR_SIGMA * sd, mu + WINSOR_SIGMA * sd, axis=1)
    return r, sec[orden].to_numpy(), descartadas


def estandarizar(r):
    r = np.asarray(r, dtype=float)
    return (r - r.mean(axis=0)) / r.std(axis=0, ddof=1)


# ----------------------------------------------------------------------------
# PCA y HPCA
# ----------------------------------------------------------------------------
def eig_desc(A):
    """Autovalores/autovectores de una matriz simétrica, en orden decreciente."""
    val, vec = np.linalg.eigh(A)
    return val[::-1], vec[:, ::-1]


def orientar(v):
    """Fija el signo de un autovector para que la suma de sus entradas sea positiva."""
    return v if v.sum() >= 0 else -v


def hpca(X, etiquetas):
    """
    HPCA de dos niveles (Avellaneda 2019).

    X: retornos estandarizados (T x n), columnas ordenadas por sector.
    Devuelve la matriz HPCA R~ (ec. 11), rho_bar y la descomposición espectral
    analítica de la Proposición 2, con el tipo de cada autovector.
    """
    T, n = X.shape
    R = X.T @ X / (T - 1)
    sectores = list(dict.fromkeys(etiquetas))
    idx = {s: np.where(etiquetas == s)[0] for s in sectores}

    beta = np.zeros(n)
    lam1 = np.zeros(len(sectores))
    F = np.zeros((T, len(sectores)))
    W1 = np.zeros((n, len(sectores)))   # W^(1,k): EV1 de cada sector embebido en R^n (ec. 13)
    intra = []                           # (lambda^(j,k), W^(j,k), sector) para j >= 2

    for k, s in enumerate(sectores):
        ii = idx[s]
        val, vec = eig_desc(R[np.ix_(ii, ii)])
        v1 = orientar(vec[:, 0])
        lam1[k] = val[0]
        beta[ii] = np.sqrt(val[0]) * v1                  # ec. 7
        F[:, k] = X[:, ii] @ v1 / np.sqrt(val[0])        # ec. 5
        W1[ii, k] = v1
        for j in range(1, len(ii)):
            w = np.zeros(n)
            w[ii] = vec[:, j]
            intra.append((val[j], w, s))

    rho_bar = np.corrcoef(F, rowvar=False)
    sec_id = np.array([sectores.index(s) for s in etiquetas])
    mismo = sec_id[:, None] == sec_id[None, :]
    R_tilde = np.where(mismo, R, np.outer(beta, beta) * rho_bar[np.ix_(sec_id, sec_id)])  # ec. 11

    # Proposición 2: espectro analítico
    M = np.sqrt(np.outer(lam1, lam1)) * rho_bar           # ec. 15
    mu, alpha = eig_desc(M)
    alpha[:, 0] = orientar(alpha[:, 0])
    espectro = [(mu[k], W1 @ alpha[:, k], "Multi-sector", alpha[:, k]) for k in range(len(sectores))]  # ec. 16
    espectro += [(lam, w, s, None) for lam, w, s in intra]
    espectro.sort(key=lambda e: -e[0])

    return {
        "R": R, "R_tilde": R_tilde, "rho_bar": rho_bar, "sectores": sectores,
        "valores": np.array([e[0] for e in espectro]),
        "vectores": np.column_stack([e[1] for e in espectro]),
        "tipo": [e[2] for e in espectro],
        "alpha": [e[3] for e in espectro],
    }


def describir(tipo, alpha, sectores, umbral=0.3):
    """Interpretación en palabras de un eigenportfolio HPCA."""
    if tipo != "Multi-sector":
        return f"Long-short dentro de {tipo}"
    if np.all(alpha > 0):
        return "Mercado (long en todos los sectores)"
    largos = " + ".join(sectores[i] for i in np.where(alpha > umbral)[0])
    cortos = " + ".join(sectores[i] for i in np.where(alpha < -umbral)[0])
    return f"{largos or 'el resto'} vs {cortos or 'el resto'}"


def analizar(retornos, etiquetas, n_ev=5):
    """PCA vs HPCA sobre toda la ventana: espectros, autovectores alineados y métricas."""
    X = estandarizar(retornos)
    T, n = X.shape
    h = hpca(X, etiquetas)
    lam_pca, V_pca = eig_desc(h["R"])
    V_pca, V_hpca = V_pca[:, :n_ev].copy(), h["vectores"][:, :n_ev].copy()
    V_pca[:, 0], V_hpca[:, 0] = orientar(V_pca[:, 0]), orientar(V_hpca[:, 0])
    for j in range(1, n_ev):
        if V_hpca[:, j] @ V_pca[:, j] < 0:  # alinea signos para comparar
            V_pca[:, j] *= -1
    lam_mp = (1 + np.sqrt(n / T)) ** 2
    return {
        "R": h["R"], "R_tilde": h["R_tilde"], "rho_bar": h["rho_bar"], "sectores": h["sectores"],
        "lam_pca": lam_pca, "lam_hpca": h["valores"], "V_pca": V_pca, "V_hpca": V_hpca,
        "similitud": np.abs(np.sum(V_pca * V_hpca, axis=0)),
        "descripcion": [describir(t, a, h["sectores"]) for t, a in zip(h["tipo"][:n_ev], h["alpha"][:n_ev])],
        "lam_mp": lam_mp, "factores_mp": int((lam_pca > lam_mp).sum()),
        "T": T, "n": n,
    }


# ----------------------------------------------------------------------------
# Cartera de mínima varianza out-of-sample
# ----------------------------------------------------------------------------
def corr_pca_truncada(R, m):
    """m factores PCA + varianza idiosincrática diagonal (modelo factorial de la ec. 8)."""
    val, vec = eig_desc(R)
    C = (vec[:, :m] * val[:m]) @ vec[:, :m].T
    np.fill_diagonal(C, 1.0)
    return C


def pesos_min_var(Sigma):
    w = np.linalg.solve(Sigma, np.ones(len(Sigma)))
    return w / w.sum()


def rotacion_min_var(retornos, etiquetas, ventana=VENTANA, paso=PASO):
    """
    Rebalanceo mensual de la cartera de mínima varianza estimada con PCA (m = b factores)
    y con HPCA. Devuelve (DataFrame mensual con rotación Σ|Δw| y exposición bruta, retornos diarios).
    """
    r = retornos.to_numpy()
    T, n = r.shape
    m = len(set(etiquetas))
    filas, rend = [], {"PCA": [], "HPCA": []}
    previos = {"PCA": None, "HPCA": None}
    for ini in range(ventana, T - paso + 1, paso):
        w_ret = r[ini - ventana:ini]
        sd = w_ret.std(axis=0, ddof=1)
        Xw = estandarizar(w_ret)
        Rw = Xw.T @ Xw / (len(Xw) - 1)
        estim = {"PCA": corr_pca_truncada(Rw, m), "HPCA": hpca(Xw, etiquetas)["R_tilde"]}
        fila = {"fecha": retornos.index[ini]}
        for nombre, C in estim.items():
            w = pesos_min_var(C * np.outer(sd, sd))
            rend[nombre].append(r[ini:ini + paso] @ w)
            fila[f"Exposición {nombre}"] = np.abs(w).sum()
            fila[f"Rotación {nombre}"] = np.nan if previos[nombre] is None else np.abs(w - previos[nombre]).sum()
            previos[nombre] = w
        filas.append(fila)
    mensual = pd.DataFrame(filas).set_index("fecha")
    fechas = retornos.index[ventana:ventana + paso * len(filas)]
    diarios = pd.DataFrame({k: np.concatenate(v) for k, v in rend.items()}, index=fechas)
    return mensual, diarios
