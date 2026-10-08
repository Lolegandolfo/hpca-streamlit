# %% [markdown]
# # Construcción de datos para la app
#
# Descarga, para cada índice:
# 1. Constituyentes y sector de cada acción (Wikipedia; para Merval, Nikkei y FTSE se usan las
#    particiones sectoriales del trabajo práctico, en `universos_tp.json`).
# 2. Precios de cierre ajustados desde 2015 (yfinance).
# 3. Un resumen comparativo entre mercados (rotación de la cartera de mínima varianza PCA vs HPCA).
#
# Todo queda en `data/`, que se commitea: la app no descarga nada en el deploy.
#
# Uso:  uv run python scripts/construir_datos.py            (todos los índices)
#       uv run python scripts/construir_datos.py sp500 dax  (sólo algunos)

# %% Imports y configuración
import io
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
import hpca_core as hc  # noqa: E402

DATA = RAIZ / "data"
DATA.mkdir(exist_ok=True)
INICIO = "2015-01-01"
ANIOS_RESUMEN = 5
UA = {"User-Agent": "hpca-streamlit/0.1 (proyecto educativo)"}
WIKI = "https://en.wikipedia.org/wiki/"

# Sectores comunes (GICS, en español). Las fuentes usan taxonomías distintas y se traducen acá.
EXACTOS = {
    "it": "Tecnología",
    "power": "Utilities",
    "holding": "Financiero",
    "finance": "Financiero",
    "properties": "Inmobiliario",
    "commerce & industry": "Comercio e Industria",   # sub-índice propio del Hang Seng
    "energy & chemicals": "Energía y Químicos",      # sector propio del KOSPI 200
    "heavy industries": "Industriales",
    "steels & materials": "Materiales",
    "constructions": "Industriales",
    "services": "Industriales",                      # Nifty: puertos y logística
    "consumer goods": "Consumo básico",
    "distribution": "Industriales",
    "inspection and certification": "Industriales",
    "communication": "Comunicaciones",
    # Industrias de la lista de B3
    "cashback": "Financiero", "payment system": "Financiero", "stock exchange": "Financiero",
    "conglomerate": "Industriales", "rental car": "Industriales", "waste management": "Industriales",
    "cosmetics": "Consumo básico", "drugstore": "Consumo básico",
    "department store": "Consumo discrecional", "travel and tourism": "Consumo discrecional",
    "internet services": "Tecnología", "shopping malls": "Inmobiliario", "wood": "Materiales",
}
PALABRAS = [  # (palabra clave, sector) en orden de prioridad
    ("real estate", "Inmobiliario"), ("propert", "Inmobiliario"),
    ("utilit", "Utilities"), ("water", "Utilities"),
    ("financ", "Financiero"), ("bank", "Financiero"), ("insur", "Financiero"),
    ("health", "Salud"), ("pharma", "Salud"), ("biotech", "Salud"), ("medic", "Salud"),
    ("communication", "Comunicaciones"), ("telecom", "Comunicaciones"), ("media", "Comunicaciones"),
    ("technology", "Tecnología"), ("software", "Tecnología"), ("semicond", "Tecnología"),
    ("consumer staples", "Consumo básico"), ("consumer defensive", "Consumo básico"),
    ("fast moving", "Consumo básico"), ("food", "Consumo básico"), ("beverage", "Consumo básico"),
    ("tobacco", "Consumo básico"), ("household", "Consumo básico"),
    ("consumer", "Consumo discrecional"), ("automo", "Consumo discrecional"),
    ("apparel", "Consumo discrecional"), ("clothing", "Consumo discrecional"),
    ("retail", "Consumo discrecional"), ("commerce", "Consumo discrecional"),
    ("hotel", "Consumo discrecional"), ("education", "Consumo discrecional"),
    ("oil", "Energía"), ("energy", "Energía"), ("coal", "Energía"), ("petro", "Energía"),
    ("material", "Materiales"), ("chemic", "Materiales"), ("metal", "Materiales"),
    ("mining", "Materiales"), ("steel", "Materiales"), ("siderurg", "Materiales"),
    ("paper", "Materiales"), ("pulp", "Materiales"), ("cement", "Materiales"),
    ("industr", "Industriales"), ("aerospace", "Industriales"), ("defen", "Industriales"),
    ("machin", "Industriales"), ("airline", "Industriales"), ("transport", "Industriales"),
    ("logistic", "Industriales"), ("construct", "Industriales"), ("engineer", "Industriales"),
    ("capital goods", "Industriales"), ("railway", "Industriales"), ("shipping", "Industriales"),
]


def a_sector(etiqueta):
    e = str(etiqueta).strip().lower()
    if e in EXACTOS:
        return EXACTOS[e]
    for clave, sector in PALABRAS:
        if clave in e:
            return sector
    return None


def tabla_wiki(pagina, col_ticker, col_sector):
    """Primera tabla de la página que tenga columnas que empiecen con col_ticker y col_sector."""
    html = requests.get(WIKI + pagina, headers=UA, timeout=30).text
    for t in pd.read_html(io.StringIO(html)):
        cols = [str(c) for c in t.columns]
        ct = next((c for c in cols if c.startswith(col_ticker)), None)
        cs = next((c for c in cols if c.startswith(col_sector)), None)
        if ct and cs and len(t) >= 20:
            t.columns = cols
            return t[[ct, cs]].dropna().set_axis(["ticker", "etiqueta"], axis=1)
    raise ValueError(f"No encontré la tabla en {pagina}")


# %% Definición de índices
# clave: (nombre, moneda, fuente)
# fuente: ("tp", clave_en_universos_tp) o ("wiki", página, col_ticker, col_sector, función de ticker)
INDICES = {
    "sp500": ("S&P 500 · EE.UU.", "USD",
              ("wiki", "List_of_S%26P_500_companies", "Symbol", "GICS Sector", lambda t: t.replace(".", "-"))),
    "sx5e": ("Euro Stoxx 50 · Eurozona", "EUR",
             ("wiki", "EURO_STOXX_50", "Ticker", "Sector", lambda t: t)),
    "dax": ("DAX 40 · Alemania", "EUR",
            ("wiki", "DAX", "Ticker", "Prime Standard Sector", lambda t: t)),
    "cac": ("CAC 40 · Francia", "EUR",
            ("wiki", "CAC_40", "Ticker", "Sector", lambda t: t)),
    "ftse": ("FTSE 100/250 · Reino Unido", "GBP", ("tp", "ftse")),
    "nikkei": ("Nikkei 225 · Japón", "JPY", ("tp", "nikkei")),
    "hsi": ("Hang Seng · Hong Kong", "HKD",
            ("wiki", "Hang_Seng_Index", "Ticker", "Sub-index",
             lambda t: f"{int(t.split(':')[-1].strip()):04d}.HK")),
    "kospi": ("KOSPI 200 · Corea del Sur", "KRW",
              ("wiki", "KOSPI_200", "Symbol", "GICS Sector", lambda t: f"{t.zfill(6)}.KS")),
    "nifty": ("Nifty 50 · India", "INR",
              ("wiki", "NIFTY_50", "Symbol", "Sector", lambda t: f"{t}.NS")),
    "asx": ("S&P/ASX 50 · Australia", "AUD",
            ("wiki", "S%26P/ASX_50", "Symbol", "Sector", lambda t: f"{t}.AX")),
    "tsx": ("S&P/TSX 60 · Canadá", "CAD",
            ("wiki", "S%26P/TSX_60", "Symbol", "Sector", lambda t: f"{t.replace('.', '-')}.TO")),
    "b3": ("B3 (principales) · Brasil", "BRL",
           ("wiki", "List_of_companies_listed_on_B3", "Ticker", "Industry",
            lambda t: f"{t.split(':')[-1].strip()}.SA")),
    "merval": ("Merval · Argentina", "USD (CCL)", ("tp", "merval")),
}


def constituyentes(clave):
    """dict {ticker_yahoo: sector}."""
    fuente = INDICES[clave][2]
    if fuente[0] == "tp":
        universos = json.loads((RAIZ / "scripts" / "universos_tp.json").read_text(encoding="utf-8"))
        return {t: s for s, ts in universos[fuente[1]].items() for t in ts}
    _, pagina, col_t, col_s, fmt = fuente
    t = tabla_wiki(pagina, col_t, col_s)
    out, sin_mapear = {}, set()
    for _, fila in t.iterrows():
        sector = a_sector(fila["etiqueta"])
        if sector is None:
            sin_mapear.add(fila["etiqueta"])
            sector = "Otros"
        out[fmt(str(fila["ticker"]).strip())] = sector
    if sin_mapear:
        print(f"  [{clave}] sectores sin mapear (→ Otros): {sorted(sin_mapear)}")
    return out


# %% Descarga de precios
def descargar(tickers, lote=40, reintentos=3):
    partes, pendientes = [], list(dict.fromkeys(tickers))
    for _ in range(reintentos):
        siguientes = []
        for i in range(0, len(pendientes), lote):
            grupo = pendientes[i:i + lote]
            d = yf.download(grupo, start=INICIO, auto_adjust=True, progress=False, threads=True)["Close"]
            if isinstance(d, pd.Series):
                d = d.to_frame(grupo[0])
            ok = [t for t in grupo if t in d.columns and d[t].notna().sum() > 0]
            partes.append(d[ok])
            siguientes += [t for t in grupo if t not in ok]
            time.sleep(1.5)
        pendientes = siguientes
        if not pendientes:
            break
        time.sleep(10)
    if pendientes:
        print(f"  no se pudieron bajar ({len(pendientes)}): {pendientes[:15]}{' …' if len(pendientes) > 15 else ''}")
    precios = pd.concat(partes, axis=1)
    precios = precios.loc[:, ~precios.columns.duplicated()].sort_index()
    return precios.astype("float32")


def ccl_merval():
    """CCL implícito: mediana de GGAL (1 ADR = 10 acciones) e YPF (1 ADR = 1 acción)."""
    p = descargar(["GGAL.BA", "GGAL", "YPFD.BA", "YPF"]).astype(float)
    ccl = pd.concat({"GGAL": p["GGAL.BA"] * 10 / p["GGAL"], "YPF": p["YPFD.BA"] / p["YPF"]}, axis=1)
    return ccl.median(axis=1).dropna().rename("ccl")


# %% Construcción
def construir(clave):
    nombre, moneda, _ = INDICES[clave]
    print(f"\n== {nombre}")
    sectores = constituyentes(clave)
    precios = descargar(list(sectores))
    precios.to_parquet(DATA / f"{clave}_precios.parquet")
    sectores = {t: s for t, s in sectores.items() if t in precios.columns}
    (DATA / f"{clave}_sectores.json").write_text(json.dumps(sectores, ensure_ascii=False, indent=1), encoding="utf-8")
    fx = None
    if clave == "merval":
        fx = ccl_merval()
        fx.to_frame().to_parquet(DATA / "merval_fx.parquet")
    print(f"  {precios.shape[1]} acciones con precios, {precios.shape[0]} fechas")
    return resumir(clave)


def resumir(clave):
    """Resumen comparativo con los últimos ANIOS_RESUMEN años, a partir de los datos ya guardados."""
    nombre, moneda, _ = INDICES[clave]
    precios = pd.read_parquet(DATA / f"{clave}_precios.parquet").astype(float)
    sectores = json.loads((DATA / f"{clave}_sectores.json").read_text(encoding="utf-8"))
    archivo_fx = DATA / f"{clave}_fx.parquet"
    fx = pd.read_parquet(archivo_fx).iloc[:, 0] if archivo_fx.exists() else None
    r, etiquetas, _ = hc.preparar_retornos(precios, sectores, ANIOS_RESUMEN, fx)
    a = hc.analizar(r, etiquetas)
    mensual, diarios = hc.rotacion_min_var(r, etiquetas)
    return {
        "nombre": nombre, "moneda": moneda, "n": int(a["n"]), "T": int(a["T"]),
        "sectores": len(a["sectores"]),
        "ev1_pca": float(a["lam_pca"][0] / a["n"]), "ev1_hpca": float(a["lam_hpca"][0] / a["n"]),
        "rotacion_pca": float(mensual["Rotación PCA"].mean()),
        "rotacion_hpca": float(mensual["Rotación HPCA"].mean()),
        "vol_pca": float(diarios["PCA"].std() * np.sqrt(252)),
        "vol_hpca": float(diarios["HPCA"].std() * np.sqrt(252)),
    }


if __name__ == "__main__":
    solo_resumen = "--solo-resumen" in sys.argv  # recalcula el resumen sin volver a descargar
    claves = [a for a in sys.argv[1:] if not a.startswith("--")] or list(INDICES)
    if solo_resumen:
        claves = [k for k in claves if (DATA / f"{k}_precios.parquet").exists()]
    archivo_resumen = DATA / "resumen.json"
    resumen = json.loads(archivo_resumen.read_text(encoding="utf-8")) if archivo_resumen.exists() else {}
    for clave in claves:
        try:
            resumen[clave] = resumir(clave) if solo_resumen else construir(clave)
            print(f"  rotación PCA {resumen[clave]['rotacion_pca']:.2f} | HPCA {resumen[clave]['rotacion_hpca']:.2f}")
        except Exception as e:  # un índice que falla no frena al resto
            print(f"  ERROR en {clave}: {e}")
    meta = {"actualizado": date.today().isoformat(), "anios_resumen": ANIOS_RESUMEN,
            "indices": {k: {"nombre": v[0], "moneda": v[1]} for k, v in INDICES.items()}}
    archivo_resumen.write_text(json.dumps(resumen, ensure_ascii=False, indent=1), encoding="utf-8")
    (DATA / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
