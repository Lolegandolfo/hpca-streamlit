# HPCA en los mercados del mundo

App de Streamlit que recrea el **Hierarchical PCA** de Avellaneda (2019) sobre los principales índices bursátiles, y lo compara con el PCA tradicional.

Sale del trabajo práctico final de Finanzas Cuantitativas, donde aplicamos el método al Merval, al Nikkei 225 y al FTSE. La app extiende ese análisis a otros mercados.

> Avellaneda, M. (2019). *Hierarchical PCA and applications to portfolio management*. arXiv. https://arxiv.org/abs/1910.02310

## Qué muestra

Se elige un índice en el menú lateral y una ventana de años. La app tiene cuatro pestañas:

| Pestaña | Contenido |
|---|---|
| **Autovectores** | Los primeros 5 autovectores de PCA y de HPCA, con las acciones ordenadas por sector. Incluye la similitud coseno entre ambos, la interpretación de cada portfolio HPCA y la varianza explicada acumulada. |
| **Correlaciones** | Heatmap de la matriz empírica, de la matriz HPCA y de su diferencia, más la correlación entre sectores (ρ̄). |
| **Rotación** | Rotación mensual de la cartera de mínima varianza estimada con PCA y con HPCA, y una comparación entre todos los mercados. |
| **Sectores** | Las acciones de cada sector y las que se descartaron por iliquidez. |

## Índices incluidos

S&P 500, Euro Stoxx 50, DAX 40, CAC 40, FTSE 100/250, Nikkei 225, Hang Seng, KOSPI 200, Nifty 50, S&P/ASX 50, S&P/TSX 60, B3 (Brasil) y Merval.

- **Constituyentes y sectores:**
  - Para la mayoría, salen de las tablas de Wikipedia, traducidos a los 11 sectores GICS.
  - El Hang Seng usa sus 4 sub-índices oficiales.
  - El KOSPI 200 usa la clasificación propia de KRX.
  - Merval, Nikkei y FTSE usan la partición sectorial del trabajo práctico.
- **Sectores chicos:** los que tienen menos de 3 acciones se agrupan en *Otros*.
- **Precios:** Yahoo Finance, precios de cierre ajustados desde 2015, en moneda local. El Merval se convierte a dólares con el CCL implícito (GGAL y YPF).
- **Sesgo de supervivencia:** se usan los componentes actuales de cada índice.

## Estructura

```
app.py                      # la app de Streamlit
hpca_core.py                # PCA, HPCA (Proposición 2) y cartera de mínima varianza
scripts/construir_datos.py  # descarga constituyentes, sectores y precios → data/
scripts/universos_tp.json   # particiones sectoriales del TP (Merval, Nikkei, FTSE)
data/                       # datos ya descargados (se commitean)
```

La app **no descarga nada al arrancar**: lee `data/`. Así carga rápido y no depende de los límites de Yahoo Finance en el servidor.

## Correr localmente

Con [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run streamlit run app.py
```

## Actualizar los datos

```bash
uv run --group datos python scripts/construir_datos.py              # todos los índices
uv run --group datos python scripts/construir_datos.py nikkei dax   # sólo algunos
```

Descarga todo de nuevo. Puede tardar varios minutos por la cantidad de tickers. Después hay que commitear `data/`.

## Deploy en Streamlit Community Cloud

1. Subir el repo a GitHub.
2. En [share.streamlit.io](https://share.streamlit.io), crear una app nueva apuntando a este repo, rama `main` y archivo `app.py`.
3. Las dependencias se instalan desde `requirements.txt`.
