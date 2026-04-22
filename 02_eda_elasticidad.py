"""
================================================================================
02_eda_elasticidad.py — ANÁLISIS DE ELASTICIDAD PRECIO vs COMPETENCIA
================================================================================
Lee el Parquet generado por 01_preparar_datos.py, construye el dataset
de comparación usando los pares definidos en config.json, calcula el
diferencial de precio en % y analiza cómo varía la demanda propia
según ese diferencial, agrupando en clusters de precio.

Adicionalmente:
  - Calcula share vs. total sal fina (usando la base de referencia del cliente)
  - Ejecuta análisis Celusal vs. polietileno más barato si está configurado

Uso:
    python 02_eda_elasticidad.py
    python 02_eda_elasticidad.py --config mi_config.json

Requisitos:
    pip install pandas numpy matplotlib seaborn scipy scikit-learn openpyxl pyarrow
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
import json
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENTOS Y CONFIG
# ─────────────────────────────────────────────────────────────────────────────

def leer_args():
    parser = argparse.ArgumentParser(description="EDA de elasticidad precio vs competencia.")
    parser.add_argument("--config", default="config.json")
    return parser.parse_args()


def cargar_config(path):
    if not os.path.exists(path):
        print(f"❌ No se encontró config: {path}")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# CARGA DEL PARQUET
# ─────────────────────────────────────────────────────────────────────────────

def cargar_datos(cfg):
    parquet = cfg["archivos"]["parquet_procesado"]
    if not os.path.exists(parquet):
        print(f"❌ No se encontró el Parquet procesado: {parquet}")
        print("   Ejecutar primero: python 01_preparar_datos.py")
        sys.exit(1)
    print(f"📂 Cargando datos procesados: {parquet}")
    df = pd.read_parquet(parquet)
    print(f"   → {len(df):,} filas | {df['origen'].value_counts().to_dict()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# BASE DE REFERENCIA (SAL FINA Y POLIETILENO)
# ─────────────────────────────────────────────────────────────────────────────

def cargar_base_referencia(cfg):
    """
    Carga las listas de SKUs de referencia del Excel del cliente.
    Retorna (Series con EANs de sal fina, Series con EANs de polietileno).
    Si el archivo no existe retorna (None, None).
    """
    ref = cfg.get("base_datos_referencia", {})
    archivo = ref.get("archivo", "")
    if not archivo or not os.path.exists(archivo):
        print(f"  ⚠️  No se encontró el archivo de referencia: '{archivo}'")
        return None, None

    sheet_fina = ref.get("sheet_sal_fina", "Base sales finas")
    sheet_poli = ref.get("sheet_polietileno", "Base Polietileno")

    skus_fina = (
        pd.read_excel(archivo, sheet_name=sheet_fina)["PROD_CODIGO_BARRAS"]
        .astype(str).str.strip()
    )
    skus_poli = (
        pd.read_excel(archivo, sheet_name=sheet_poli)["PROD_CODIGO_BARRAS"]
        .astype(str).str.strip()
    )
    print(f"  📊 Base sal fina cargada: {len(skus_fina)} SKUs")
    print(f"  📊 Base polietileno cargada: {len(skus_poli)} SKUs")
    return skus_fina, skus_poli


def calcular_vol_total_sal_fina(df, skus_sal_fina, cols):
    """
    Suma el volumen de TODOS los SKUs de sal fina por (período, PDV).
    Se usa como denominador para el share vs. total sal fina.
    Retorna DataFrame con columnas [período, PDV, vol_total_sal].
    """
    COL_PERIODO = cols["periodo"]
    COL_PDV     = cols["pdv_codigo"]
    COL_CODIGO  = cols["prod_codigo"]
    COL_VOL     = cols["volumen_vta"]

    mask = df[COL_CODIGO].astype(str).str.strip().isin(set(skus_sal_fina))
    df_sal = df[mask]

    if df_sal.empty:
        print("  ⚠️  No se encontraron SKUs de sal fina en los datos")
        return pd.DataFrame(columns=[COL_PERIODO, COL_PDV, "vol_total_sal"])

    vol_total = (
        df_sal.groupby([COL_PERIODO, COL_PDV])[COL_VOL]
        .sum()
        .rename("vol_total_sal")
        .reset_index()
    )
    print(f"  ✅ Vol. total sal fina: {len(vol_total):,} combinaciones período×PDV calculadas")
    return vol_total


# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DEL DATASET DE COMPARACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def resolver_columnas(df, cfg):
    """Resuelve el diccionario de columnas según modo (auto/semanal/mensual) o config antiguo."""
    if "columnas" in cfg and "columnas_semanal" not in cfg:
        return cfg["columnas"]
    cols_sem = cfg.get("columnas_semanal", {})
    cols_men = cfg.get("columnas_mensual", {})
    cols_com = cfg.get("columnas_comunes", {})
    modo_cfg = cfg.get("modo", "auto").lower()
    if modo_cfg == "semanal":
        modo = "semanal"
    elif modo_cfg == "mensual":
        modo = "mensual"
    else:
        col_sem = cols_sem.get("periodo", "SEM_FECHA_INICIO")
        modo = "semanal" if col_sem in df.columns else "mensual"
    cols_modo = cols_sem if modo == "semanal" else cols_men
    return {**cols_com, **cols_modo}


def construir_comparacion(df, cfg, vol_total_sal=None):
    """
    Por cada par comparable definido en config, hace un join entre
    registros propios y de competencia usando (período, PDV) como clave,
    y calcula:
      - dif_precio_pct: diferencial de precio %
                        (precio_propio - precio_comp) / precio_comp × 100
      - share_volumen:  vol_propio / (vol_propio + vol_comp) × 100
      - share_vs_total: vol_propio / vol_total_sal × 100  (si vol_total_sal es provisto)

    Precio propio y de competencia son promedios ponderados por volumen.
    """
    cols       = resolver_columnas(df, cfg)
    pares      = cfg.get("pares_comparables", [])
    bins       = cfg["clusters_precio"]["bins"]
    labels     = cfg["clusters_precio"]["labels"]

    COL_PERIODO  = cols["periodo"]
    COL_PDV      = cols["pdv_codigo"]
    COL_CODIGO   = cols["prod_codigo"]
    COL_NOMBRE   = cols["prod_nombre"]
    COL_CAT      = cols["prod_categoria"]
    COL_PRECIO   = cols["precio_unitario"]
    COL_VOL      = cols["volumen_vta"]
    COL_IMP      = cols["importe_vta"]

    if not pares:
        print("\n❌ No hay pares comparables en config.json. No se puede continuar.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("  CONSTRUYENDO DATASET DE COMPARACIÓN")
    print(f"{'='*60}")

    df_propio = df[df["origen"] == "propio"].copy()
    df_comp   = df[df["origen"] == "competencia"].copy()

    frames = []

    for par in pares:
        cod_p = str(par["propio"])
        cod_c = str(par["competencia"])
        desc  = par.get("descripcion", f"{cod_p} vs {cod_c}")

        sub_p = df_propio[df_propio[COL_CODIGO].astype(str).str.strip() == cod_p]
        sub_c = df_comp[df_comp[COL_CODIGO].astype(str).str.strip() == cod_c]

        if sub_p.empty or sub_c.empty:
            faltante = "propio" if sub_p.empty else "competencia"
            print(f"  ⚠️  [{desc}] Sin datos de {faltante} — omitido")
            continue

        # Precio promedio ponderado por volumen + suma de volumen por (período, PDV)
        agg_p = sub_p.groupby([COL_PERIODO, COL_PDV]).apply(
            lambda g: pd.Series({
                "precio_propio": np.average(g[COL_PRECIO], weights=g[COL_VOL]) if g[COL_VOL].sum() > 0 else g[COL_PRECIO].mean(),
                "vol_propio":    g[COL_VOL].sum(),
                "imp_propio":    g[COL_IMP].sum(),
            })
        ).reset_index()

        agg_c = sub_c.groupby([COL_PERIODO, COL_PDV]).apply(
            lambda g: pd.Series({
                "precio_comp": np.average(g[COL_PRECIO], weights=g[COL_VOL]) if g[COL_VOL].sum() > 0 else g[COL_PRECIO].mean(),
                "vol_comp":    g[COL_VOL].sum(),
            })
        ).reset_index()

        merged = pd.merge(agg_p, agg_c, on=[COL_PERIODO, COL_PDV], how="inner")

        if merged.empty:
            print(f"  ⚠️  [{desc}] Sin períodos/PDVs en común — omitido")
            continue

        merged["cod_propio"]  = cod_p
        merged["cod_comp"]    = cod_c
        merged["descripcion"] = desc

        cat_row = sub_p[[COL_CAT, COL_NOMBRE]].drop_duplicates()
        merged[COL_CAT]    = cat_row.iloc[0][COL_CAT]    if not cat_row.empty else ""
        merged[COL_NOMBRE] = cat_row.iloc[0][COL_NOMBRE] if not cat_row.empty else desc

        frames.append(merged)
        print(f"  ✅ [{desc}] {len(merged):,} combinaciones período × PDV")

    if not frames:
        print("\n❌ No se generó ningún par. Verificar códigos en config.json")
        sys.exit(1)

    resultado = pd.concat(frames, ignore_index=True)

    # Diferencial de precio %: (P_propio − P_comp) / P_comp × 100
    resultado["dif_precio_pct"] = (
        (resultado["precio_propio"] - resultado["precio_comp"])
        / resultado["precio_comp"] * 100
    ).round(4)

    # Share vs par comparable: vol_propio / (vol_propio + vol_comp) × 100
    vol_par = resultado["vol_propio"] + resultado["vol_comp"]
    resultado["share_volumen"] = np.where(
        vol_par > 0,
        resultado["vol_propio"] / vol_par * 100,
        np.nan
    ).round(4)

    # Share vs total sal fina: vol_propio / vol_total_sal × 100
    if vol_total_sal is not None and not vol_total_sal.empty:
        resultado = resultado.merge(vol_total_sal, on=[COL_PERIODO, COL_PDV], how="left")
        resultado["share_vs_total"] = np.where(
            resultado["vol_total_sal"] > 0,
            resultado["vol_propio"] / resultado["vol_total_sal"] * 100,
            np.nan
        ).round(4)
        n_match = resultado["vol_total_sal"].notna().sum()
        print(f"\n  ✅ share_vs_total calculado en {n_match:,}/{len(resultado):,} filas")

    # Cluster de precio
    resultado["cluster_precio"] = pd.cut(
        resultado["dif_precio_pct"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    # Filtro de outliers por diferencial de precio
    max_dif = cfg.get("filtros", {}).get("max_diferencial_pct", None)
    if max_dif is not None:
        antes = len(resultado)
        resultado = resultado[resultado["dif_precio_pct"].abs() <= max_dif]
        descartados = antes - len(resultado)
        if descartados > 0:
            print(f"\n  🔍 Filtro outliers: se descartaron {descartados:,} filas con diferencial > ±{max_dif}%")
            print(f"     Quedan {len(resultado):,} combinaciones para el análisis")

    print(f"\n  Total combinaciones: {len(resultado):,}")
    print(f"\n  Distribución por cluster de precio:")
    dist = resultado["cluster_precio"].value_counts().reindex(labels).dropna()
    for label, n in dist.items():
        barra = "█" * int(n / dist.max() * 30)
        print(f"    {label:<15} {barra} {n:,}")

    return resultado, cols


# ─────────────────────────────────────────────────────────────────────────────
# ANÁLISIS ESTADÍSTICO
# ─────────────────────────────────────────────────────────────────────────────

def analisis_estadistico(df, cols):
    COL_CAT = cols["prod_categoria"]
    print(f"\n{'='*60}")
    print("  ANÁLISIS DE ELASTICIDAD")
    print(f"{'='*60}")

    clean = df.dropna(subset=["dif_precio_pct", "share_volumen"])

    # Correlación global: r = Cov(dif, share) / (σ_dif × σ_share)
    r, p = stats.pearsonr(clean["dif_precio_pct"], clean["share_volumen"])
    fuerza = "FUERTE" if abs(r) > 0.5 else "MODERADA" if abs(r) > 0.3 else "DÉBIL"
    print(f"\n  Correlación global (Pearson): r={r:.4f}  p={p:.4e}  → {fuerza}")
    print(f"  Interpretación: precio más alto que competencia {'→ menor' if r < 0 else '→ mayor'} participación")

    # Resumen por cluster (promedios simples)
    print(f"\n  Share promedio por cluster:")
    resumen = clean.groupby("cluster_precio", observed=True).agg(
        share_medio=("share_volumen", "mean"),
        share_mediana=("share_volumen", "median"),
        n=("share_volumen", "count")
    )
    print(resumen.round(2).to_string())

    # Regresión por categoría: share = m × dif_precio + b  (OLS)
    if COL_CAT in clean.columns and clean[COL_CAT].nunique() > 1:
        print(f"\n  Regresión por categoría (coef m: cambio en share % por cada 1% de diferencial):")
        print(f"  {'Categoría':<35} {'R²':>6}  {'Coef m':>8}  {'N':>6}")
        print(f"  {'-'*58}")
        for cat, grupo in clean.groupby(COL_CAT):
            if len(grupo) < 10:
                continue
            X = grupo["dif_precio_pct"].values.reshape(-1, 1)
            y = grupo["share_volumen"].values
            reg = LinearRegression().fit(X, y)
            r2  = reg.score(X, y)
            print(f"  {str(cat):<35} {r2:>6.3f}  {reg.coef_[0]:>8.3f}  {len(grupo):>6,}")

    return clean


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZACIONES
# ─────────────────────────────────────────────────────────────────────────────

COLOR_PROPIO = "#1a6fa8"
COLOR_COMP   = "#d94f3d"
COLOR_NEUT   = "#6c757d"


def set_estilo():
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "figure.facecolor": "#f8f9fa",
        "axes.facecolor": "#ffffff",
    })


def guardar(fig, nombre, output_dir):
    path = os.path.join(output_dir, nombre)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  💾 {path}")
    plt.close(fig)


def grafico_boxplot_cluster(df, cfg, cols,
                             titulo=None,
                             filename="1_boxplot_share_cluster.png",
                             labels=None,
                             col_share="share_volumen"):
    """
    Boxplot de share por cluster de diferencial de precio.
    Agrega anotaciones por banda: N PDVs únicos, volumen total, rango min-max.
    """
    labels_uso = labels or cfg["clusters_precio"]["labels"]
    COL_PDV    = cols["pdv_codigo"]

    df_plot = df.dropna(subset=["cluster_precio", col_share])
    # Convertir cluster_precio a str para ordenar por labels_uso
    df_plot = df_plot.copy()
    df_plot["cluster_precio"] = df_plot["cluster_precio"].astype(str)
    # Filtrar solo los labels presentes en los datos para evitar boxes vacíos en la lista base
    labels_presentes = [l for l in labels_uso if l in df_plot["cluster_precio"].values]

    fig, ax = plt.subplots(figsize=(max(14, len(labels_presentes) * 1.4), 8))
    paleta = sns.color_palette("RdYlGn_r", len(labels_presentes))
    sns.boxplot(
        data=df_plot,
        x="cluster_precio", y=col_share,
        order=labels_presentes, palette=paleta, ax=ax,
        linewidth=1.2, flierprops=dict(marker=".", alpha=0.3, markersize=3)
    )
    ax.axhline(50, color=COLOR_NEUT, linestyle="--", linewidth=1, label="Paridad (50%)")

    # Estadísticas por banda: N PDVs únicos, Vol total, min-max share
    stats_banda = (
        df_plot.groupby("cluster_precio", observed=True)
        .agg(
            n_pdv=(COL_PDV, "nunique"),
            vol_total=("vol_propio", "sum"),
            min_s=(col_share, "min"),
            max_s=(col_share, "max"),
        )
    )
    y_lim_bottom = ax.get_ylim()[0]
    y_annot = y_lim_bottom + (ax.get_ylim()[1] - y_lim_bottom) * 0.02
    for i, label in enumerate(labels_presentes):
        if label not in stats_banda.index:
            continue
        row = stats_banda.loc[label]
        if pd.isna(row["n_pdv"]):
            continue
        txt = (
            f"N={int(row['n_pdv'])} PDVs\n"
            f"Vol: {row['vol_total']:,.0f}\n"
            f"{row['min_s']:.0f}%–{row['max_s']:.0f}%"
        )
        ax.text(i, y_annot, txt,
                ha="center", va="bottom", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          alpha=0.8, edgecolor="#cccccc"))

    titulo_final = titulo or "Share de volumen propio según diferencial de precio vs competencia"
    ax.set_title(titulo_final)
    label_share = "Participación vs par comparable (%)" if col_share == "share_volumen" else "Participación vs total sal fina (%)"
    ax.set_xlabel("Diferencial de precio (positivo = más caro que competencia)")
    ax.set_ylabel(label_share)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=9)
    fig.tight_layout()
    guardar(fig, filename, cfg["output_dir"])


def grafico_heatmap(df, cfg, cols):
    COL_CAT = cols["prod_categoria"]
    labels  = cfg["clusters_precio"]["labels"]

    if COL_CAT not in df.columns or df[COL_CAT].nunique() < 2:
        print("  ⚠️  Heatmap omitido (una sola categoría o sin datos)")
        return

    pivot = df.dropna(subset=["cluster_precio", "share_volumen"]).pivot_table(
        values="share_volumen",
        index=COL_CAT,
        columns="cluster_precio",
        aggfunc="mean",
        observed=True
    ).reindex(columns=labels)

    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(16, max(4, len(pivot) * 0.65 + 2)))
    sns.heatmap(
        pivot, annot=True, fmt=".1f", cmap="RdYlGn",
        center=50, vmin=20, vmax=80,
        linewidths=0.5, ax=ax,
        cbar_kws={"label": "Share promedio (%)"}
    )
    ax.set_title("Share promedio de volumen: Categoría × Cluster de diferencial de precio")
    ax.set_xlabel("Cluster de diferencial de precio")
    ax.set_ylabel("Categoría de producto")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    guardar(fig, "2_heatmap_categoria_cluster.png", cfg["output_dir"])


def grafico_scatter(df, cfg,
                    titulo=None,
                    filename="3_scatter_precio_share.png",
                    col_share="share_volumen"):
    """
    Scatter diferencial de precio vs share.
    Muestra:
      - Puntos coloreados por par comparable
      - Línea de regresión global (OLS): share = m × dif_precio + b
      - Línea de regresión por SKU propio (línea punteada) con pendiente y R²
    """
    clean = df.dropna(subset=["dif_precio_pct", col_share])
    if clean.empty:
        print(f"  ⚠️  Scatter omitido: sin datos suficientes")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Puntos por descripción de par
    pares_unicos = clean["descripcion"].unique()
    paleta_pares = sns.color_palette("tab10", len(pares_unicos))
    for i, par in enumerate(pares_unicos):
        sub = clean[clean["descripcion"] == par]
        ax.scatter(sub["dif_precio_pct"], sub[col_share],
                   alpha=0.3, s=15, color=paleta_pares[i], label=par[:40])

    x_min = clean["dif_precio_pct"].min()
    x_max = clean["dif_precio_pct"].max()
    x_line_global = np.linspace(x_min, x_max, 200)

    # Regresión global (OLS): share = m × dif_precio + b
    m, b, r, p_val, _ = stats.linregress(clean["dif_precio_pct"], clean[col_share])
    ax.plot(x_line_global, m * x_line_global + b,
            color="black", linewidth=2.5,
            label=f"Regresión global  m={m:.2f}  r={r:.2f}  p={p_val:.3f}")

    # Regresión por SKU propio (línea punteada) con pendiente y R²
    cod_propios_unicos = clean["cod_propio"].unique()
    if len(cod_propios_unicos) > 1:
        paleta_prop = sns.color_palette("Set2", len(cod_propios_unicos))
        for j, cod in enumerate(cod_propios_unicos):
            sub_cod = clean[clean["cod_propio"] == cod]
            if len(sub_cod) < 5:
                continue
            m_c, b_c, r_c, _, _ = stats.linregress(
                sub_cod["dif_precio_pct"], sub_cod[col_share]
            )
            r2_c = r_c ** 2
            x_line_c = np.linspace(
                sub_cod["dif_precio_pct"].min(),
                sub_cod["dif_precio_pct"].max(), 100
            )
            # Etiqueta corta usando la descripción del par asociado
            desc_sku = sub_cod["descripcion"].iloc[0][:30]
            ax.plot(x_line_c, m_c * x_line_c + b_c,
                    linewidth=1.8, linestyle="--", color=paleta_prop[j],
                    label=f"{desc_sku}  m={m_c:.2f}  R²={r2_c:.2f}")

    ax.axhline(50, color=COLOR_NEUT, linestyle=":", linewidth=0.8)
    ax.axvline(0,  color=COLOR_NEUT, linestyle=":", linewidth=0.8)

    titulo_final = titulo or "Diferencial de precio vs Participación de mercado"
    ax.set_title(titulo_final)
    ax.set_xlabel("Diferencial de precio % (positivo = más caro que competencia)")
    label_y = "Share vs par comparable (%)" if col_share == "share_volumen" else "Share vs total sal fina (%)"
    ax.set_ylabel(label_y)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    guardar(fig, filename, cfg["output_dir"])


def _media_ponderada(grupo, col_val, col_peso):
    """Promedio ponderado ignorando NaN en ambas columnas."""
    g = grupo.dropna(subset=[col_val, col_peso])
    if g.empty or g[col_peso].sum() == 0:
        return np.nan
    return float(np.average(g[col_val], weights=g[col_peso]))


def grafico_serie_temporal(df, cfg, cols,
                            titulo=None,
                            col_share="share_volumen"):
    """
    Serie temporal con dos paneles:
      - Panel superior: diferencial de precio y share (promedio simple y ponderado por vol)
      - Panel inferior: volumen propio acumulado por período
    """
    COL_PERIODO = cols["periodo"]

    if not pd.api.types.is_datetime64_any_dtype(df[COL_PERIODO]):
        print("  ⚠️  Serie temporal omitida (período no es datetime)")
        return

    # Promedios simples
    ts = df.groupby(COL_PERIODO).agg(
        dif_precio_simple=("dif_precio_pct", "mean"),
        share_simple=(col_share, "mean"),
        vol_propio=("vol_propio", "sum"),
    ).reset_index()

    # Promedios ponderados por vol_propio
    ts_pond = (
        df.groupby(COL_PERIODO)
        .apply(lambda g: pd.Series({
            "dif_precio_pond": _media_ponderada(g, "dif_precio_pct", "vol_propio"),
            "share_pond":      _media_ponderada(g, col_share, "vol_propio"),
        }))
        .reset_index()
    )
    ts = ts.merge(ts_pond, on=COL_PERIODO)

    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    ax2 = ax1.twinx()

    # Diferencial de precio (eje izquierdo): simple = sólido, ponderado = punteado
    ax1.plot(ts[COL_PERIODO], ts["dif_precio_simple"],
             color=COLOR_COMP, linewidth=2, marker="o", markersize=4,
             label="Dif. precio — promedio simple")
    ax1.plot(ts[COL_PERIODO], ts["dif_precio_pond"],
             color=COLOR_COMP, linewidth=1.5, linestyle="--", marker="o", markersize=3,
             alpha=0.7, label="Dif. precio — ponderado vol.")

    # Share (eje derecho): simple = sólido, ponderado = punteado
    ax2.plot(ts[COL_PERIODO], ts["share_simple"],
             color=COLOR_PROPIO, linewidth=2, marker="s", markersize=4,
             linestyle="-", label="Share — promedio simple")
    ax2.plot(ts[COL_PERIODO], ts["share_pond"],
             color=COLOR_PROPIO, linewidth=1.5, linestyle=":", marker="s", markersize=3,
             alpha=0.7, label="Share — ponderado vol.")

    ax1.axhline(0,  color=COLOR_NEUT, linestyle=":", linewidth=0.7)
    ax2.axhline(50, color="lightblue", linestyle=":", linewidth=0.7)
    ax1.set_ylabel("Diferencial de precio (%)", color=COLOR_COMP)
    ax2.set_ylabel("Share de volumen (%)", color=COLOR_PROPIO)

    titulo_panel = titulo or "Evolución temporal: diferencial de precio y participación de mercado"
    ax1.set_title(titulo_panel)

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=8, ncol=2)

    # Panel inferior: volumen propio
    ax3.bar(ts[COL_PERIODO], ts["vol_propio"], color=COLOR_PROPIO, alpha=0.7, width=20)
    ax3.set_ylabel("Volumen vendido (propio)")
    ax3.set_xlabel("Período")
    ax3.set_title("Volumen de ventas propio en el tiempo")
    ax3.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    guardar(fig, "4_serie_temporal.png", cfg["output_dir"])


def grafico_ranking_sensibilidad(df, cfg, top_n=15):
    resultados = []
    for cod, grupo in df.groupby("cod_propio"):
        g = grupo.dropna(subset=["dif_precio_pct", "share_volumen"])
        if len(g) < 8:
            continue
        r, _ = stats.pearsonr(g["dif_precio_pct"], g["share_volumen"])
        desc = g["descripcion"].iloc[0]
        resultados.append({"producto": desc[:45], "correlacion": r, "n_obs": len(g)})

    if not resultados:
        print("  ⚠️  Ranking omitido (menos de 8 observaciones por producto)")
        return

    rank_df = pd.DataFrame(resultados).sort_values("correlacion")

    fig, ax = plt.subplots(figsize=(11, max(5, len(rank_df) * 0.5 + 2)))
    colors = [COLOR_COMP if r < 0 else COLOR_PROPIO for r in rank_df["correlacion"]]
    bars = ax.barh(rank_df["producto"], rank_df["correlacion"], color=colors, edgecolor="white", height=0.6)
    ax.axvline(0, color="black", linewidth=0.8)

    for bar, val in zip(bars, rank_df["correlacion"]):
        offset = 0.01 if val >= 0 else -0.01
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}", va="center", ha=ha, fontsize=8)

    ax.set_xlabel("Correlación (diferencial precio % → share volumen)")
    ax.set_title("Sensibilidad al precio por producto comparable")
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_COMP,   label="Precio más alto → pierde share"),
        Patch(facecolor=COLOR_PROPIO, label="Precio más alto → gana share"),
    ]
    ax.legend(handles=legend_elements, fontsize=9)
    fig.tight_layout()
    guardar(fig, "5_ranking_sensibilidad.png", cfg["output_dir"])


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT EXCEL
# ─────────────────────────────────────────────────────────────────────────────

def exportar_excel(df, cfg, cols, filename="resumen_elasticidad.xlsx"):
    COL_CAT = cols["prod_categoria"]
    labels  = cfg["clusters_precio"]["labels"]
    path    = os.path.join(cfg["output_dir"], filename)

    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Datos_Comparacion", index=False)

            resumen = df.groupby("cluster_precio", observed=True).agg(
                share_medio=("share_volumen", "mean"),
                share_mediana=("share_volumen", "median"),
                dif_precio_media=("dif_precio_pct", "mean"),
                n_observaciones=("share_volumen", "count"),
                vol_propio_total=("vol_propio", "sum"),
            ).reindex(labels).reset_index()
            resumen.to_excel(writer, sheet_name="Resumen_Cluster", index=False)

            # Share vs total sal fina por cluster (si está disponible)
            if "share_vs_total" in df.columns:
                resumen_total = df.groupby("cluster_precio", observed=True).agg(
                    share_total_medio=("share_vs_total", "mean"),
                    share_total_mediana=("share_vs_total", "median"),
                    n_observaciones=("share_vs_total", "count"),
                ).reindex(labels).reset_index()
                resumen_total.to_excel(writer, sheet_name="Resumen_Share_Total", index=False)

            if COL_CAT in df.columns and df[COL_CAT].nunique() > 1:
                pivot = df.pivot_table(
                    values="share_volumen",
                    index=COL_CAT,
                    columns="cluster_precio",
                    aggfunc="mean",
                    observed=True
                ).reindex(columns=labels).round(2)
                pivot.to_excel(writer, sheet_name="Heatmap_Categoria")

            resumen_par = df.groupby("descripcion").agg(
                dif_precio_media=("dif_precio_pct", "mean"),
                dif_precio_std=("dif_precio_pct", "std"),
                share_medio=("share_volumen", "mean"),
                vol_propio_total=("vol_propio", "sum"),
                n=("share_volumen", "count"),
            ).round(3).reset_index()
            resumen_par.to_excel(writer, sheet_name="Resumen_Por_Par", index=False)

        print(f"  💾 {path}")
    except ImportError:
        print("  ⚠️  openpyxl no instalado: pip install openpyxl")


# ─────────────────────────────────────────────────────────────────────────────
# ANÁLISIS VS POLIETILENO
# ─────────────────────────────────────────────────────────────────────────────

def analisis_polietileno(df, skus_poli, vol_total_sal, cfg, cols):
    """
    Analiza Celusal vs. el SKU de polietileno más barato en cada PDV-período.

    Fórmulas:
      precio_poli_min = MIN(precio_ponderado_vol) por SKU de polietileno en PDV-período
      dif_precio_pct  = (precio_celusal − precio_poli_min) / precio_poli_min × 100
      share_volumen   = vol_celusal / vol_total_sal_fina × 100

    PDV-períodos sin polietileno → cluster "Sin polietileno".
    """
    cfg_poli = cfg.get("analisis_polietileno", {})
    if not cfg_poli.get("activo", False):
        return

    sku_celusal  = str(cfg_poli.get("sku_propio_celusal", ""))
    desc_anal    = cfg_poli.get("descripcion", "SAL CELUSAL vs POLIETILENO (más barato)")
    bins         = cfg_poli["bins"]
    labels_poli  = cfg_poli["labels"]
    label_sin    = cfg_poli.get("label_cluster_sin_comp", "Sin polietileno")
    output_sub   = os.path.join(cfg["output_dir"], cfg_poli.get("output_subdir", "polietileno"))
    os.makedirs(output_sub, exist_ok=True)

    COL_PERIODO = cols["periodo"]
    COL_PDV     = cols["pdv_codigo"]
    COL_CODIGO  = cols["prod_codigo"]
    COL_PRECIO  = cols["precio_unitario"]
    COL_VOL     = cols["volumen_vta"]

    print(f"\n{'='*60}")
    print(f"  ANÁLISIS POLIETILENO — {desc_anal}")
    print(f"{'='*60}")

    # 1. Precio Celusal ponderado por volumen por (período, PDV)
    df_celusal = df[df[COL_CODIGO].astype(str).str.strip() == sku_celusal]
    if df_celusal.empty:
        print(f"  ⚠️  No se encontraron datos para el SKU Celusal {sku_celusal} — análisis omitido")
        return

    agg_cel = df_celusal.groupby([COL_PERIODO, COL_PDV]).apply(
        lambda g: pd.Series({
            "precio_propio": np.average(g[COL_PRECIO], weights=g[COL_VOL]) if g[COL_VOL].sum() > 0 else g[COL_PRECIO].mean(),
            "vol_propio":    g[COL_VOL].sum(),
        })
    ).reset_index()

    # 2. Precio mínimo de polietileno por (período, PDV)
    # Paso a: precio ponderado por SKU × período × PDV
    # Paso b: mínimo de esos precios por período × PDV
    skus_poli_set = set(skus_poli.astype(str).str.strip())
    df_poli_filt = df[df[COL_CODIGO].astype(str).str.strip().isin(skus_poli_set)]

    if not df_poli_filt.empty:
        precio_por_sku = df_poli_filt.groupby([COL_PERIODO, COL_PDV, COL_CODIGO]).apply(
            lambda g: np.average(g[COL_PRECIO], weights=g[COL_VOL]) if g[COL_VOL].sum() > 0 else g[COL_PRECIO].mean()
        ).rename("precio_poli").reset_index()

        min_poli = (
            precio_por_sku
            .groupby([COL_PERIODO, COL_PDV])["precio_poli"]
            .min()
            .rename("precio_poli_min")
            .reset_index()
        )
    else:
        print("  ⚠️  No se encontraron SKUs de polietileno en los datos")
        min_poli = pd.DataFrame(columns=[COL_PERIODO, COL_PDV, "precio_poli_min"])

    # 3. Join Celusal + precio mínimo polietileno (left join)
    resultado = agg_cel.merge(min_poli, on=[COL_PERIODO, COL_PDV], how="left")

    # 4. Diferencial de precio (solo donde hay polietileno)
    mask_poli = resultado["precio_poli_min"].notna() & (resultado["precio_poli_min"] > 0)
    resultado["dif_precio_pct"] = np.where(
        mask_poli,
        (resultado["precio_propio"] - resultado["precio_poli_min"])
        / resultado["precio_poli_min"] * 100,
        np.nan
    ).round(4)

    # 5. Cluster de precio
    resultado["cluster_precio"] = label_sin  # valor por defecto (sin polietileno)
    mask_dif = resultado["dif_precio_pct"].notna()
    if mask_dif.sum() > 0:
        cut = pd.cut(
            resultado.loc[mask_dif, "dif_precio_pct"],
            bins=bins,
            labels=labels_poli,
            include_lowest=True
        )
        resultado.loc[mask_dif, "cluster_precio"] = cut.astype(str)

    # 6. Share vs total sal fina
    if vol_total_sal is not None and not vol_total_sal.empty:
        resultado = resultado.merge(vol_total_sal, on=[COL_PERIODO, COL_PDV], how="left")
        resultado["share_volumen"] = np.where(
            resultado["vol_total_sal"] > 0,
            resultado["vol_propio"] / resultado["vol_total_sal"] * 100,
            np.nan
        ).round(4)
    else:
        resultado["share_volumen"] = np.nan

    resultado["cod_propio"]  = sku_celusal
    resultado["descripcion"] = desc_anal

    # Filtro de outliers (solo para registros con polietileno)
    max_dif = cfg.get("filtros", {}).get("max_diferencial_pct", None)
    if max_dif is not None and mask_dif.sum() > 0:
        mask_ok  = resultado["dif_precio_pct"].abs() <= max_dif
        mask_nan = resultado["dif_precio_pct"].isna()
        resultado = resultado[mask_ok | mask_nan]

    n_con = resultado["dif_precio_pct"].notna().sum()
    n_sin = resultado["dif_precio_pct"].isna().sum()
    print(f"  ✅ Con polietileno: {n_con:,}  |  Sin polietileno: {n_sin:,}")

    dist = resultado["cluster_precio"].value_counts()
    for lbl, n in dist.items():
        barra = "█" * int(n / dist.max() * 25)
        print(f"    {lbl:<25} {barra} {n:,}")

    # 7. Gráficos (mismas funciones, distinta config de output y labels)
    cfg_graf = dict(cfg)
    cfg_graf["output_dir"] = output_sub
    labels_full = labels_poli + [label_sin]
    cfg_graf["clusters_precio"] = {"bins": bins, "labels": labels_full}

    set_estilo()
    grafico_boxplot_cluster(
        resultado, cfg_graf, cols,
        titulo=f"Share vs total sal fina — {desc_anal}",
        filename="1_boxplot_share_cluster.png",
        labels=labels_full,
        col_share="share_volumen",
    )
    grafico_scatter(
        resultado, cfg_graf,
        titulo=f"Diferencial de precio vs share — {desc_anal}",
        filename="3_scatter_precio_share.png",
        col_share="share_volumen",
    )
    grafico_serie_temporal(
        resultado, cfg_graf, cols,
        titulo=f"Evolución temporal — {desc_anal}",
        col_share="share_volumen",
    )

    # 8. Export Excel
    labels_full_cfg = {"bins": bins, "labels": labels_full}
    cfg_excel = dict(cfg_graf)
    cfg_excel["clusters_precio"] = labels_full_cfg
    exportar_excel(resultado, cfg_excel, cols, filename="resumen_polietileno.xlsx")

    print(f"\n  ✅ Análisis polietileno → ./{output_sub}/")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = leer_args()
    cfg  = cargar_config(args.config)
    os.makedirs(cfg["output_dir"], exist_ok=True)

    print("\n" + "="*60)
    print("  PASO 2 — EDA DE ELASTICIDAD PRECIO VS COMPETENCIA")
    print("="*60)

    df = cargar_datos(cfg)

    # Cargar base de referencia (sal fina y polietileno)
    print(f"\n{'='*60}")
    print("  CARGANDO BASE DE REFERENCIA")
    print(f"{'='*60}")
    skus_sal_fina, skus_poli = cargar_base_referencia(cfg)

    # Resolver columnas antes de pasarlas a las funciones auxiliares
    cols = resolver_columnas(df, cfg)

    # Calcular volumen total sal fina por (período, PDV)
    vol_total_sal = None
    if skus_sal_fina is not None:
        vol_total_sal = calcular_vol_total_sal_fina(df, skus_sal_fina, cols)

    # Construir dataset de comparación (pares propio vs competencia)
    df_comp, cols = construir_comparacion(df, cfg, vol_total_sal=vol_total_sal)
    analisis_estadistico(df_comp, cols)

    print(f"\n{'='*60}")
    print("  GENERANDO GRÁFICOS — ANÁLISIS BASE (vs par comparable)")
    print(f"{'='*60}")
    set_estilo()
    grafico_boxplot_cluster(df_comp, cfg, cols)
    grafico_heatmap(df_comp, cfg, cols)
    grafico_scatter(df_comp, cfg)
    grafico_serie_temporal(df_comp, cfg, cols)
    grafico_ranking_sensibilidad(df_comp, cfg)

    # Gráfico adicional con share vs total sal fina
    if "share_vs_total" in df_comp.columns:
        print(f"\n{'='*60}")
        print("  GENERANDO GRÁFICOS — SHARE VS TOTAL SAL FINA")
        print(f"{'='*60}")
        grafico_boxplot_cluster(
            df_comp, cfg, cols,
            titulo="Share vs TOTAL sal fina según diferencial de precio",
            filename="1b_boxplot_share_total_sal.png",
            col_share="share_vs_total",
        )
        grafico_scatter(
            df_comp, cfg,
            titulo="Diferencial de precio vs Share vs total sal fina",
            filename="3b_scatter_share_total_sal.png",
            col_share="share_vs_total",
        )

    print(f"\n{'='*60}")
    print("  EXPORTANDO EXCEL")
    print(f"{'='*60}")
    exportar_excel(df_comp, cfg, cols)

    # Análisis vs polietileno
    if skus_poli is not None and cfg.get("analisis_polietileno", {}).get("activo", False):
        analisis_polietileno(df, skus_poli, vol_total_sal, cfg, cols)

    print(f"\n✅ Análisis completo. Archivos en: ./{cfg['output_dir']}/\n")


if __name__ == "__main__":
    main()
