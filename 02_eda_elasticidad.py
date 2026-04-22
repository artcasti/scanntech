"""
================================================================================
02_eda_elasticidad.py — ANÁLISIS DE ELASTICIDAD PRECIO vs COMPETENCIA
================================================================================
Lee el Parquet generado por 01_preparar_datos.py, construye el dataset
de comparación usando los pares definidos en config.json, calcula el
diferencial de precio en % y analiza cómo varía la demanda propia
según ese diferencial, agrupando en clusters de precio.

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


def construir_comparacion(df, cfg):
    """
    Por cada par comparable definido en config, hace un join entre
    registros propios y de competencia usando (período, PDV) como clave,
    y calcula diferencial de precio % y share de volumen.
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

        # Agregar por (período, PDV): precio promedio ponderado por volumen, suma de volumen
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

        # Metadata del producto
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

    # Diferencial de precio %
    resultado["dif_precio_pct"] = (
        (resultado["precio_propio"] - resultado["precio_comp"])
        / resultado["precio_comp"] * 100
    ).round(4)

    # Share de volumen (%)
    vol_total = resultado["vol_propio"] + resultado["vol_comp"]
    resultado["share_volumen"] = np.where(
        vol_total > 0,
        resultado["vol_propio"] / vol_total * 100,
        np.nan
    ).round(4)

    # Cluster de precio
    resultado["cluster_precio"] = pd.cut(
        resultado["dif_precio_pct"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )
    # ── Filtro de outliers por diferencial de precio ──────────────────────────
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

    # Correlación global
    r, p = stats.pearsonr(clean["dif_precio_pct"], clean["share_volumen"])
    fuerza = "FUERTE" if abs(r) > 0.5 else "MODERADA" if abs(r) > 0.3 else "DÉBIL"
    print(f"\n  Correlación global (Pearson): r={r:.4f}  p={p:.4e}  → {fuerza}")
    print(f"  Interpretación: precio más alto que competencia {'→ menor' if r < 0 else '→ mayor'} participación")

    # Resumen por cluster
    print(f"\n  Share promedio por cluster:")
    resumen = clean.groupby("cluster_precio", observed=True).agg(
        share_medio=("share_volumen", "mean"),
        share_mediana=("share_volumen", "median"),
        n=("share_volumen", "count")
    )
    print(resumen.round(2).to_string())

    # Regresión por categoría
    if COL_CAT in clean.columns and clean[COL_CAT].nunique() > 1:
        print(f"\n  Regresión por categoría (coef: cambio en share % por cada 1% de diferencial):")
        print(f"  {'Categoría':<35} {'R²':>6}  {'Coef':>8}  {'N':>6}")
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


def grafico_boxplot_cluster(df, cfg):
    labels = cfg["clusters_precio"]["labels"]
    fig, ax = plt.subplots(figsize=(14, 6))
    paleta = sns.color_palette("RdYlGn_r", len(labels))
    sns.boxplot(
        data=df.dropna(subset=["cluster_precio", "share_volumen"]),
        x="cluster_precio", y="share_volumen",
        order=labels, palette=paleta, ax=ax,
        linewidth=1.2, flierprops=dict(marker=".", alpha=0.3, markersize=3)
    )
    ax.axhline(50, color=COLOR_NEUT, linestyle="--", linewidth=1, label="Paridad (50%)")
    ax.set_title("Share de volumen propio según diferencial de precio vs competencia")
    ax.set_xlabel("Diferencial de precio (positivo = más caro que competencia)")
    ax.set_ylabel("Participación de mercado — volumen (%)")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=9)
    fig.tight_layout()
    guardar(fig, "1_boxplot_share_cluster.png", cfg["output_dir"])


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


def grafico_scatter(df, cfg):
    clean = df.dropna(subset=["dif_precio_pct", "share_volumen"])
    fig, ax = plt.subplots(figsize=(12, 6))

    # Puntos con color por descripción de par
    pares_unicos = clean["descripcion"].unique()
    paleta_pares = sns.color_palette("tab10", len(pares_unicos))
    for i, par in enumerate(pares_unicos):
        sub = clean[clean["descripcion"] == par]
        ax.scatter(sub["dif_precio_pct"], sub["share_volumen"],
                   alpha=0.3, s=15, color=paleta_pares[i], label=par[:35])

    # Línea de regresión global
    m, b, r, p_val, _ = stats.linregress(clean["dif_precio_pct"], clean["share_volumen"])
    x_line = np.linspace(clean["dif_precio_pct"].min(), clean["dif_precio_pct"].max(), 200)
    ax.plot(x_line, m * x_line + b, color="black", linewidth=2,
            label=f"Regresión global (r={r:.2f}, p={p_val:.3f})")

    ax.axhline(50, color=COLOR_NEUT, linestyle=":", linewidth=0.8)
    ax.axvline(0,  color=COLOR_NEUT, linestyle=":", linewidth=0.8)
    ax.set_title("Diferencial de precio vs Participación de mercado")
    ax.set_xlabel("Diferencial de precio % (positivo = más caro que competencia)")
    ax.set_ylabel("Share de volumen propio (%)")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    guardar(fig, "3_scatter_precio_share.png", cfg["output_dir"])


def grafico_serie_temporal(df, cfg, cols):
    COL_PERIODO = cols["periodo"]

    if not pd.api.types.is_datetime64_any_dtype(df[COL_PERIODO]):
        print("  ⚠️  Serie temporal omitida (período no es datetime)")
        return

    ts = df.groupby(COL_PERIODO).agg(
        dif_precio_media=("dif_precio_pct", "mean"),
        share_media=("share_volumen", "mean"),
        vol_propio=("vol_propio", "sum"),
    ).reset_index()

    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    ax2 = ax1.twinx()

    # Panel superior: diferencial + share
    ax1.plot(ts[COL_PERIODO], ts["dif_precio_media"], color=COLOR_COMP,
             linewidth=2, marker="o", markersize=4, label="Diferencial precio %")
    ax2.plot(ts[COL_PERIODO], ts["share_media"], color=COLOR_PROPIO,
             linewidth=2, marker="s", markersize=4, linestyle="--", label="Share volumen %")
    ax1.axhline(0,  color=COLOR_NEUT, linestyle=":", linewidth=0.7)
    ax2.axhline(50, color="lightblue", linestyle=":", linewidth=0.7)
    ax1.set_ylabel("Diferencial de precio (%)", color=COLOR_COMP)
    ax2.set_ylabel("Share de volumen (%)", color=COLOR_PROPIO)
    ax1.set_title("Evolución temporal: diferencial de precio y participación de mercado")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=9)

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

    # Etiquetas en las barras
    for bar, val in zip(bars, rank_df["correlacion"]):
        offset = 0.01 if val >= 0 else -0.01
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}", va="center", ha=ha, fontsize=8)

    ax.set_xlabel("Correlación (diferencial precio % → share volumen)")
    ax.set_title(f"Sensibilidad al precio por producto comparable")
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

def exportar_excel(df, cfg, cols):
    COL_CAT = cols["prod_categoria"]
    labels  = cfg["clusters_precio"]["labels"]
    path    = os.path.join(cfg["output_dir"], "resumen_elasticidad.xlsx")

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
    df_comp, cols = construir_comparacion(df, cfg)
    analisis_estadistico(df_comp, cols)

    print(f"\n{'='*60}")
    print("  GENERANDO GRÁFICOS")
    print(f"{'='*60}")
    set_estilo()
    grafico_boxplot_cluster(df_comp, cfg)
    grafico_heatmap(df_comp, cfg, cols)
    grafico_scatter(df_comp, cfg)
    grafico_serie_temporal(df_comp, cfg, cols)
    grafico_ranking_sensibilidad(df_comp, cfg)

    print(f"\n{'='*60}")
    print("  EXPORTANDO EXCEL")
    print(f"{'='*60}")
    exportar_excel(df_comp, cfg, cols)

    print(f"\n✅ Análisis completo. Archivos en: ./{cfg['output_dir']}/\n")


if __name__ == "__main__":
    main()
