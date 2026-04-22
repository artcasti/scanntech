"""
================================================================================
01_preparar_datos.py — PREPARACIÓN Y CLASIFICACIÓN DEL DATASET
================================================================================
Lee el CSV unificado (propios + competencia en el mismo archivo),
detecta automáticamente si es semanal o mensual según las columnas presentes,
clasifica el origen de cada registro según la lista de códigos propios
definida en config.json, parsea la columna de período a datetime y
exporta el resultado como Parquet para uso eficiente en el paso 2.

Uso:
    python 01_preparar_datos.py
    python 01_preparar_datos.py --config mi_config.json

Requisitos:
    pip install pandas pyarrow
================================================================================
"""

import pandas as pd
import json
import argparse
import os
import sys
import time


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENTOS
# ─────────────────────────────────────────────────────────────────────────────

def leer_args():
    parser = argparse.ArgumentParser(description="Prepara el CSV de ventas para el EDA.")
    parser.add_argument("--config", default="config.json", help="Ruta al archivo config.json")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

def cargar_config(path):
    if not os.path.exists(path):
        print(f"❌ No se encontró el archivo de config: {path}")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# DETECCIÓN AUTOMÁTICA DE MODO (semanal / mensual)
# ─────────────────────────────────────────────────────────────────────────────

def detectar_modo(df, cfg):
    """
    Detecta si el CSV es semanal o mensual mirando las columnas presentes,
    y construye el diccionario 'columnas' completo para el resto del script.

    Lógica:
      - Si existe 'SEM_FECHA_INICIO'  → modo semanal
      - Si existe 'MES'               → modo mensual
      - Si modo='semanal'/'mensual' en config → fuerza ese modo (sin autodetección)
    """
    modo_config = cfg.get("modo", "auto").lower()
    cols_sem    = cfg.get("columnas_semanal", {})
    cols_men    = cfg.get("columnas_mensual", {})
    cols_com    = cfg.get("columnas_comunes", {})

    # Soporte para config antiguo (estructura plana "columnas")
    if "columnas" in cfg and "columnas_semanal" not in cfg:
        print("⚙️  Config formato anterior detectado — usando sección 'columnas' directamente")
        return cfg["columnas"]

    col_sem = cols_sem.get("periodo", "SEM_FECHA_INICIO")
    col_men = cols_men.get("periodo", "MES")

    if modo_config == "semanal":
        modo = "semanal"
    elif modo_config == "mensual":
        modo = "mensual"
    else:
        # Autodetección
        if col_sem in df.columns:
            modo = "semanal"
        elif col_men in df.columns:
            modo = "mensual"
        else:
            print("❌ No se pudo detectar el modo. No se encontró ni 'SEM_FECHA_INICIO' ni 'MES'.")
            print(f"   Columnas disponibles: {list(df.columns)}")
            sys.exit(1)

    # Construir diccionario de columnas unificado
    cols_modo = cols_sem if modo == "semanal" else cols_men
    columnas  = {**cols_com, **cols_modo}

    print(f"⚙️  Modo detectado: {modo.upper()}")
    print(f"   Columna de período : '{columnas['periodo']}'")
    print(f"   Columna de código  : '{columnas['prod_codigo']}'")

    return columnas


# ─────────────────────────────────────────────────────────────────────────────
# CARGA DEL CSV
# ─────────────────────────────────────────────────────────────────────────────

def cargar_csv(cfg):
    archivo  = cfg["archivos"]["csv_entrada"]
    sep      = cfg["archivos"]["separador"]
    encoding = cfg["archivos"]["encoding"]

    if not os.path.exists(archivo):
        print(f"❌ No se encontró el archivo CSV: {archivo}")
        sys.exit(1)

    print(f"📂 Leyendo: {archivo}")
    t0 = time.time()
    df = pd.read_csv(archivo, sep=sep, encoding=encoding, low_memory=False)
    df.columns = df.columns.str.strip()
    elapsed = time.time() - t0

    print(f"   → {len(df):,} filas | {df.shape[1]} columnas | {elapsed:.1f}s")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PARSEO DE FECHA
# ─────────────────────────────────────────────────────────────────────────────

def parsear_periodo(df, columnas):
    col     = columnas["periodo"]
    formato = columnas.get("formato_fecha", "%Y%m")

    if col not in df.columns:
        print(f"❌ Columna de período '{col}' no encontrada.")
        print(f"   Columnas disponibles: {list(df.columns)}")
        sys.exit(1)

    print(f"📅 Parseando columna '{col}' con formato '{formato}'...")

    parsed = pd.to_datetime(df[col].astype(str).str.strip(), format=formato, errors="coerce")

    pct_nat = parsed.isna().mean()
    if pct_nat > 0.05:
        print(f"   ⚠️  {pct_nat*100:.1f}% no parseados con formato '{formato}'. Intentando inferencia automática...")
        parsed2 = pd.to_datetime(df[col].astype(str).str.strip(), infer_datetime_format=True, errors="coerce")
        if parsed2.isna().mean() < pct_nat:
            parsed = parsed2

    df[col] = parsed

    n_nat = df[col].isna().sum()
    if n_nat > 0:
        print(f"   ⚠️  {n_nat:,} registros con fecha no parseada (NaT)")
        print(f"      Ejemplos: {df.loc[df[col].isna(), col].head(5).tolist()}")
    else:
        print(f"   ✅ Todos los períodos parseados correctamente")

    rango = f"{df[col].min().date()} → {df[col].max().date()}" if not df[col].isna().all() else "N/D"
    print(f"   Rango temporal: {rango} | Períodos únicos: {df[col].nunique()}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLASIFICACIÓN DE ORIGEN
# ─────────────────────────────────────────────────────────────────────────────

def clasificar_origen(df, cfg, columnas):
    col_codigo   = columnas["prod_codigo"]
    codigos_prop = set(cfg["codigos_propios"])

    if col_codigo not in df.columns:
        print(f"❌ Columna de código '{col_codigo}' no encontrada.")
        sys.exit(1)

    print(f"\n🏷️  Clasificando origen por columna '{col_codigo}'...")
    print(f"   Códigos propios definidos en config: {len(codigos_prop)}")

    df["origen"] = df[col_codigo].astype(str).str.strip().isin(codigos_prop).map(
        {True: "propio", False: "competencia"}
    )

    conteo = df["origen"].value_counts()
    total  = len(df)
    for origen, n in conteo.items():
        print(f"   {origen:<15}: {n:>10,} registros ({n/total*100:.1f}%)")

    n_propios = conteo.get("propio", 0)
    if n_propios == 0:
        print("\n   ⚠️  ATENCIÓN: 0 registros clasificados como 'propio'.")
        print("   Verificar que los códigos en config.json coincidan con el CSV.")
        print(f"   Primeros 10 códigos en el CSV:")
        print("  ", list(df[col_codigo].astype(str).unique()[:10]))

    return df


# ─────────────────────────────────────────────────────────────────────────────
# VALIDACIÓN DE PARES COMPARABLES
# ─────────────────────────────────────────────────────────────────────────────

def validar_pares_comparables(df, cfg, columnas):
    col_codigo = columnas["prod_codigo"]
    pares      = cfg.get("pares_comparables", [])

    if not pares:
        print("\n⚠️  No hay pares comparables definidos en config.json")
        return

    print(f"\n🔗 Validando {len(pares)} par(es) comparable(s)...")
    codigos_csv = set(df[col_codigo].astype(str).str.strip().unique())

    todos_ok = True
    for par in pares:
        cod_p = str(par["propio"])
        cod_c = str(par["competencia"])
        desc  = par.get("descripcion", "")
        ok_p  = cod_p in codigos_csv
        ok_c  = cod_c in codigos_csv
        estado = "✅" if (ok_p and ok_c) else "❌"
        print(f"   {estado}  {desc}")
        if not ok_p:
            print(f"        Propio '{cod_p}' NO encontrado en CSV")
        if not ok_c:
            print(f"        Competencia '{cod_c}' NO encontrado en CSV")
        if not (ok_p and ok_c):
            todos_ok = False

    if todos_ok:
        print("   Todos los pares validados correctamente ✅")
    else:
        print("\n   ⚠️  Algunos pares tienen problemas. Revisar config.json antes de continuar.")


# ─────────────────────────────────────────────────────────────────────────────
# REPORTE DE CALIDAD
# ─────────────────────────────────────────────────────────────────────────────

def reporte_calidad(df, columnas):
    print(f"\n{'='*60}")
    print("  REPORTE DE CALIDAD DEL DATASET")
    print(f"{'='*60}")

    nulos = df.isnull().sum()
    nulos = nulos[nulos > 0].sort_values(ascending=False)
    if len(nulos):
        print("\n  Columnas con valores nulos:")
        for col, n in nulos.items():
            print(f"    {col:<30}: {n:>8,} ({n/len(df)*100:.1f}%)")
    else:
        print("  ✅ Sin valores nulos")

    col_precio = columnas.get("precio_unitario")
    if col_precio and col_precio in df.columns:
        p = df[col_precio].dropna()
        q1, q3 = p.quantile(0.25), p.quantile(0.75)
        iqr = q3 - q1
        n_out = ((p < q1 - 1.5*iqr) | (p > q3 + 1.5*iqr)).sum()
        print(f"\n  Precio unitario:")
        print(f"    min={p.min():.2f} | Q1={q1:.2f} | mediana={p.median():.2f} | Q3={q3:.2f} | max={p.max():.2f}")
        print(f"    Outliers (IQR): {n_out:,} ({n_out/len(df)*100:.1f}%)")

    n_dup = df.duplicated().sum()
    print(f"\n  Filas duplicadas exactas: {n_dup:,}")


# ─────────────────────────────────────────────────────────────────────────────
# EXPORTAR PARQUET
# ─────────────────────────────────────────────────────────────────────────────

def exportar_parquet(df, cfg):
    salida = cfg["archivos"]["parquet_procesado"]
    os.makedirs(os.path.dirname(salida) if os.path.dirname(salida) else ".", exist_ok=True)

    print(f"\n💾 Exportando a Parquet: {salida}")
    t0 = time.time()
    df.to_parquet(salida, index=False, engine="pyarrow")
    elapsed = time.time() - t0
    size_mb = os.path.getsize(salida) / 1024 / 1024
    print(f"   ✅ Listo en {elapsed:.1f}s | Tamaño: {size_mb:.1f} MB")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = leer_args()
    cfg  = cargar_config(args.config)

    os.makedirs(cfg["output_dir"], exist_ok=True)

    print("\n" + "="*60)
    print("  PASO 1 — PREPARACIÓN DE DATOS")
    print("="*60)

    df       = cargar_csv(cfg)
    columnas = detectar_modo(df, cfg)       # ← detecta semanal/mensual automáticamente
    df       = parsear_periodo(df, columnas)
    df       = clasificar_origen(df, cfg, columnas)

    reporte_calidad(df, columnas)
    validar_pares_comparables(df, cfg, columnas)

    exportar_parquet(df, cfg)

    print("\n" + "="*60)
    print("  ✅ Preparación completa.")
    print(f"  Próximo paso: python 02_eda_elasticidad.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
