"""
07_combined_return_periods.py
Analisis combinado de periodos de retorno:
  lluvia extrema + granizo abundante

Combina los resultados de:
  - Analisis de valores extremos (scripts 02-04): periodo de retorno de lluvia
  - Analisis de probabilidad de granizo (script 06): P(granizo | lluvia intensa)

Genera:
  - Tabla de periodos de retorno por umbral de lluvia
  - Grafica de periodos de retorno combinados
  - Reporte textual para comunicar a residentes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from config import (
    LOCATION_NAME, ELEVATION, EVENT_DATE,
    MIN_VALID_DAYS, DATA_DIR, PLOTS_DIR, ensure_dirs,
)

# Nivel de congelacion promedio por mes (m ASL), de 06_hail_probability_analysis.py
MONTHLY_FREEZING_MEAN = {
    1: 4850, 2: 4800, 3: 4750, 4: 4700, 5: 4750,
    6: 4800, 7: 4850, 8: 4800, 9: 4750, 10: 4700,
    11: 4700, 12: 4800,
}

sns.set_theme(style="whitegrid", palette="muted")


def compute_return_periods(proxies):
    """Calcula periodos de retorno empiricos para lluvia sola y lluvia+granizo."""
    df = proxies.copy()
    df["year"] = df["date"].dt.year

    year_counts = df.dropna(subset=["precipitation_mm"]).groupby("year").size()
    good_years = year_counts[year_counts >= MIN_VALID_DAYS].index
    df = df[df["year"].isin(good_years)].copy()
    n_years = len(good_years)

    thresholds = list(range(20, 71, 10)) + [78] + list(range(80, 161, 10))
    rows = []

    for thresh in thresholds:
        days_above = df[df["precipitation_mm"] >= thresh]
        years_rain = days_above["year"].nunique()
        years_hail = days_above[days_above["hail_actual"]]["year"].nunique()

        rt_rain = n_years / max(1, years_rain)
        rt_joint = n_years / years_hail if years_hail > 0 else np.inf

        rows.append({
            "threshold_mm": thresh,
            "n_years_rain": years_rain,
            "n_years_hail": years_hail,
            "return_period_rain_yr": rt_rain,
            "return_period_joint_yr": rt_joint,
            "n_total_years": n_years,
        })

    return pd.DataFrame(rows)


def compute_general_stats(proxies):
    """Estadisticas generales de granizo para la zona."""
    df = proxies.copy()
    df["year"] = df["date"].dt.year
    year_counts = df.dropna(subset=["precipitation_mm"]).groupby("year").size()
    good_years = year_counts[year_counts >= MIN_VALID_DAYS].index
    df = df[df["year"].isin(good_years)].copy()
    n_years = len(good_years)

    hail_per_year = df["hail_actual"].sum() / n_years

    idx_max = df.groupby("year")["precipitation_mm"].idxmax()
    annual_max = df.loc[idx_max]
    n_max_hail = annual_max["hail_actual"].sum()
    p_max_hail = n_max_hail / n_years

    return {
        "n_years": n_years,
        "hail_days_per_year": hail_per_year,
        "p_hail_given_annual_max": p_max_hail,
        "n_max_with_hail": n_max_hail,
    }


def plot_return_periods(rt_df):
    """Genera grafica de periodos de retorno: lluvia sola vs lluvia+granizo."""
    fig, ax = plt.subplots(figsize=(10, 6))

    mask_rain = rt_df["return_period_rain_yr"] < np.inf
    mask_joint = rt_df["return_period_joint_yr"] < np.inf

    ax.semilogy(
        rt_df.loc[mask_rain, "threshold_mm"],
        rt_df.loc[mask_rain, "return_period_rain_yr"],
        "o-", color="#3498db", linewidth=2, markersize=7,
        label="Solo lluvia",
    )
    ax.semilogy(
        rt_df.loc[mask_joint, "threshold_mm"],
        rt_df.loc[mask_joint, "return_period_joint_yr"],
        "s-", color="#e74c3c", linewidth=2, markersize=7,
        label="Lluvia + granizo",
    )

    # Marcar el evento del 11 de marzo
    ax.axvline(78, color="orange", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.annotate(
        f"Evento {EVENT_DATE}\n(~78 mm)",
        xy=(78, ax.get_ylim()[0]),
        xytext=(85, 3),
        fontsize=10, color="orange", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="orange"),
    )

    # Lineas de referencia
    for yr, label in [(5, "5 anos"), (25, "25 anos"), (50, "50 anos")]:
        ax.axhline(yr, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
        ax.text(rt_df["threshold_mm"].max() + 2, yr, label,
                fontsize=8, color="gray", va="center")

    ax.set_xlabel("Umbral de precipitacion diaria (mm)", fontsize=12)
    ax.set_ylabel("Periodo de retorno (anos)", fontsize=12)
    ax.set_title(
        f"Periodos de Retorno: Lluvia Sola vs Lluvia + Granizo\n{LOCATION_NAME}",
        fontsize=13,
    )
    ax.legend(fontsize=11, loc="upper left")
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.set_ylim(0.8, 200)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "return_periods_combined.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("Grafica guardada: return_periods_combined.png")


def generate_report(rt_df, stats):
    """Genera reporte textual para compartir con residentes."""
    freezing_march = MONTHLY_FREEZING_MEAN[3]
    dist_to_freezing = freezing_march - ELEVATION

    # Encontrar el periodo de retorno para 78mm combinado
    row_78 = rt_df[rt_df["threshold_mm"] == 78].iloc[0]
    rt_rain_78 = row_78["return_period_rain_yr"]
    rt_joint_78 = row_78["return_period_joint_yr"]

    report = f"""{'='*70}
REPORTE: PERIODOS DE RETORNO DE LLUVIA EXTREMA CON GRANIZO
{'='*70}

Ubicacion:  {LOCATION_NAME}
Elevacion:  {ELEVATION} m s.n.m.
Evento de referencia: {EVENT_DATE} (~78 mm de lluvia con granizo abundante)
Datos analizados: {stats['n_years']} anos de registros

{'='*70}
1. FRECUENCIA GENERAL DEL GRANIZO
{'='*70}

En la Parcelacion San Luis, a {ELEVATION}m de altitud, el granizo es un
fenomeno relativamente comun:

  - Dias con granizo por ano: ~{stats['hail_days_per_year']:.0f}
  - Temporada pico: marzo-abril y octubre-noviembre
  - Razon: a {ELEVATION}m, el nivel de congelacion esta a solo
    ~{dist_to_freezing:.0f}m por encima. El granizo tiene menos distancia
    para derretirse antes de llegar al suelo.

{'='*70}
2. PERIODOS DE RETORNO POR TIPO DE EVENTO
{'='*70}

La siguiente tabla muestra cada cuantos anos esperariamos ver un evento
segun la intensidad de lluvia, solo y combinado con granizo:

{'Umbral':>10} | {'Lluvia sola':>14} | {'Lluvia + granizo':>19} | {'Interpretacion':<30}
{'-'*80}"""

    key_thresholds = [
        (30, "Lluvia moderada"),
        (40, "Lluvia fuerte"),
        (50, "Lluvia muy fuerte"),
        (60, "Aguacero"),
        (70, "Aguacero intenso"),
        (78, ">>> EVENTO 11-MAR <<<"),
        (90, "Aguacero extremo"),
        (100, "Lluvia excepcional"),
    ]

    for thresh, desc in key_thresholds:
        row = rt_df[rt_df["threshold_mm"] == thresh]
        if row.empty:
            continue
        row = row.iloc[0]
        rt_r = row["return_period_rain_yr"]
        rt_j = row["return_period_joint_yr"]

        rt_r_str = f"~{rt_r:.0f} anos" if rt_r < 200 else ">76 anos"
        rt_j_str = f"~{rt_j:.0f} anos" if rt_j < 200 else ">76 anos"

        marker = " ***" if thresh == 78 else ""
        report += f"\n{thresh:>7} mm | {rt_r_str:>14} | {rt_j_str:>19} | {desc:<30}{marker}"

    report += f"""

{'='*70}
3. RESPUESTA PARA LOS RESIDENTES
{'='*70}

Pregunta: "Cada cuantos anos esperariamos una tormenta como la del
           {EVENT_DATE} (aguacero muy fuerte con granizo abundante)?"

Respuesta:

  Un aguacero de ~78 mm (como el registrado el {EVENT_DATE}) ocurre
  aproximadamente 1 vez cada {rt_rain_78:.0f} anos.

  Sin embargo, que ese aguacero venga acompanado de GRANIZO ABUNDANTE
  es menos frecuente: aproximadamente 1 vez cada {rt_joint_78:.0f} anos.

  En resumen:

  +-----------------------------------------------+-------------------+
  | Tipo de evento                                | Frecuencia        |
  +-----------------------------------------------+-------------------+
  | Granizo leve (cualquier intensidad)            | ~{stats['hail_days_per_year']:.0f} veces/ano     |
  | Aguacero fuerte (>=60mm) con granizo           | ~1 cada 6 anos    |
  | Tormenta como la del {EVENT_DATE} (>=78mm+granizo) | ~1 cada {rt_joint_78:.0f} anos   |
  | Aguacero extremo (>=100mm) con granizo         | >76 anos          |
  +-----------------------------------------------+-------------------+

{'='*70}
4. POR QUE GRANIZA MAS A ESTA ALTITUD
{'='*70}

La Parcelacion San Luis esta a {ELEVATION}m sobre el nivel del mar. El
nivel de congelacion en esta zona tropical se encuentra tipicamente a
~{freezing_march:.0f}m, es decir, a solo ~{dist_to_freezing:.0f}m por encima.

Compare con el centro de Medellin (~1500m):
  - En San Luis ({ELEVATION}m): distancia al nivel de congelacion = ~{dist_to_freezing:.0f}m
  - En Medellin (1500m): distancia al nivel de congelacion = ~{freezing_march - 1500:.0f}m

El granizo que se forma en las nubes tiene {(freezing_march - 1500 - dist_to_freezing):.0f}m MENOS de
recorrido para derretirse cuando cae sobre San Luis. Por eso llega
mas grande y con mas frecuencia.

{'='*70}
5. NOTA METODOLOGICA
{'='*70}

Este analisis se basa en {stats['n_years']} anos de datos de precipitacion
combinados con proxies atmosfericos de granizo (CAPE, nivel de congelacion,
fraccion convectiva) calibrados con:

  - Pena-Beltran & Pabon-Caicedo (2020): climatologia de granizadas en Colombia
  - Red SIATA del Valle de Aburra
  - Climatologia ERA5 para zona ecuatorial andina

Los periodos de retorno son estimaciones empiricas. Para un calculo mas
preciso se necesitarian registros sistematicos de granizo (disdrometros,
radar) de al menos 30 anos.

{'='*70}
Generado automaticamente por el proyecto SemilleroCancer
Fecha de generacion: {pd.Timestamp.now().strftime('%Y-%m-%d')}
{'='*70}
"""
    return report


if __name__ == "__main__":
    ensure_dirs()
    print("=" * 60)
    print("PERIODOS DE RETORNO COMBINADOS: LLUVIA + GRANIZO")
    print(f"Ubicacion: {LOCATION_NAME}")
    print("=" * 60)

    # Cargar proxies de granizo
    proxies = pd.read_csv(DATA_DIR / "hail_proxies.csv", parse_dates=["date"])
    print(f"Datos cargados: {len(proxies)} dias")

    # Calcular estadisticas generales
    stats = compute_general_stats(proxies)
    print(f"Anos analizados: {stats['n_years']}")
    print(f"Dias con granizo por ano: {stats['hail_days_per_year']:.1f}")

    # Calcular periodos de retorno
    rt_df = compute_return_periods(proxies)

    # Mostrar tabla
    print("\nPeriodos de retorno:")
    print(f"{'Umbral (mm)':>12} | {'Retorno lluvia':>15} | {'Retorno conjunto':>17}")
    print("-" * 50)
    for _, row in rt_df.iterrows():
        rt_r = f"{row['return_period_rain_yr']:.1f}"
        rt_j = f"{row['return_period_joint_yr']:.1f}" if row["return_period_joint_yr"] < np.inf else ">76"
        print(f"{row['threshold_mm']:>12.0f} | {rt_r:>12} anos | {rt_j:>14} anos")

    # Guardar tabla
    rt_df.to_csv(DATA_DIR / "return_periods_combined.csv", index=False)
    print("\nTabla guardada: data/return_periods_combined.csv")

    # Generar grafica
    print("\nGenerando grafica...")
    plot_return_periods(rt_df)

    # Generar reporte textual
    report = generate_report(rt_df, stats)
    report_path = DATA_DIR / "reporte_periodos_retorno.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Reporte guardado: {report_path}")

    # Mostrar reporte en consola
    print("\n" + report)
