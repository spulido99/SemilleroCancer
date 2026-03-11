"""
02_exploratory_analysis.py
Analisis exploratorio de la precipitacion historica:
climatologia mensual, totales anuales, serie temporal reciente.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from config import (
    LOCATION_NAME, EVENT_DATE, MIN_VALID_DAYS,
    DATA_DIR, PLOTS_DIR, ensure_dirs,
)

sns.set_theme(style="whitegrid", palette="muted")


def load_data():
    df = pd.read_csv(DATA_DIR / "historical_daily_precip.csv", parse_dates=["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    return df


def print_summary(df):
    valid = df.dropna(subset=["precipitation_mm"])
    print("\n--- ESTADISTICAS RESUMEN ---")
    print(f"Periodo: {df['date'].min().date()} a {df['date'].max().date()}")
    print(f"Total de dias: {len(df)}")
    print(f"Dias con datos: {len(valid)}")
    print(f"Precipitacion diaria media: {valid['precipitation_mm'].mean():.1f} mm")
    print(f"Precipitacion diaria mediana: {valid['precipitation_mm'].median():.1f} mm")
    print(f"Desviacion estandar: {valid['precipitation_mm'].std():.1f} mm")
    print(f"Maximo historico: {valid['precipitation_mm'].max():.1f} mm "
          f"({valid.loc[valid['precipitation_mm'].idxmax(), 'date'].date()})")
    print(f"Dias secos (<1 mm): {(valid['precipitation_mm'] < 1).sum()} "
          f"({(valid['precipitation_mm'] < 1).mean() * 100:.1f}%)")
    print(f"Dias con lluvia fuerte (>30 mm): {(valid['precipitation_mm'] > 30).sum()}")
    print(f"Dias con lluvia muy fuerte (>50 mm): {(valid['precipitation_mm'] > 50).sum()}")


def report_data_quality(df):
    """Identifica anios con datos insuficientes."""
    yearly_counts = df.dropna(subset=["precipitation_mm"]).groupby("year").size()
    bad_years = yearly_counts[yearly_counts < MIN_VALID_DAYS].index.tolist()
    if bad_years:
        print(f"\nAnios con <{MIN_VALID_DAYS} dias validos (excluidos de maximos anuales):")
        for y in bad_years:
            print(f"  {y}: {yearly_counts[y]} dias")
    else:
        print(f"\nTodos los anios tienen >={MIN_VALID_DAYS} dias validos.")
    return bad_years


def plot_monthly_climatology(df):
    """Precipitacion media mensual con IQR."""
    monthly = df.dropna(subset=["precipitation_mm"]).copy()
    # Totales mensuales por anio-mes
    monthly_totals = monthly.groupby(["year", "month"])["precipitation_mm"].sum().reset_index()
    # Estadisticas por mes
    stats = monthly_totals.groupby("month")["precipitation_mm"].agg(
        ["mean", "median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
    )
    stats.columns = ["mean", "median", "q25", "q75"]

    fig, ax = plt.subplots(figsize=(10, 5))
    months = np.arange(1, 13)
    month_names = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                   "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

    colors = ["#e74c3c" if m == 3 else "#3498db" for m in months]
    bars = ax.bar(months, stats["mean"], color=colors, edgecolor="white", width=0.7)
    ax.errorbar(months, stats["mean"],
                yerr=[stats["mean"] - stats["q25"], stats["q75"] - stats["mean"]],
                fmt="none", ecolor="gray", capsize=4, capthick=1.5)

    ax.set_xticks(months)
    ax.set_xticklabels(month_names)
    ax.set_ylabel("Precipitacion mensual (mm)")
    ax.set_title(f"Climatologia Mensual de Precipitacion\n{LOCATION_NAME}")
    ax.legend(["Rango intercuartilico"], loc="upper right")

    # Anotar marzo
    march_val = stats.loc[3, "mean"]
    ax.annotate(f"Marzo\n{march_val:.0f} mm", xy=(3, march_val),
                xytext=(3, march_val + 40), ha="center", fontsize=9, color="#e74c3c",
                arrowprops=dict(arrowstyle="->", color="#e74c3c"))

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "monthly_climatology.png", dpi=150)
    plt.close()
    print("Grafica guardada: monthly_climatology.png")


def plot_annual_totals(df):
    """Serie temporal de precipitacion total anual con tendencia."""
    annual = df.dropna(subset=["precipitation_mm"]).groupby("year")["precipitation_mm"].sum()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(annual.index, annual.values, color="#3498db", alpha=0.7, edgecolor="white")

    # Tendencia lineal
    z = np.polyfit(annual.index, annual.values, 1)
    p = np.poly1d(z)
    ax.plot(annual.index, p(annual.index), "r--", linewidth=2,
            label=f"Tendencia: {z[0]:+.1f} mm/anio")

    mean_val = annual.mean()
    ax.axhline(mean_val, color="gray", linestyle=":", linewidth=1)
    ax.text(annual.index[-1] + 1, mean_val, f"Media: {mean_val:.0f} mm",
            va="center", fontsize=9, color="gray")

    ax.set_xlabel("Anio")
    ax.set_ylabel("Precipitacion total anual (mm)")
    ax.set_title(f"Precipitacion Total Anual\n{LOCATION_NAME}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "annual_totals.png", dpi=150)
    plt.close()
    print("Grafica guardada: annual_totals.png")


def plot_daily_timeseries(df):
    """Precipitacion diaria de los ultimos 5 anios."""
    recent = df[df["year"] >= df["year"].max() - 4].copy()
    event_date = pd.Timestamp(EVENT_DATE)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(recent["date"], recent["precipitation_mm"], color="#3498db",
           alpha=0.6, width=1.0, edgecolor="none")

    # Marcar evento
    ax.axvline(event_date, color="red", linestyle="--", linewidth=2, label=f"Evento: {EVENT_DATE}")

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylabel("Precipitacion diaria (mm)")
    ax.set_title(f"Precipitacion Diaria (ultimos 5 anios)\n{LOCATION_NAME}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "daily_timeseries.png", dpi=150)
    plt.close()
    print("Grafica guardada: daily_timeseries.png")


if __name__ == "__main__":
    ensure_dirs()
    print("=" * 60)
    print("ANALISIS EXPLORATORIO DE PRECIPITACION")
    print("=" * 60)

    df = load_data()
    print_summary(df)
    bad_years = report_data_quality(df)

    print("\nGenerando graficas...")
    plot_monthly_climatology(df)
    plot_annual_totals(df)
    plot_daily_timeseries(df)

    print("\nAnalisis exploratorio completado.")
