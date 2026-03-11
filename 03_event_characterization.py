"""
03_event_characterization.py
Caracteriza el evento del 11 de marzo de 2026 en el contexto
de la distribucion historica de precipitacion diaria.
Incluye analisis del contexto meteorologico (granizo/conveccion).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from config import (
    LOCATION_NAME, EVENT_DATE, ELEVATION,
    EVENT_HAD_HAIL, EVENT_DESCRIPTION,
    DATA_DIR, PLOTS_DIR, ensure_dirs,
)

sns.set_theme(style="whitegrid", palette="muted")


def load_data():
    hist = pd.read_csv(DATA_DIR / "historical_daily_precip.csv", parse_dates=["date"])
    today = pd.read_csv(DATA_DIR / "today_precip.csv")
    event_precip = today["precipitation_mm"].iloc[0]
    return hist, event_precip


def characterize_event(hist, event_precip):
    valid = hist.dropna(subset=["precipitation_mm"])
    all_precip = valid["precipitation_mm"].values

    # Percentil general
    pct_all = (all_precip <= event_precip).mean() * 100

    # Percentil solo en marzo
    march = valid[valid["date"].dt.month == 3]["precipitation_mm"].values
    pct_march = (march <= event_precip).mean() * 100

    # Ranking
    sorted_desc = np.sort(all_precip)[::-1]
    rank = np.searchsorted(-sorted_desc, -event_precip) + 1

    # Comparacion con promedios de marzo
    march_daily_mean = march.mean()
    march_monthly_mean = valid[valid["date"].dt.month == 3].groupby(
        valid[valid["date"].dt.month == 3]["date"].dt.year
    )["precipitation_mm"].sum().mean()

    print("\n--- CARACTERIZACION DEL EVENTO ---")
    print(f"Fecha: {EVENT_DATE}")
    print(f"Precipitacion del evento: {event_precip:.1f} mm")
    print(f"\nPercentil en toda la serie historica: {pct_all:.1f}%")
    print(f"Percentil entre dias de marzo: {pct_march:.1f}%")
    print(f"Ranking: #{rank} entre {len(all_precip)} dias registrados")
    print(f"\nPrecipitacion media diaria en marzo: {march_daily_mean:.1f} mm")
    print(f"Precipitacion media mensual de marzo: {march_monthly_mean:.0f} mm")
    print(f"El evento equivale a {event_precip / march_daily_mean:.1f}x "
          f"la precipitacion media diaria de marzo")
    print(f"El evento equivale al {event_precip / march_monthly_mean * 100:.0f}% "
          f"del total mensual promedio de marzo")

    # Contexto meteorologico: granizo
    if EVENT_HAD_HAIL:
        print_hail_context(event_precip)


def print_hail_context(event_precip):
    """Imprime analisis del contexto meteorologico asociado al granizo."""
    print("\n--- CONTEXTO METEOROLOGICO: GRANIZO ---")
    print(f"Descripcion: {EVENT_DESCRIPTION}")
    print()
    print("Implicaciones del granizo para el analisis:")
    print(f"  1. INTENSIDAD: El granizo confirma que esta fue una tormenta")
    print(f"     convectiva severa, no solo lluvia estratiforme. La intensidad")
    print(f"     instantanea (mm/hora) probablemente fue muy alta (>50 mm/h).")
    print(f"  2. ESCALA ESPACIAL: Las tormentas convectivas con granizo son")
    print(f"     localizadas (5-20 km). La precipitacion en la Parcelacion")
    print(f"     San Luis pudo ser significativamente mayor que en estaciones")
    print(f"     meteorologicas cercanas o que el promedio del grid ERA5 (25 km).")
    print(f"  3. SUBESTIMACION: Los datos ERA5/reanálisis probablemente")
    print(f"     subestiman este tipo de eventos locales. El periodo de retorno")
    print(f"     real del evento puede ser MAYOR que el estimado con ERA5.")
    print(f"  4. RIESGO COMPUESTO: Granizo + lluvia intensa = mayor riesgo de")
    print(f"     dano (impacto mecanico + inundacion subita + obstruccion de")
    print(f"     drenajes por acumulacion de hielo).")
    print(f"  5. ALTITUD ({ELEVATION}m): A esta elevacion, la distancia al nivel")
    print(f"     de congelacion es menor, facilitando que el granizo llegue")
    print(f"     al suelo sin derretirse completamente.")


def plot_histogram(hist, event_precip):
    """Histograma de precipitacion diaria con el evento marcado."""
    valid = hist.dropna(subset=["precipitation_mm"])
    rainy = valid[valid["precipitation_mm"] >= 1]["precipitation_mm"]

    pct = (valid["precipitation_mm"] <= event_precip).mean() * 100

    hail_label = " + granizo" if EVENT_HAD_HAIL else ""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(rainy, bins=80, color="#3498db", alpha=0.7, edgecolor="white", log=True)
    ax.axvline(event_precip, color="red", linestyle="--", linewidth=2.5,
               label=f"Evento {EVENT_DATE}: {event_precip:.1f} mm{hail_label}")
    ax.annotate(f"Percentil {pct:.1f}%",
                xy=(event_precip, ax.get_ylim()[1] * 0.3),
                xytext=(event_precip + 5, ax.get_ylim()[1] * 0.5),
                fontsize=11, color="red",
                arrowprops=dict(arrowstyle="->", color="red"))

    if EVENT_HAD_HAIL:
        ax.annotate("Con granizo\n(conveccion severa)",
                     xy=(event_precip, ax.get_ylim()[1] * 0.01),
                     xytext=(event_precip - 30, ax.get_ylim()[1] * 0.005),
                     fontsize=9, color="darkred", fontstyle="italic",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffe0e0", alpha=0.8))

    ax.set_xlabel("Precipitacion diaria (mm)")
    ax.set_ylabel("Frecuencia (escala log)")
    ax.set_title(f"Distribucion de Precipitacion Diaria (dias con >=1 mm)\n{LOCATION_NAME}")
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "event_histogram.png", dpi=150)
    plt.close()
    print("Grafica guardada: event_histogram.png")


def plot_ecdf(hist, event_precip):
    """CDF empirica con probabilidad de excedencia."""
    valid = hist.dropna(subset=["precipitation_mm"])
    sorted_vals = np.sort(valid["precipitation_mm"].values)
    ecdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

    p_exceedance = 1 - (sorted_vals <= event_precip).mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sorted_vals, ecdf, color="#3498db", linewidth=1.5)
    ax.axvline(event_precip, color="red", linestyle="--", linewidth=2)
    ax.axhline(1 - p_exceedance, color="red", linestyle=":", alpha=0.5)

    ax.fill_between(sorted_vals[sorted_vals >= event_precip],
                     ecdf[sorted_vals >= event_precip], 1,
                     alpha=0.15, color="red", label=f"P(excedencia) = {p_exceedance:.4f}")

    ax.set_xlabel("Precipitacion diaria (mm)")
    ax.set_ylabel("Probabilidad acumulada")
    ax.set_title(f"CDF Empirica de Precipitacion Diaria\n{LOCATION_NAME}")
    ax.legend(fontsize=11)

    # Anotar evento
    cdf_at_event = (sorted_vals <= event_precip).mean()
    ax.annotate(f"{event_precip:.1f} mm\nF(x) = {cdf_at_event:.4f}",
                xy=(event_precip, cdf_at_event),
                xytext=(event_precip + 10, cdf_at_event - 0.1),
                fontsize=10, color="red",
                arrowprops=dict(arrowstyle="->", color="red"))

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "event_ecdf.png", dpi=150)
    plt.close()
    print("Grafica guardada: event_ecdf.png")


if __name__ == "__main__":
    ensure_dirs()
    print("=" * 60)
    print("CARACTERIZACION DEL EVENTO DE PRECIPITACION")
    print("=" * 60)

    hist, event_precip = load_data()

    if event_precip is None or np.isnan(event_precip):
        print("ERROR: No hay dato de precipitacion para el evento.")
        print("Ejecute primero 01_fetch_data.py y verifique today_precip.csv")
        exit(1)

    characterize_event(hist, event_precip)

    print("\nGenerando graficas...")
    plot_histogram(hist, event_precip)
    plot_ecdf(hist, event_precip)

    print("\nCaracterizacion del evento completada.")
