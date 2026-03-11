"""
05_generate_report_plots.py
Genera un panel resumen 2x2 combinando las graficas mas informativas.
"""

import pandas as pd
import numpy as np
from scipy.stats import genextreme, gumbel_r
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from config import (
    LAT, LON, ELEVATION, LOCATION_NAME, EVENT_DATE, MIN_VALID_DAYS,
    DATA_DIR, PLOTS_DIR, ensure_dirs,
)

sns.set_theme(style="whitegrid", palette="muted")


def load_all_data():
    hist = pd.read_csv(DATA_DIR / "historical_daily_precip.csv", parse_dates=["date"])
    today = pd.read_csv(DATA_DIR / "today_precip.csv")
    event_precip = today["precipitation_mm"].iloc[0]
    return hist, event_precip


def get_annual_maxima(hist):
    valid = hist.dropna(subset=["precipitation_mm"]).copy()
    valid["year"] = valid["date"].dt.year
    year_counts = valid.groupby("year").size()
    good_years = year_counts[year_counts >= MIN_VALID_DAYS].index
    return valid[valid["year"].isin(good_years)].groupby("year")["precipitation_mm"].max()


def main():
    ensure_dirs()
    hist, event_precip = load_all_data()
    annual_maxima = get_annual_maxima(hist)
    data = annual_maxima.values

    # Ajustar distribuciones
    c, loc, scale = genextreme.fit(data, 0)
    loc_g, scale_g = gumbel_r.fit(data)

    # Periodo de retorno del evento
    p_gev = 1 - genextreme.cdf(event_precip, c, loc=loc, scale=scale)
    t_event = 1 / p_gev if p_gev > 0 else float("inf")

    # Bootstrap para IC
    rng = np.random.default_rng(42)
    return_periods = np.array([2, 5, 10, 25, 50, 100, 200, 500])
    boot_levels = []
    for _ in range(1000):
        sample = rng.choice(data, size=len(data), replace=True)
        try:
            cp, lp, sp = genextreme.fit(sample, 0)
            boot_levels.append(genextreme.ppf(1 - 1 / return_periods, cp, loc=lp, scale=sp))
        except Exception:
            pass
    boot_levels = np.array(boot_levels)
    ci_lo = np.percentile(boot_levels, 2.5, axis=0)
    ci_hi = np.percentile(boot_levels, 97.5, axis=0)

    # --- CREAR PANEL 2x2 ---
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.32, wspace=0.28)

    # (A) Climatologia mensual
    ax1 = fig.add_subplot(gs[0, 0])
    valid = hist.dropna(subset=["precipitation_mm"]).copy()
    valid["year"] = valid["date"].dt.year
    valid["month"] = valid["date"].dt.month
    monthly_totals = valid.groupby(["year", "month"])["precipitation_mm"].sum().reset_index()
    stats = monthly_totals.groupby("month")["precipitation_mm"].agg(["mean", "std"])
    months = np.arange(1, 13)
    month_names = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                   "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    colors = ["#e74c3c" if m == 3 else "#3498db" for m in months]
    ax1.bar(months, stats["mean"], color=colors, edgecolor="white", width=0.7)
    ax1.errorbar(months, stats["mean"], yerr=stats["std"], fmt="none",
                 ecolor="gray", capsize=3)
    ax1.set_xticks(months)
    ax1.set_xticklabels(month_names, fontsize=9)
    ax1.set_ylabel("Precipitacion mensual (mm)")
    ax1.set_title("(A) Climatologia Mensual", fontweight="bold")

    # (B) Histograma con evento
    ax2 = fig.add_subplot(gs[0, 1])
    rainy = valid[valid["precipitation_mm"] >= 1]["precipitation_mm"]
    pct = (valid["precipitation_mm"] <= event_precip).mean() * 100
    ax2.hist(rainy, bins=80, color="#3498db", alpha=0.7, edgecolor="white", log=True)
    ax2.axvline(event_precip, color="red", linestyle="--", linewidth=2.5)
    ax2.annotate(f"{event_precip:.1f} mm\nPercentil {pct:.1f}%",
                 xy=(event_precip, ax2.get_ylim()[1] * 0.01),
                 xytext=(event_precip + 8, ax2.get_ylim()[1] * 0.05),
                 fontsize=10, color="red",
                 arrowprops=dict(arrowstyle="->", color="red"))
    ax2.set_xlabel("Precipitacion diaria (mm)")
    ax2.set_ylabel("Frecuencia (log)")
    ax2.set_title("(B) Distribucion Diaria + Evento", fontweight="bold")

    # (C) Curva de periodos de retorno
    ax3 = fig.add_subplot(gs[1, 0])
    n = len(data)
    sorted_data = np.sort(data)[::-1]
    ranks = np.arange(1, n + 1)
    t_emp = (n + 1) / ranks
    t_smooth = np.logspace(np.log10(1.01), np.log10(1000), 300)
    rl_smooth = genextreme.ppf(1 - 1 / t_smooth, c, loc=loc, scale=scale)
    rl_gum = gumbel_r.ppf(1 - 1 / t_smooth, loc=loc_g, scale=scale_g)

    ax3.fill_between(return_periods, ci_lo, ci_hi, alpha=0.2, color="#3498db")
    ax3.plot(t_smooth, rl_smooth, "b-", linewidth=2, label="GEV")
    ax3.plot(t_smooth, rl_gum, "g--", linewidth=1.5, alpha=0.7, label="Gumbel")
    ax3.scatter(t_emp, sorted_data, c="red", edgecolor="white", s=40, zorder=5)
    ax3.axhline(event_precip, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    ax3.annotate(f"T = {t_event:.0f} anios",
                 xy=(t_event, event_precip),
                 xytext=(t_event * 2.5, event_precip * 0.8),
                 fontsize=10, color="red", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", edgecolor="red"),
                 arrowprops=dict(arrowstyle="->", color="red"))
    ax3.set_xscale("log")
    ax3.set_xlabel("Periodo de retorno (anios)")
    ax3.set_ylabel("Precipitacion max. diaria (mm)")
    ax3.set_title("(C) Curva de Periodos de Retorno", fontweight="bold")
    ax3.legend(loc="upper left", fontsize=9)

    # (D) Serie de maximos anuales
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.bar(annual_maxima.index, annual_maxima.values, color="#3498db",
            alpha=0.7, edgecolor="white")
    ax4.axhline(event_precip, color="red", linestyle="--", linewidth=2,
                label=f"Evento: {event_precip:.1f} mm")
    ax4.axhline(annual_maxima.mean(), color="gray", linestyle=":", linewidth=1)
    ax4.set_xlabel("Anio")
    ax4.set_ylabel("Precipitacion max. diaria (mm)")
    ax4.set_title("(D) Maximos Anuales de Precipitacion", fontweight="bold")
    ax4.legend(fontsize=9)

    # Titulo principal
    fig.suptitle(
        f"Analisis de Probabilidad de Lluvia\n"
        f"{LOCATION_NAME} ({LAT:.2f}°N, {abs(LON):.2f}°W, {ELEVATION}m)",
        fontsize=14, fontweight="bold", y=1.0
    )

    plt.savefig(PLOTS_DIR / "summary_panel.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Panel resumen guardado: plots/summary_panel.png")


if __name__ == "__main__":
    print("=" * 60)
    print("GENERANDO PANEL RESUMEN")
    print("=" * 60)
    main()
    print("Completado.")
