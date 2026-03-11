"""
04_extreme_value_analysis.py
Analisis de valores extremos: ajuste GEV y Gumbel a maximos anuales,
periodos de retorno, intervalos de confianza bootstrap.
"""

import pandas as pd
import numpy as np
from scipy.stats import genextreme, gumbel_r, kstest
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    LOCATION_NAME, EVENT_DATE, MIN_VALID_DAYS,
    EVENT_HAD_HAIL,
    DATA_DIR, PLOTS_DIR, ensure_dirs,
)

sns.set_theme(style="whitegrid", palette="muted")

RETURN_PERIODS = np.array([2, 5, 10, 25, 50, 100, 200, 500])
N_BOOTSTRAP = 1000
SENSITIVITY_VALUES = [30, 40, 50, 60, 80, 100, 120]


def load_data():
    hist = pd.read_csv(DATA_DIR / "historical_daily_precip.csv", parse_dates=["date"])
    today = pd.read_csv(DATA_DIR / "today_precip.csv")
    event_precip = today["precipitation_mm"].iloc[0]
    return hist, event_precip


def extract_annual_maxima(hist):
    """Extrae maximos anuales, excluyendo anios con datos insuficientes."""
    valid = hist.dropna(subset=["precipitation_mm"]).copy()
    valid["year"] = valid["date"].dt.year

    # Contar dias validos por anio
    year_counts = valid.groupby("year").size()
    good_years = year_counts[year_counts >= MIN_VALID_DAYS].index

    # Maximos anuales solo de anios con datos suficientes
    annual_max = valid[valid["year"].isin(good_years)].groupby("year")["precipitation_mm"].max()

    print(f"\nMaximos anuales extraidos: {len(annual_max)} anios")
    print(f"  Anios excluidos por datos insuficientes: "
          f"{len(year_counts) - len(good_years)}")
    print(f"  Rango: {annual_max.index.min()} - {annual_max.index.max()}")
    print(f"  Media de maximos anuales: {annual_max.mean():.1f} mm")
    print(f"  Maximo absoluto: {annual_max.max():.1f} mm ({annual_max.idxmax()})")

    return annual_max


def fit_distributions(annual_maxima):
    """Ajusta GEV y Gumbel a los maximos anuales."""
    data = annual_maxima.values

    # GEV
    c_gev, loc_gev, scale_gev = genextreme.fit(data, 0)
    # Gumbel (caso especial de GEV con shape=0)
    loc_gum, scale_gum = gumbel_r.fit(data)

    print("\n--- PARAMETROS AJUSTADOS ---")
    print(f"GEV:    shape (c) = {c_gev:.4f}, loc (mu) = {loc_gev:.2f}, scale (sigma) = {scale_gev:.2f}")
    print(f"  Nota: en scipy, c < 0 => cola pesada (Frechet), tipico para precipitacion")
    print(f"Gumbel: loc (mu) = {loc_gum:.2f}, scale (sigma) = {scale_gum:.2f}")

    # Bondad de ajuste: K-S test
    ks_gev = kstest(data, "genextreme", args=(c_gev, loc_gev, scale_gev))
    ks_gum = kstest(data, "gumbel_r", args=(loc_gum, scale_gum))

    print(f"\n--- BONDAD DE AJUSTE (Kolmogorov-Smirnov) ---")
    print(f"GEV:    D = {ks_gev.statistic:.4f}, p-valor = {ks_gev.pvalue:.4f}")
    print(f"Gumbel: D = {ks_gum.statistic:.4f}, p-valor = {ks_gum.pvalue:.4f}")

    # AIC/BIC
    n = len(data)
    ll_gev = np.sum(genextreme.logpdf(data, c_gev, loc=loc_gev, scale=scale_gev))
    ll_gum = np.sum(gumbel_r.logpdf(data, loc=loc_gum, scale=scale_gum))
    aic_gev = 2 * 3 - 2 * ll_gev
    aic_gum = 2 * 2 - 2 * ll_gum
    bic_gev = 3 * np.log(n) - 2 * ll_gev
    bic_gum = 2 * np.log(n) - 2 * ll_gum

    print(f"\n--- CRITERIOS DE INFORMACION ---")
    print(f"GEV:    AIC = {aic_gev:.1f}, BIC = {bic_gev:.1f}")
    print(f"Gumbel: AIC = {aic_gum:.1f}, BIC = {bic_gum:.1f}")
    best = "GEV" if aic_gev < aic_gum else "Gumbel"
    print(f"Mejor modelo por AIC: {best}")

    return (c_gev, loc_gev, scale_gev), (loc_gum, scale_gum)


def compute_return_levels(gev_params, gum_params):
    """Calcula niveles de retorno para periodos estandar."""
    c, loc, scale = gev_params
    loc_g, scale_g = gum_params

    rl_gev = genextreme.ppf(1 - 1 / RETURN_PERIODS, c, loc=loc, scale=scale)
    rl_gum = gumbel_r.ppf(1 - 1 / RETURN_PERIODS, loc=loc_g, scale=scale_g)

    print(f"\n--- PERIODOS DE RETORNO ---")
    print(f"{'T (anios)':>10} | {'GEV (mm)':>10} | {'Gumbel (mm)':>12}")
    print("-" * 38)
    for t, g, gb in zip(RETURN_PERIODS, rl_gev, rl_gum):
        print(f"{t:>10} | {g:>10.1f} | {gb:>12.1f}")

    return rl_gev, rl_gum


def bootstrap_confidence(annual_maxima, n_bootstrap=N_BOOTSTRAP):
    """Intervalos de confianza bootstrap para niveles de retorno GEV."""
    data = annual_maxima.values
    n = len(data)
    boot_levels = np.zeros((n_bootstrap, len(RETURN_PERIODS)))

    rng = np.random.default_rng(42)
    for i in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        try:
            c, loc, scale = genextreme.fit(sample, 0)
            boot_levels[i] = genextreme.ppf(1 - 1 / RETURN_PERIODS, c, loc=loc, scale=scale)
        except Exception:
            boot_levels[i] = np.nan

    # Remover filas con NaN
    boot_levels = boot_levels[~np.isnan(boot_levels).any(axis=1)]

    ci_lower = np.percentile(boot_levels, 2.5, axis=0)
    ci_upper = np.percentile(boot_levels, 97.5, axis=0)

    print(f"\n--- INTERVALOS DE CONFIANZA 95% (Bootstrap, {len(boot_levels)} muestras) ---")
    print(f"{'T (anios)':>10} | {'IC inferior':>12} | {'IC superior':>12}")
    print("-" * 40)
    for t, lo, hi in zip(RETURN_PERIODS, ci_lower, ci_upper):
        print(f"{t:>10} | {lo:>12.1f} | {hi:>12.1f}")

    return ci_lower, ci_upper


def event_return_period(event_precip, gev_params, gum_params):
    """Calcula el periodo de retorno del evento."""
    c, loc, scale = gev_params
    loc_g, scale_g = gum_params

    p_gev = 1 - genextreme.cdf(event_precip, c, loc=loc, scale=scale)
    p_gum = 1 - gumbel_r.cdf(event_precip, loc=loc_g, scale=scale_g)

    t_gev = 1 / p_gev if p_gev > 0 else float("inf")
    t_gum = 1 / p_gum if p_gum > 0 else float("inf")

    print(f"\n--- PERIODO DE RETORNO DEL EVENTO ---")
    print(f"Precipitacion del evento: {event_precip:.1f} mm")
    print(f"GEV:    P(excedencia) = {p_gev:.6f} ({p_gev * 100:.4f}%), "
          f"T = {t_gev:.1f} anios")
    print(f"Gumbel: P(excedencia) = {p_gum:.6f} ({p_gum * 100:.4f}%), "
          f"T = {t_gum:.1f} anios")
    print(f"\nInterpretacion: Un evento de {event_precip:.1f} mm tiene una probabilidad")
    print(f"de ~{p_gev * 100:.2f}% de ocurrir (o ser superado) en cualquier anio dado.")

    if EVENT_HAD_HAIL:
        print(f"\n--- NOTA SOBRE EL GRANIZO ---")
        print(f"El evento incluyo granizo, lo cual indica conveccion severa.")
        print(f"CAVEAT IMPORTANTE: Los datos ERA5 (resolucion ~25 km) tienden a")
        print(f"subestimar eventos convectivos localizados. La precipitacion real")
        print(f"en la Parcelacion San Luis pudo haber sido significativamente mayor")
        print(f"que lo que captura el grid de reanálisis. Por lo tanto:")
        print(f"  - El periodo de retorno estimado ({t_gev:.1f} anios) es un LIMITE")
        print(f"    INFERIOR. El periodo de retorno real puede ser 2-5x mayor.")
        print(f"  - Para un analisis mas preciso, se recomienda usar datos de")
        print(f"    estaciones locales (SIATA) en lugar de reanálisis ERA5.")

    return t_gev, t_gum


def sensitivity_table(gev_params, gum_params):
    """Tabla de periodos de retorno para diferentes magnitudes."""
    c, loc, scale = gev_params

    print(f"\n--- TABLA DE SENSIBILIDAD ---")
    print(f"{'Precipitacion (mm)':>20} | {'P(excedencia)':>14} | {'T GEV (anios)':>14}")
    print("-" * 55)
    for val in SENSITIVITY_VALUES:
        p = 1 - genextreme.cdf(val, c, loc=loc, scale=scale)
        t = 1 / p if p > 0 else float("inf")
        print(f"{val:>20} | {p * 100:>13.4f}% | {t:>14.1f}")


def plot_gev_fit(annual_maxima, gev_params, gum_params):
    """Histograma de maximos anuales + PDFs ajustadas + QQ-plot."""
    data = annual_maxima.values
    c, loc, scale = gev_params
    loc_g, scale_g = gum_params

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel izquierdo: histograma + PDFs
    ax = axes[0]
    x = np.linspace(data.min() * 0.7, data.max() * 1.5, 200)
    ax.hist(data, bins=15, density=True, color="#3498db", alpha=0.5,
            edgecolor="white", label="Datos observados")
    ax.plot(x, genextreme.pdf(x, c, loc=loc, scale=scale),
            "r-", linewidth=2, label=f"GEV (c={c:.3f})")
    ax.plot(x, gumbel_r.pdf(x, loc=loc_g, scale=scale_g),
            "g--", linewidth=2, label="Gumbel")
    ax.set_xlabel("Precipitacion maxima anual (mm)")
    ax.set_ylabel("Densidad")
    ax.set_title("Ajuste de Distribuciones a Maximos Anuales")
    ax.legend()

    # Panel derecho: QQ-plot para GEV
    ax = axes[1]
    sorted_data = np.sort(data)
    n = len(data)
    theoretical = genextreme.ppf(np.arange(1, n + 1) / (n + 1), c, loc=loc, scale=scale)
    ax.scatter(theoretical, sorted_data, c="#3498db", edgecolor="white", s=50, zorder=5)
    lims = [min(theoretical.min(), sorted_data.min()),
            max(theoretical.max(), sorted_data.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Linea 1:1")
    ax.set_xlabel("Cuantiles teoricos GEV (mm)")
    ax.set_ylabel("Cuantiles observados (mm)")
    ax.set_title("QQ-Plot (GEV)")
    ax.legend()

    plt.suptitle(LOCATION_NAME, fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "gev_fit.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Grafica guardada: gev_fit.png")


def plot_return_period_curve(annual_maxima, gev_params, gum_params,
                              ci_lower, ci_upper, event_precip, t_event_gev):
    """Curva de periodos de retorno con banda de confianza y evento."""
    data = annual_maxima.values
    c, loc, scale = gev_params
    loc_g, scale_g = gum_params
    n = len(data)

    # Posiciones empiricas (Weibull)
    sorted_data = np.sort(data)[::-1]
    ranks = np.arange(1, n + 1)
    t_empirical = (n + 1) / ranks

    # Curva teorica suave
    t_smooth = np.logspace(np.log10(1.01), np.log10(1000), 300)
    rl_gev_smooth = genextreme.ppf(1 - 1 / t_smooth, c, loc=loc, scale=scale)
    rl_gum_smooth = gumbel_r.ppf(1 - 1 / t_smooth, loc=loc_g, scale=scale_g)

    fig, ax = plt.subplots(figsize=(11, 6))

    # Banda de confianza
    ax.fill_between(RETURN_PERIODS, ci_lower, ci_upper,
                     alpha=0.2, color="#3498db", label="IC 95% (bootstrap)")

    # Curvas teoricas
    ax.plot(t_smooth, rl_gev_smooth, "b-", linewidth=2, label="GEV ajustada")
    ax.plot(t_smooth, rl_gum_smooth, "g--", linewidth=1.5, alpha=0.7, label="Gumbel ajustada")

    # Posiciones empiricas
    ax.scatter(t_empirical, sorted_data, c="red", edgecolor="white",
               s=50, zorder=5, label="Maximos anuales observados")

    # Evento de hoy
    ax.axhline(event_precip, color="red", linestyle="--", linewidth=2, alpha=0.7)
    ax.axvline(t_event_gev, color="red", linestyle=":", linewidth=1.5, alpha=0.5)

    # Anotacion del evento
    ax.annotate(
        f"Evento {EVENT_DATE}\n{event_precip:.1f} mm{' + granizo' if EVENT_HAD_HAIL else ''}\nT = {t_event_gev:.1f} anios",
        xy=(t_event_gev, event_precip),
        xytext=(t_event_gev * 2, event_precip * 0.75),
        fontsize=10, color="red", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="red"),
        arrowprops=dict(arrowstyle="->", color="red", linewidth=1.5),
    )

    ax.set_xscale("log")
    ax.set_xlabel("Periodo de retorno (anios)", fontsize=12)
    ax.set_ylabel("Precipitacion maxima diaria (mm)", fontsize=12)
    ax.set_title(f"Curva de Periodos de Retorno\n{LOCATION_NAME}", fontsize=13)
    ax.legend(loc="upper left")
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "return_period_curve.png", dpi=150)
    plt.close()
    print("Grafica guardada: return_period_curve.png")


def plot_annual_maxima_series(annual_maxima, event_precip):
    """Serie de maximos anuales con el evento marcado."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(annual_maxima.index, annual_maxima.values, color="#3498db",
           alpha=0.7, edgecolor="white")
    ax.axhline(event_precip, color="red", linestyle="--", linewidth=2,
               label=f"Evento {EVENT_DATE}: {event_precip:.1f} mm{' + granizo' if EVENT_HAD_HAIL else ''}")
    ax.axhline(annual_maxima.mean(), color="gray", linestyle=":", linewidth=1)
    ax.text(annual_maxima.index[-1] + 1, annual_maxima.mean(),
            f"Media: {annual_maxima.mean():.0f} mm", va="center", fontsize=9, color="gray")

    ax.set_xlabel("Anio")
    ax.set_ylabel("Precipitacion maxima diaria (mm)")
    ax.set_title(f"Serie de Maximos Anuales de Precipitacion\n{LOCATION_NAME}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "annual_maxima_series.png", dpi=150)
    plt.close()
    print("Grafica guardada: annual_maxima_series.png")


if __name__ == "__main__":
    ensure_dirs()
    print("=" * 60)
    print("ANALISIS DE VALORES EXTREMOS DE PRECIPITACION")
    print("=" * 60)

    hist, event_precip = load_data()

    if event_precip is None or np.isnan(event_precip):
        print("ERROR: No hay dato de precipitacion para el evento.")
        exit(1)

    # 1. Extraer maximos anuales
    annual_maxima = extract_annual_maxima(hist)

    # 2. Ajustar distribuciones
    gev_params, gum_params = fit_distributions(annual_maxima)

    # 3. Niveles de retorno
    rl_gev, rl_gum = compute_return_levels(gev_params, gum_params)

    # 4. Bootstrap
    ci_lower, ci_upper = bootstrap_confidence(annual_maxima)

    # 5. Periodo de retorno del evento
    t_gev, t_gum = event_return_period(event_precip, gev_params, gum_params)

    # 6. Tabla de sensibilidad
    sensitivity_table(gev_params, gum_params)

    # 7. Graficas
    print("\nGenerando graficas...")
    plot_gev_fit(annual_maxima, gev_params, gum_params)
    plot_return_period_curve(annual_maxima, gev_params, gum_params,
                             ci_lower, ci_upper, event_precip, t_gev)
    plot_annual_maxima_series(annual_maxima, event_precip)

    print("\nAnalisis de valores extremos completado.")
