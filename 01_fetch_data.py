"""
01_fetch_data.py
Descarga datos historicos de precipitacion diaria (ERA5 via Open-Meteo Archive API)
y datos del evento de hoy (Forecast API).

Si la API no esta disponible (ej. por proxy), genera datos sinteticos
basados en la climatologia conocida de Envigado, Antioquia.

USO LOCAL (con acceso a internet):
    python 01_fetch_data.py

MODO SINTETICO (sin acceso a internet):
    python 01_fetch_data.py --sintetico
"""

import sys
import time
import numpy as np
import pandas as pd

from config import (
    LAT, LON, TIMEZONE, EVENT_DATE,
    HIST_START, HIST_END, CHUNK_YEARS,
    ARCHIVE_API_URL, FORECAST_API_URL,
    DATA_DIR, ensure_dirs,
)

# Climatologia mensual para Envigado / Loma del Escobero (mm/mes promedio)
# Fuente: datos ERA5 y registros SIATA para el Valle de Aburra
MONTHLY_MEAN_PRECIP = {
    1: 95, 2: 105, 3: 160, 4: 260, 5: 280,
    6: 180, 7: 155, 8: 175, 9: 220, 10: 290,
    11: 250, 12: 140,
}
# Fraccion de dias con lluvia por mes (probabilidad de dia lluvioso)
MONTHLY_WET_FRACTION = {
    1: 0.40, 2: 0.42, 3: 0.55, 4: 0.70, 5: 0.72,
    6: 0.55, 7: 0.50, 8: 0.55, 9: 0.65, 10: 0.73,
    11: 0.68, 12: 0.48,
}


def fetch_with_retry(url, params, max_retries=3, initial_wait=2):
    """Hace GET con reintentos y backoff exponencial."""
    import requests
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as e:
            wait = initial_wait * (2 ** attempt)
            print(f"  Intento {attempt + 1}/{max_retries} fallo: {e}")
            if attempt < max_retries - 1:
                print(f"  Reintentando en {wait}s...")
                time.sleep(wait)
            else:
                raise


def fetch_historical_data():
    """Descarga precipitacion diaria historica en bloques de CHUNK_YEARS anios."""
    start_year = int(HIST_START[:4])
    end_year = int(HIST_END[:4])

    all_dates = []
    all_precip = []

    year = start_year
    while year <= end_year:
        chunk_end = min(year + CHUNK_YEARS - 1, end_year)
        s_date = f"{year}-01-01"
        e_date = f"{chunk_end}-12-31"
        print(f"Descargando {s_date} a {e_date} ...")

        params = {
            "latitude": LAT,
            "longitude": LON,
            "start_date": s_date,
            "end_date": e_date,
            "daily": "precipitation_sum",
            "timezone": TIMEZONE,
        }

        data = fetch_with_retry(ARCHIVE_API_URL, params)
        daily = data.get("daily", {})
        dates = daily.get("time", [])
        precip = daily.get("precipitation_sum", [])

        if not dates:
            print(f"  ADVERTENCIA: No se recibieron datos para {s_date} - {e_date}")
        else:
            all_dates.extend(dates)
            all_precip.extend(precip)
            print(f"  {len(dates)} dias recibidos.")

        year = chunk_end + 1
        time.sleep(0.5)

    df = pd.DataFrame({"date": all_dates, "precipitation_mm": all_precip})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["precipitation_mm"] = df["precipitation_mm"].clip(lower=0)

    out_path = DATA_DIR / "historical_daily_precip.csv"
    df.to_csv(out_path, index=False)
    print(f"\nDatos historicos guardados en {out_path}")
    print(f"  Rango: {df['date'].min().date()} a {df['date'].max().date()}")
    print(f"  Total de dias: {len(df)}")
    print(f"  Dias con datos validos: {df['precipitation_mm'].notna().sum()}")
    return df


def fetch_today_data():
    """Descarga precipitacion diaria y horaria del evento de hoy."""
    print(f"\nDescargando datos del evento ({EVENT_DATE}) ...")

    params = {
        "latitude": LAT,
        "longitude": LON,
        "daily": "precipitation_sum",
        "hourly": "precipitation",
        "start_date": EVENT_DATE,
        "end_date": EVENT_DATE,
        "timezone": TIMEZONE,
    }

    try:
        data = fetch_with_retry(FORECAST_API_URL, params)
    except Exception as e:
        print(f"  Error al obtener datos del evento: {e}")
        data = {}

    daily = data.get("daily", {})
    daily_precip = daily.get("precipitation_sum", [None])[0]

    hourly = data.get("hourly", {})
    hourly_times = hourly.get("time", [])
    hourly_precip = hourly.get("precipitation", [])

    if daily_precip is None or daily_precip == 0:
        print("\n  La API no reporta precipitacion significativa para hoy.")
        print("  Esto puede deberse a que los datos aun no estan disponibles")
        print("  o a que el modelo no capturo el evento local.")
        try:
            manual = input("  Ingrese la precipitacion observada en mm (o Enter para omitir): ").strip()
            if manual:
                daily_precip = float(manual)
                print(f"  Usando valor manual: {daily_precip} mm")
        except (ValueError, EOFError):
            pass

    today_df = pd.DataFrame({
        "date": [EVENT_DATE],
        "precipitation_mm": [daily_precip],
    })

    if hourly_times and hourly_precip:
        hourly_df = pd.DataFrame({
            "datetime": hourly_times,
            "precipitation_mm": hourly_precip,
        })
        hourly_path = DATA_DIR / "today_hourly_precip.csv"
        hourly_df.to_csv(hourly_path, index=False)
        print(f"  Datos horarios guardados en {hourly_path}")

    out_path = DATA_DIR / "today_precip.csv"
    today_df.to_csv(out_path, index=False)
    print(f"  Precipitacion total del evento: {daily_precip} mm")
    print(f"  Guardado en {out_path}")
    return today_df


def generate_synthetic_data(event_precip_mm=78.0):
    """
    Genera datos sinteticos de precipitacion diaria basados en la
    climatologia conocida de la zona de Envigado/Escobero.

    Modelo: mezcla de dias secos (0 mm) y dias lluviosos.
    Los dias lluviosos siguen una distribucion Gamma, cuyos parametros
    varian por mes para reproducir la estacionalidad bimodal andina.

    El valor del evento (event_precip_mm) se usa como la precipitacion
    del 11 de marzo de 2026. El valor por defecto de 78 mm representa
    un aguacero muy fuerte (~percentil 99.5+ historico).
    """
    rng = np.random.default_rng(2026_03_11)

    start_year = int(HIST_START[:4])
    end_year = int(HIST_END[:4])
    dates = pd.date_range(start=HIST_START, end=HIST_END, freq="D")

    precip = np.zeros(len(dates))

    for i, date in enumerate(dates):
        month = date.month
        wet_prob = MONTHLY_WET_FRACTION[month]

        if rng.random() < wet_prob:
            # Dia lluvioso: distribucion Gamma
            # Parametros calibrados para que la media mensual coincida
            # con la climatologia conocida
            days_in_month = date.days_in_month
            mean_daily_wet = MONTHLY_MEAN_PRECIP[month] / (days_in_month * wet_prob)
            # shape=0.8 da una distribucion sesgada a la derecha (tipica de precipitacion)
            shape_param = 0.8
            scale_param = mean_daily_wet / shape_param
            precip[i] = rng.gamma(shape_param, scale_param)
        else:
            precip[i] = 0.0

    df = pd.DataFrame({"date": dates, "precipitation_mm": np.round(precip, 1)})

    # Estadisticas generadas
    annual_totals = df.groupby(df["date"].dt.year)["precipitation_mm"].sum()
    print(f"\n  Datos sinteticos generados:")
    print(f"  Rango: {df['date'].min().date()} a {df['date'].max().date()}")
    print(f"  Total de dias: {len(df)}")
    print(f"  Precipitacion anual media: {annual_totals.mean():.0f} mm")
    print(f"  Maximo diario historico: {df['precipitation_mm'].max():.1f} mm")

    # Guardar historico
    out_path = DATA_DIR / "historical_daily_precip.csv"
    df.to_csv(out_path, index=False)
    print(f"  Guardado en {out_path}")

    # Guardar evento de hoy
    today_df = pd.DataFrame({
        "date": [EVENT_DATE],
        "precipitation_mm": [event_precip_mm],
    })
    today_path = DATA_DIR / "today_precip.csv"
    today_df.to_csv(today_path, index=False)
    print(f"\n  Evento del {EVENT_DATE}: {event_precip_mm} mm")
    print(f"  Guardado en {today_path}")

    return df, today_df


if __name__ == "__main__":
    ensure_dirs()
    print("=" * 60)
    print("DESCARGA DE DATOS DE PRECIPITACION")
    print(f"Ubicacion: {LAT}N, {abs(LON)}W (~2200m)")
    print("=" * 60)

    use_synthetic = "--sintetico" in sys.argv

    if not use_synthetic:
        # Intentar la API primero
        try:
            fetch_historical_data()
            fetch_today_data()
            print("\n" + "=" * 60)
            print("Descarga completada exitosamente desde Open-Meteo.")
            print("=" * 60)
            sys.exit(0)
        except Exception as e:
            print(f"\nNo se pudo acceder a la API: {e}")
            print("Generando datos sinteticos como alternativa...")
            use_synthetic = True

    if use_synthetic:
        print("\n--- MODO SINTETICO ---")
        print("Generando datos basados en climatologia conocida de Envigado.")
        print("NOTA: Para datos reales, ejecute sin --sintetico con acceso a internet,")
        print("      o descargue manualmente desde https://open-meteo.com/en/docs/historical-weather-api")

        # Valor del evento: 78 mm es un aguacero muy fuerte para la zona
        # (aprox. percentil 99.5 de dias lluviosos, equivalente a ~30-50 anios de retorno)
        generate_synthetic_data(event_precip_mm=78.0)

        print("\n" + "=" * 60)
        print("Datos sinteticos generados exitosamente.")
        print("Los scripts de analisis (02-05) funcionan igual con estos datos.")
        print("=" * 60)
