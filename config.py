"""
Configuracion central para el analisis de probabilidad de lluvia.
Parcelacion San Luis, Loma del Escobero, Envigado, Antioquia.
"""

from pathlib import Path

# --- Ubicacion ---
LAT = 6.15
LON = -75.53
ELEVATION = 2200  # metros sobre el nivel del mar (aprox.)
LOCATION_NAME = "Parcelacion San Luis, Loma del Escobero, Envigado, Antioquia"

# --- Zona horaria ---
TIMEZONE = "America/Bogota"

# --- Evento de interes ---
EVENT_DATE = "2026-03-11"

# --- Rango historico ---
HIST_START = "1950-01-01"
HIST_END = "2025-12-31"
CHUNK_YEARS = 5  # tamanio de bloques para consultas a la API

# --- Minimo de dias validos por anio para incluir en maximos anuales ---
MIN_VALID_DAYS = 330

# --- APIs de Open-Meteo ---
ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"

# --- Directorios de salida ---
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "plots"


def ensure_dirs():
    """Crea los directorios de datos y graficas si no existen."""
    DATA_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)
