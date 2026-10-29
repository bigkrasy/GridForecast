"""
GridForecast — синтетичен генератор на потребление на електроенергия.

Генерира 4 типа datasets:
  - stable:   без drift
  - sudden:   рязка промяна (COVID lockdown)
  - gradual:  постепенна промяна (EV adoption)
  - recurring: сезонен pattern

Всеки dataset е hourly DataFrame с колони:
  - timestamp:       datetime index
  - consumption_mw:  потребление в MW
  - temperature_c:   температура в °C (known forecast)
  - is_holiday:      bool
  - is_weekend:      bool
"""

from __future__ import annotations

from pathlib import Path

import holidays
import numpy as np
import pandas as pd

from gridforecast.config import (
    BASE_LOAD_MW,
    DAILY_AMPLITUDE_MW,
    DRIFT_TYPES,
    EVENING_PEAK_HOURS,
    GRADUAL_DRIFT_END_DAY,
    GRADUAL_DRIFT_MAGNITUDE,
    GRADUAL_DRIFT_START_DAY,
    HOLIDAY_REDUCTION,
    HOURS_PER_DAY,
    MORNING_PEAK_HOURS,
    N_DAYS_TOTAL,
    NOISE_STD_MW,
    RANDOM_SEED,
    RECURRING_PERIOD_DAYS,
    SEASONAL_AMPLITUDE,
    SUDDEN_DRIFT_DAY,
    SUDDEN_DRIFT_MAGNITUDE,
    SUDDEN_DRIFT_RECOVERY_DAYS,
    SUMMER_PEAK_MONTHS,
    SYNTHETIC_DIR,
    TEMP_COMFORT_HIGH,
    TEMP_COMFORT_LOW,
    TEMP_COMFORT_MID,
    WEEKEND_DAYS,
    WEEKEND_REDUCTION,
    WINTER_PEAK_MONTHS,
)


# ============================================================
# 🌡️  TEMPERATURE
# ============================================================

def generate_temperature(
    timestamps: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Реалистична температура:
      - годишна синусоида (лято ~28°C, зима ~2°C)
      - дневна вариация (±5°C, пик следобед)
      - шум
    """
    day_of_year = timestamps.dayofyear.values
    hour = timestamps.hour.values

    # Годишна: min в средата на януари, max в средата на юли
    yearly = 15.0 + 13.0 * np.sin(2 * np.pi * (day_of_year - 105) / 365.25)

    # Дневна: пик в 15:00, min в 4:00
    daily = 5.0 * np.sin(2 * np.pi * (hour - 9) / 24)

    # Шум
    noise = rng.normal(0, 2.0, size=len(timestamps))

    return yearly + daily + noise


# ============================================================
# 🔥 TEMPERATURE EFFECT ON CONSUMPTION (U-shape)
# ============================================================

def temperature_effect(temp_c: np.ndarray) -> np.ndarray:
    """
    U-shaped ефект:
      - T < 10°C  → отопление → +потребление
      - T > 25°C  → климатици → +потребление
      - 10–25°C   → комфорт → min потребление

    Връща множител ~[1.0, 1.4].
    """
    effect = np.ones_like(temp_c, dtype=float)

    # Отопление (под comfort_low)
    heating = np.maximum(0, TEMP_COMFORT_LOW - temp_c)
    effect += 0.03 * heating  # +3% на всеки градус под 10°C

    # Охлаждане (над comfort_high)
    cooling = np.maximum(0, temp_c - TEMP_COMFORT_HIGH)
    effect += 0.04 * cooling  # +4% на всеки градус над 25°C

    return effect


# ============================================================
# 📅 CALENDAR FEATURES
# ============================================================

def make_calendar_features(
    timestamps: pd.DatetimeIndex,
    country: str = "BG",
) -> pd.DataFrame:
    """Календарни features + holidays."""
    df = pd.DataFrame(index=timestamps)
    df["hour"] = timestamps.hour
    df["day_of_week"] = timestamps.dayofweek
    df["day_of_year"] = timestamps.dayofyear
    df["month"] = timestamps.month
    df["is_weekend"] = df["day_of_week"].isin(WEEKEND_DAYS)

    # Holidays (България) — поправено
    years = sorted(set(timestamps.year))
    bg_holidays = holidays.country_holidays(country, years=years)

    # ⚠️ Fix: конвертираме Timestamps към date обекти преди isin()
    dates_only = pd.DatetimeIndex(timestamps).normalize().date
    df["is_holiday"] = [d in bg_holidays for d in dates_only]

    return df

# ============================================================
# ⚡ DAILY PATTERN (double peak)
# ============================================================

def daily_pattern(hour: np.ndarray) -> np.ndarray:
    """
    Двоен пик: сутрин (7-9) + вечер (18-21), нощен минимум.
    Връща стойности в [-1, +1] (нормализирано).
    """
    pattern = np.zeros_like(hour, dtype=float)

    for h in MORNING_PEAK_HOURS:
        pattern += np.exp(-0.5 * ((hour - h) / 1.2) ** 2)

    for h in EVENING_PEAK_HOURS:
        pattern += np.exp(-0.5 * ((hour - h) / 1.5) ** 2)

    # Нормализирай до [-1, +1]
    pattern = pattern / pattern.max() if pattern.max() > 0 else pattern
    return pattern


# ============================================================
# 🌞 YEARLY PATTERN (summer + winter peaks)
# ============================================================

def yearly_pattern(day_of_year: np.ndarray) -> np.ndarray:
    """
    Годишна вариация: лято (климатици) + зима (отопление).
    Връща множител ~[1 - A, 1 + A].
    """
    # Два пика: януари (15) и юли (196)
    winter = np.exp(-0.5 * ((day_of_year - 15) / 40) ** 2)
    summer = np.exp(-0.5 * ((day_of_year - 196) / 40) ** 2)
    pattern = winter + summer  # ~[0, 2]

    # Нормализирай до [0, 1], после мапни към [1-A, 1+A]
    pattern = pattern / pattern.max() if pattern.max() > 0 else pattern
    return 1.0 + SEASONAL_AMPLITUDE * (pattern - 0.5) * 2


# ============================================================
# 🔄 DRIFT FUNCTIONS
# ============================================================

def drift_factor_stable(day_idx: np.ndarray) -> np.ndarray:
    """Без drift."""
    return np.ones_like(day_idx, dtype=float)


def drift_factor_sudden(day_idx: np.ndarray) -> np.ndarray:
    """
    Sudden drift в SUDDEN_DRIFT_DAY, recovery за SUDDEN_DRIFT_RECOVERY_DAYS.
    Имитира COVID lockdown.
    """
    factor = np.ones_like(day_idx, dtype=float)
    recovery_end = SUDDEN_DRIFT_DAY + SUDDEN_DRIFT_RECOVERY_DAYS

    # Lockdown период
    mask_drift = (day_idx >= SUDDEN_DRIFT_DAY) & (day_idx < recovery_end)
    # Recovery (линейно обратно)
    mask_recovery = day_idx >= recovery_end

    factor[mask_drift] = 1.0 + SUDDEN_DRIFT_MAGNITUDE
    factor[mask_recovery] = 1.0

    # Плавен преход в recovery
    n_recovery = (day_idx >= SUDDEN_DRIFT_DAY) & (day_idx < recovery_end)
    progress = (day_idx[n_recovery] - SUDDEN_DRIFT_DAY) / SUDDEN_DRIFT_RECOVERY_DAYS
    factor[n_recovery] = (1.0 + SUDDEN_DRIFT_MAGNITUDE) + (
        -SUDDEN_DRIFT_MAGNITUDE
    ) * progress

    return factor


def drift_factor_gradual(day_idx: np.ndarray) -> np.ndarray:
    """
    Постепенна промяна от GRADUAL_DRIFT_START_DAY до GRADUAL_DRIFT_END_DAY.
    Имитира EV adoption.
    """
    factor = np.ones_like(day_idx, dtype=float)

    mask = day_idx >= GRADUAL_DRIFT_START_DAY
    span = GRADUAL_DRIFT_END_DAY - GRADUAL_DRIFT_START_DAY
    progress = np.clip(
        (day_idx[mask] - GRADUAL_DRIFT_START_DAY) / span, 0.0, 1.0
    )
    factor[mask] = 1.0 + GRADUAL_DRIFT_MAGNITUDE * progress

    return factor


def drift_factor_recurring(day_idx: np.ndarray) -> np.ndarray:
    """
    Recurring (сезонен) drift — не 'истински', а повтарящ се pattern.
    Тук: плавна годишна промяна (не drift, а естествена вариация).
    """
    return 1.0 + 0.05 * np.sin(2 * np.pi * day_idx / RECURRING_PERIOD_DAYS)


DRIFT_FUNCTIONS = {
    "stable": drift_factor_stable,
    "sudden": drift_factor_sudden,
    "gradual": drift_factor_gradual,
    "recurring": drift_factor_recurring,
}


# ============================================================
# ⚡ MAIN GENERATOR
# ============================================================

def generate_consumption(
    drift_type: str = "stable",
    n_days: int = N_DAYS_TOTAL,
    start_date: str = "2023-01-01",
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Генерира hourly DataFrame с потребление + drift scenario.

    Parameters
    ----------
    drift_type : str
        "stable" | "sudden" | "gradual" | "recurring"
    n_days : int
        Колко дни да генерираме
    start_date : str
        Начална дата (ISO)
    seed : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Индексиран по timestamp, с колони:
        consumption_mw, temperature_c, is_holiday, is_weekend,
        hour, day_of_week, day_of_year, month, drift_factor
    """
    if drift_type not in DRIFT_FUNCTIONS:
        raise ValueError(
            f"Unknown drift_type: {drift_type}. "
            f"Choose from {list(DRIFT_FUNCTIONS.keys())}"
        )

    rng = np.random.default_rng(seed)

    # 1. Timestamps (hourly)
    timestamps = pd.date_range(
        start=start_date,
        periods=n_days * HOURS_PER_DAY,
        freq="h",
        tz=None,
    )

    # 2. Calendar
    cal = make_calendar_features(timestamps)
    hour = cal["hour"].values
    day_idx = (timestamps - timestamps[0]).days.values

    # 3. Temperature
    temp = generate_temperature(timestamps, rng)

    # 4. Base load + patterns
    daily = daily_pattern(hour)
    yearly = yearly_pattern(cal["day_of_year"].values)
    temp_eff = temperature_effect(temp)

    # 5. Calendar effects
    weekend_mask = cal["is_weekend"].values
    holiday_mask = cal["is_holiday"].values

    weekday_factor = np.where(weekend_mask, WEEKEND_REDUCTION, 1.0)
    holiday_factor = np.where(holiday_mask, HOLIDAY_REDUCTION, 1.0)

    # 6. Drift
    drift_fn = DRIFT_FUNCTIONS[drift_type]
    drift = drift_fn(day_idx)

    # 7. Compose
    consumption = (
        BASE_LOAD_MW
        + DAILY_AMPLITUDE_MW * daily
    ) * yearly * temp_eff * weekday_factor * holiday_factor * drift

    # 8. Noise
    consumption += rng.normal(0, NOISE_STD_MW, size=len(timestamps))

    # 9. Negatives → clip (физически невъзможно)
    consumption = np.maximum(consumption, 0.0)

    # 10. Build DataFrame
    df = pd.DataFrame(
        {
            "consumption_mw": consumption,
            "temperature_c": temp,
            "is_holiday": cal["is_holiday"].values,
            "is_weekend": cal["is_weekend"].values,
            "hour": hour,
            "day_of_week": cal["day_of_week"].values,
            "day_of_year": cal["day_of_year"].values,
            "month": cal["month"].values,
            "drift_factor": drift,
        },
        index=timestamps,
    )
    df.index.name = "timestamp"

    return df


# ============================================================
# 💾 SAVE / LOAD
# ============================================================

def generate_all_datasets(
    save: bool = True,
    output_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Генерира всичките 4 drift datasets и (опционално) ги записва като parquet.
    """
    if output_dir is None:
        output_dir = SYNTHETIC_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets: dict[str, pd.DataFrame] = {}

    for drift_type in DRIFT_TYPES:
        print(f"⚡ Генерирам: {drift_type}...", end=" ")
        df = generate_consumption(drift_type=drift_type)
        datasets[drift_type] = df

        if save:
            path = output_dir / f"consumption_{drift_type}.parquet"
            df.to_parquet(path, compression="snappy")
            print(f"→ {path.name} ({len(df):,} rows)")
        else:
            print(f"({len(df):,} rows)")

    return datasets


# ============================================================
# 🧪 CLI
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("⚡ GridForecast — Data Generator")
    print("=" * 60)
    datasets = generate_all_datasets(save=True)
    print("=" * 60)
    print(f"✅ Готови: {len(datasets)} datasets")
    for name, df in datasets.items():
        print(
            f"   {name:10s} | "
            f"{len(df):>7,} rows | "
            f"mean={df['consumption_mw'].mean():>8.1f} MW | "
            f"std={df['consumption_mw'].std():>7.1f} MW"
        )
    print("=" * 60)