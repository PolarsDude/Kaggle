import polars as pl
import polars.selectors as cs
import polars_ds as ds
import polars_xdt as xdt
from datetime import date


def transform_to_datetime(df):
    # Transform time columns to datetime
    time_columns = [c for c in df.columns if c.startswith("time")]

    df = df.with_columns(pl.col(*time_columns).str.to_datetime().name.keep())

    # Rename time columns
    datetime_columns = [c.replace("time", "datetime") for c in time_columns]

    # Create dictionary for renaming
    rename_time = {key: value for key, value in zip(time_columns, datetime_columns)}

    # Rename columns
    return df.rename(rename_time)


# Consecutive time differences
def diff_time(df: pl.DataFrame) -> pl.DataFrame:
    datetime_columns = [c for c in df.columns if c.startswith("datetime")]
    diff_exprs = [
        pl.col(datetime_columns[i + 1])
        .sub(pl.col(datetime_columns[i]))
        .dt.total_seconds()
        .alias(f"diff_time_{i + 1}_{i + 2}")
        for i in range(len(datetime_columns) - 1)
    ]
    return df.with_columns(*diff_exprs)


# Total duration between datetime1 and datetime10
def total_duration(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("datetime10")
        .sub(pl.col("datetime1"))
        .dt.total_seconds()
        .alias("total_duration")
    )


# Number of unique sites
def num_sites(df: pl.DataFrame) -> pl.DataFrame:
    # Create a list of all sites such that the number of unique sites can be calculated
    site_columns = [c for c in df.columns if c.startswith("site")]

    df = df.with_columns(pl.concat_list(*site_columns).alias("all_sites"))

    # Remove nulls
    df = df.with_columns(pl.col("all_sites").list.drop_nulls())

    # Calculate the number of unique sites
    df = df.with_columns(pl.col("all_sites").list.n_unique().alias("num_sites"))
    # Calculate the number of unique sites
    return df.with_columns(pl.col("all_sites").list.n_unique().alias("num_sites"))


# Time of day per site visit
def time_of_day(df: pl.DataFrame) -> pl.DataFrame:
    time_bins = [
        ("night", (0, 5)),
        ("morning", (6, 11)),
        ("afternoon", (12, 17)),
        ("evening", (18, 23)),
    ]
    expressions = [
        (
            pl.col(c)
            .dt.hour()
            .is_between(start, end, closed="both")
            .cast(pl.Int8)
            .alias(f"time_of_day_{c}_{label}")
        )
        for c in df.columns
        if c.startswith("datetime")
        for label, (start, end) in time_bins
    ]
    return df.with_columns(*expressions)


# Create booleans of months
def months(df: pl.DataFrame) -> pl.DataFrame:
    month_names = [
        ("january", 1),
        ("february", 2),
        ("march", 3),
        ("april", 4),
        ("may", 5),
        ("june", 6),
        ("july", 7),
        ("august", 8),
        ("september", 9),
        ("october", 10),
        ("november", 11),
        ("december", 12),
    ]
    expressions = [
        (
            (pl.col("datetime1").dt.month() == month_num)
            .cast(pl.Int8)
            .alias(f"month_{month_name}")
        )
        for month_name, month_num in month_names
    ]
    return df.with_columns(*expressions)


# Create booleans of weeks
def weeks(df: pl.DataFrame):
    # All weeks
    weeks = list(range(1, 53))

    # Create expressions for each week based on datetime1
    expressions = [
        (
            (pl.col("datetime1").dt.week() == week_num)
            .cast(pl.Int8)
            .alias(f"week_{week_num}")
        )
        for week_num in weeks
    ]

    return df.with_columns(*expressions)


# Categorize days
def days(df: pl.DataFrame):
    days = list(range(7))  # 0 til 6

    expressions_days = [
        (
            (pl.col("datetime1").dt.weekday() == day_num)
            .cast(pl.Int8)
            .alias(f"weekday_{day_num}")
        )
        for day_num in days
    ]
    return df.with_columns(*expressions_days)


def is_workday(df: pl.DataFrame):
    return df.with_columns(
        xdt.is_workday("datetime1").cast(pl.Int64).alias("is_workday")
    )
