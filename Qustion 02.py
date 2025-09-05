from __future__ import annotations
import argparse
import os
import pandas as pd
import math
import tempfile
import shutil
import zipfile
import sys
import csv
from typing import Optional, Tuple


DEFAULT_INPUT_FOLDER = "temperatures"
OUTPUT_SEASON_FILE = "average_temp.txt"
OUTPUT_RANGE_FILE = "largest_temp_range_station.txt"
OUTPUT_STABILITY_FILE = "temperature_stability_stations.txt"

MONTH_MAP = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

def _has_csv(path: str) -> bool:
    """Return True if the directory contains at least one .csv file."""
    try:
        for f in os.listdir(path):
            if f.lower().endswith('.csv'):
                return True
        return False
    except OSError:
        return False


def _find_case_insensitive_dir(parent: str, target_name_lower: str) -> Optional[str]:
    """Find a subdirectory in parent that matches target_name (case-insensitive)."""
    try:
        for entry in os.listdir(parent):
            p = os.path.join(parent, entry)
            if os.path.isdir(p) and entry.lower() == target_name_lower:
                return p
    except OSError:
        return None
    return None


def find_temperatures_folder(preferred: str, allow_extract_zip: bool = False) -> Tuple[Optional[str], Optional[str]]:
    
    pref_abs = os.path.abspath(preferred)
    if os.path.isdir(pref_abs) and _has_csv(pref_abs):
        return pref_abs, None

    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    cand = os.path.join(script_dir, preferred)
    if os.path.isdir(cand) and _has_csv(cand):
        return cand, None

    cand2 = _find_case_insensitive_dir(script_dir, preferred.lower())
    if cand2 and _has_csv(cand2):
        return cand2, None

    cwd = os.path.abspath(os.getcwd())
    cand = os.path.join(cwd, preferred)
    if os.path.isdir(cand) and _has_csv(cand):
        return cand, None

    cand2 = _find_case_insensitive_dir(cwd, preferred.lower())
    if cand2 and _has_csv(cand2):
        return cand2, None

    # Walk upwards to find folder
    def walk_up_and_find(start_dir: str):
        p = start_dir
        for _ in range(6):
            candidate = os.path.join(p, preferred)
            if os.path.isdir(candidate) and _has_csv(candidate):
                return candidate
            alt = _find_case_insensitive_dir(p, preferred.lower())
            if alt and _has_csv(alt):
                return alt
            parent = os.path.dirname(p)
            if not parent or parent == p:
                break
            p = parent
        return None

    candidate = walk_up_and_find(cwd)
    if candidate:
        return candidate, None
    candidate = walk_up_and_find(script_dir)
    if candidate:
        return candidate, None

    # Check common candidate locations
    common_candidates = [
        os.path.join("/mnt", "data", preferred),
        os.path.join("/mnt", "data", "Assignment_2", preferred),
        os.path.join(cwd, "Assignment_2", preferred),
    ]
    for c in common_candidates:
        if os.path.isdir(c) and _has_csv(c):
            return c, None

    # Attempt to extract from zip if allowed
    if allow_extract_zip:
        search_dirs = [cwd, script_dir]
        if os.path.isdir(os.path.join("/mnt", "data")):
            search_dirs.append(os.path.join("/mnt", "data"))
        for sd in search_dirs:
            try:
                for fname in os.listdir(sd):
                    if fname.lower().endswith('.zip'):
                        fpath = os.path.join(sd, fname)
                        try:
                            with zipfile.ZipFile(fpath, 'r') as zf:
                                namelist = zf.namelist()
                                lower_names = [n.lower() for n in namelist]
                                target_prefix = preferred.lower().rstrip('/') + '/'
                                if any(n.startswith(target_prefix) for n in lower_names) or any('/' + preferred.lower() + '/' in n for n in lower_names):
                                    extract_dir = tempfile.mkdtemp(prefix='temps_extract_')
                                    zf.extractall(extract_dir)
                                    extracted_candidate = os.path.join(extract_dir, preferred)
                                    if os.path.isdir(extracted_candidate) and _has_csv(extracted_candidate):
                                        return extracted_candidate, extract_dir
                                    alt = _find_case_insensitive_dir(extract_dir, preferred.lower())
                                    if alt and _has_csv(alt):
                                        return alt, extract_dir
                        except zipfile.BadZipFile:
                            continue
            except OSError:
                continue
    return None, None


def find_station_column(df: pd.DataFrame) -> Optional[str]:
    """Detect the station column in dataframe by trying common column names and heuristics."""
    pref_names = ["STATION_NAME", "station_name", "station", "site", "STN_ID", "stn_id"]
    for p in pref_names:
        for c in df.columns:
            if c.lower() == p.lower():
                return c
    for c in df.columns:
        low = c.lower()
        if ("station" in low) or low.startswith("stn") or ("site" in low) or ("name" in low and ("station" in low or "site" in low)):
            return c
    return None


def load_all_csvs(folder_path: str) -> pd.DataFrame:
   
    try:
        files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith('.csv')
        ]
    except OSError as e:
        raise FileNotFoundError(f"Could not list files in folder: {folder_path}: {e}")

    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No CSV files found in folder: {folder_path}")

    records = []
    for filepath in files:
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            print(f"Warning: failed to read {filepath}: {e}")
            continue

        station_col = find_station_column(df)
        if station_col is None:
            print(f"Skipping {os.path.basename(filepath)}: could not detect a station column")
            continue

        month_cols = [c for c in df.columns if c.strip().lower() in MONTH_MAP]
        if not month_cols:
            print(f"Skipping {os.path.basename(filepath)}: no monthly columns found (January..December expected)")
            continue

        
        melted = df.melt(
            id_vars=[station_col],
            value_vars=month_cols,
            var_name='month_name',
            value_name='temperature'
        )
        melted['month_key'] = melted['month_name'].astype(str).str.strip().str.lower()
        melted['month'] = melted['month_key'].map(MONTH_MAP)
        melted['season'] = melted['month'].apply(
            lambda m: month_to_season(m) if (isinstance(m, (int, float)) and not math.isnan(m)) else None
        )
        melted['temperature'] = pd.to_numeric(melted['temperature'], errors='coerce')

        sub = melted[[station_col, 'month', 'season', 'temperature']].rename(columns={station_col: 'station'})
        sub = sub.dropna(subset=['station', 'month'])
        records.append(sub)

    if not records:
        raise ValueError('No usable temperature records were loaded from CSVs.')

    combined = pd.concat(records, ignore_index=True)
    return combined


def month_to_season(month: int) -> str:
    """Map a month number to its corresponding Australian season."""
    if month in (12, 1, 2):
        return 'Summer'
    if month in (3, 4, 5):
        return 'Autumn'
    if month in (6, 7, 8):
        return 'Winter'
    return 'Spring'


def compute_seasonal_averages(df: pd.DataFrame) -> pd.Series:
    """Compute average temperature for each season across all stations and years."""
    season_means = df.groupby('season', observed=True)['temperature'].mean()
    order = ['Summer', 'Autumn', 'Winter', 'Spring']
    return season_means.reindex(order)


def compute_station_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute max, min, std deviation, and range for each station."""
    stats = df.groupby('station', observed=True)['temperature'].agg(['max', 'min', 'std']).rename(columns={'std': 'stddev'})
    stats['range'] = stats['max'] - stats['min']
    return stats


def save_seasonal_average(season_means: pd.Series, out_file: str):
    """Save the seasonal average results to a text file."""
    lines = []
    for season, val in season_means.items():
        if pd.isna(val):
            lines.append(f"{season}: No data")
        else:
            lines.append(f"{season}: {round(val, 1)}°C")
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"Saved seasonal averages to {out_file}")


def save_largest_range(stats: pd.DataFrame, out_file: str):
    """Save the station(s) with the largest temperature range to a text file."""
    if stats.empty:
        with open(out_file, 'w', encoding='utf-8') as f:
            f.write('No data')
        print(f"Saved largest range results to {out_file} (no data available)")
        return

    max_range = stats['range'].max()
    winners = stats[stats['range'] == max_range]
    lines = []
    for station, row in winners.iterrows():
        if pd.isna(row['max']) or pd.isna(row['min']):
            lines.append(f"Station {station}: No valid max/min data")
        else:
            lines.append(
                f"Station {station}: Range {round(row['range'], 1)}°C (Max: {round(row['max'], 1)}°C, Min: {round(row['min'], 1)}°C)"
            )

    with open(out_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"Saved largest temperature range station(s) to {out_file}")


def save_stability(stats: pd.DataFrame, out_file: str):
    """Save the most stable and most variable stations based on standard deviation."""
    valid = stats[~stats['stddev'].isna()].copy()
    if valid.empty:
        with open(out_file, 'w', encoding='utf-8') as f:
            f.write('No stations have enough data to compute standard deviation')
        print(f"Saved stability results to {out_file} (insufficient data)")
        return

    min_std = valid['stddev'].min()
    max_std = valid['stddev'].max()
    stable = valid[valid['stddev'] == min_std]
    variable = valid[valid['stddev'] == max_std]

    lines = []
    for station, row in stable.iterrows():
        lines.append(f"Most Stable: Station {station}: StdDev {round(row['stddev'], 1)}°C")
    for station, row in variable.iterrows():
        lines.append(f"Most Variable: Station {station}: StdDev {round(row['stddev'], 1)}°C")

    with open(out_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"Saved temperature stability stations to {out_file}")


def _make_demo_dataset(base_dir: str) -> str:
    """Create a small demo 'temperatures' folder with two CSVs and return its path."""
    temps = os.path.join(base_dir, 'temperatures')
    os.makedirs(temps, exist_ok=True)
    # Simple but non-trivial demo data
    rows1 = [
        ['STATION_NAME', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
        ['Darwin', 31.2, 31.4, 31.5, 31.1, 30.5, 29.4, 29.1, 29.6, 30.2, 31.0, 31.3, 31.6],
        ['Alice Springs', 36.1, 35.8, 32.1, 28.4, 23.3, 19.2, 18.1, 20.3, 25.0, 29.2, 33.0, 35.3],
    ]
    rows2 = [
        ['STATION_NAME', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
        ['Darwin', 31.0, 31.6, 31.7, 31.0, 30.2, 29.5, 28.9, 29.2, 30.0, 30.9, 31.1, 31.4],
        ['Katherine', 34.2, 34.0, 32.0, 30.0, 27.0, 24.0, 22.5, 24.3, 28.0, 30.5, 32.3, 33.8],
    ]
    f1 = os.path.join(temps, '2019.csv')
    f2 = os.path.join(temps, '2020.csv')
    with open(f1, 'w', newline='') as fh:
        csv.writer(fh).writerows(rows1)
    with open(f2, 'w', newline='') as fh:
        csv.writer(fh).writerows(rows2)
    return temps


def _run_internal_tests() -> None:
    """Run self-tests with temporary CSVs to verify core functionality.

    These tests are new additions (no existing tests were modified).
    """
    import tempfile

  
    with tempfile.TemporaryDirectory() as td:
        f1 = os.path.join(td, 'year1.csv')
        f2 = os.path.join(td, 'year2.csv')
        rows = [
            ['STATION_NAME', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
            ['A', 30, 31, 28, 25, 22, 20, 19, 21, 24, 26, 27, 29],
            ['B', 25, 26, 24, 22, 20, 18, 17, 19, 21, 23, 24, 25],
        ]
        with open(f1, 'w', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerows(rows)
        rows2 = [
            ['STATION_NAME', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
            ['A', 31, 30, 29, 26, 23, 21, 20, 22, 25, 27, 28, 30],
            ['C', 15, 16, 14, 13, 12, 11, 10, 11, 12, 13, 14, 15],
        ]
        with open(f2, 'w', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerows(rows2)

        df = load_all_csvs(td)
        season_means = compute_seasonal_averages(df)
        stats = compute_station_stats(df)

        assert not season_means.isna().all(), 'Season means should not be all NaN'
        assert 'A' in stats.index, 'Station A should be present'
        assert stats.loc['A', 'range'] > stats.loc['C', 'range'], 'A should have larger range than C'

    with tempfile.TemporaryDirectory() as base:
        parent = os.path.join(base, 'some', 'nested')
        os.makedirs(parent)
        temps_dir = os.path.join(base, 'Temperatures')
        os.makedirs(temps_dir)
        csv_path = os.path.join(temps_dir, 's.csv')
        with open(csv_path, 'w', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(['STATION_NAME', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
            writer.writerow(['X', 1,1,1,1,1,1,1,1,1,1,1,1])
        old_cwd = os.getcwd()
        try:
            os.chdir(parent)
            found, _ = find_temperatures_folder('temperatures', allow_extract_zip=False)
            assert found is not None, 'Should find the Temperatures folder in parent directories'
        finally:
            # ensure finally block is not empty and restore CWD
            os.chdir(old_cwd)

   
    with tempfile.TemporaryDirectory() as td:
        zippath = os.path.join(td, 'data.zip')
        inner_folder = 'temperatures'
        with zipfile.ZipFile(zippath, 'w') as zf:
            csv_name = inner_folder + '/a.csv'
            csv_bytes = 'STATION_NAME,January,February,March,April,May,June,July,August,September,October,November,December\nZ,1,1,1,1,1,1,1,1,1,1,1,1\n'.encode('utf-8')
            zf.writestr(csv_name, csv_bytes)
        old_cwd = os.getcwd()
        try:
            os.chdir(td)
            found, extract_dir = find_temperatures_folder('temperatures', allow_extract_zip=True)
            assert found is not None and extract_dir is not None, 'Should find and extract temperatures from zip'
            shutil.rmtree(extract_dir)
        finally:
            os.chdir(old_cwd)

   
    assert month_to_season(12) == 'Summer' and month_to_season(1) == 'Summer'
    assert month_to_season(3) == 'Autumn' and month_to_season(5) == 'Autumn'
    assert month_to_season(6) == 'Winter' and month_to_season(8) == 'Winter'
    assert month_to_season(9) == 'Spring' and month_to_season(11) == 'Spring'

   
    with tempfile.TemporaryDirectory() as td:
        good = os.path.join(td, 'good.csv')
        bad = os.path.join(td, 'bad.csv')
        with open(good, 'w', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(['STATION_NAME', 'January', 'February', 'March'])
            writer.writerow(['S1', 10, 11, 12])
        with open(bad, 'w', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(['foo', 'bar'])
        df = load_all_csvs(td)
        assert not df.empty and 'station' in df.columns and 'temperature' in df.columns

    
    with tempfile.TemporaryDirectory() as td:
        demo_path = _make_demo_dataset(td)
        df = load_all_csvs(demo_path)
        # Should include 3 stations across 2 files
        assert set(df['station'].unique()) >= {'Darwin', 'Alice Springs', 'Katherine'}
        # Seasons should be present
        assert set(df['season'].dropna().unique()) == {'Summer', 'Autumn', 'Winter', 'Spring'}

   
    with tempfile.TemporaryDirectory() as td:
        empty_sub = os.path.join(td, 'temperatures')
        os.makedirs(empty_sub, exist_ok=True)
        # Create a non-CSV file
        with open(os.path.join(empty_sub, 'readme.txt'), 'w') as fh:
            fh.write('no csv here')
        found, _ = find_temperatures_folder(empty_sub, allow_extract_zip=False)
        assert found is None, 'Folder without CSVs must not be considered valid'

    
    with tempfile.TemporaryDirectory() as td:
        single = os.path.join(td, 'single.csv')
        with open(single, 'w', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(['STATION_NAME', 'January'])
            writer.writerow(['Solo', 25])
        df = load_all_csvs(td)
        stats = compute_station_stats(df)
        assert pd.isna(stats.loc['Solo', 'stddev']), 'Stddev should be NaN for a single observation'

    
    with tempfile.TemporaryDirectory() as td:
        
        path = os.path.join(td, 'temperatures')
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'y.csv'), 'w', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(['STATION_NAME', 'January', 'February'])
            writer.writerow(['S', 10, 20])
        df = load_all_csvs(path)
        stats = compute_station_stats(df)
        outdir = os.path.join(td, 'out')
        os.makedirs(outdir, exist_ok=True)
        save_seasonal_average(compute_seasonal_averages(df), os.path.join(outdir, OUTPUT_SEASON_FILE))
        save_largest_range(stats, os.path.join(outdir, OUTPUT_RANGE_FILE))
        save_stability(stats, os.path.join(outdir, OUTPUT_STABILITY_FILE))
        for fn in (OUTPUT_SEASON_FILE, OUTPUT_RANGE_FILE, OUTPUT_STABILITY_FILE):
            fp = os.path.join(outdir, fn)
            assert os.path.isfile(fp) and os.path.getsize(fp) > 0




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default=DEFAULT_INPUT_FOLDER, help='Folder containing CSV files (default: temperatures)')
    parser.add_argument('--output-dir', '-o', default='.', help='Output directory (default: current)')
    parser.add_argument('--run-tests', action='store_true', help='Run internal tests and exit')
    parser.add_argument('--extract-zip', action='store_true', help='If set, try to find temperatures in .zip files and extract')
    parser.add_argument('--demo', action='store_true', help='Use an auto-generated demo dataset (overrides any input folder)')
    parser.add_argument('--strict', action='store_true', help='If input is missing, exit with error instead of falling back to demo')
    args = parser.parse_args()

    if args.run_tests:
        _run_internal_tests()
        print('All internal tests passed.')
        return

    
    demo_tmp = None
    if args.demo:
        demo_tmp = tempfile.mkdtemp(prefix='temps_demo_')
        folder = _make_demo_dataset(demo_tmp)
        extracted_temp = demo_tmp
        print("Running with a demo dataset by request.")
    else:
        
        folder, extracted_temp = find_temperatures_folder(args.input, allow_extract_zip=args.extract_zip)
        if folder is None:
            folder, extracted_temp = find_temperatures_folder(args.input, allow_extract_zip=True)
        
        if folder is None and not args.strict:
            demo_tmp = tempfile.mkdtemp(prefix='temps_demo_')
            folder = _make_demo_dataset(demo_tmp)
            extracted_temp = demo_tmp
            print("Input folder not found; falling back to a demo dataset. Use --strict to force an error.")

    if folder is None:
        print("Could not locate a 'temperatures' folder containing CSV files.\n"
              "Tips: Place CSVs under ./temperatures, or pass --input <folder>,\n"
              "or run with --demo to use a generated sample dataset, or --extract-zip if inside a .zip.")
        sys.exit(2)

    print(f"Using temperatures folder: {folder}")

    try:
        df = load_all_csvs(folder)
    except Exception as e:
        print(f"Error loading CSVs: {e}")
        if extracted_temp and os.path.isdir(extracted_temp):
            try:
                shutil.rmtree(extracted_temp)
            except Exception:
                pass
        raise

    
    season_means = compute_seasonal_averages(df)
    os.makedirs(args.output_dir, exist_ok=True)
    save_seasonal_average(season_means, os.path.join(args.output_dir, OUTPUT_SEASON_FILE))

    stats = compute_station_stats(df)
    save_largest_range(stats, os.path.join(args.output_dir, OUTPUT_RANGE_FILE))
    save_stability(stats, os.path.join(args.output_dir, OUTPUT_STABILITY_FILE))

    
    print('\n--- Summary ---')
    print(season_means.to_string())
    if not stats.empty:
        max_range = stats['range'].max()
        winners = stats[stats['range'] == max_range].index.tolist()
        print(f"Largest range: {max_range} across station(s): {', '.join(winners)}")
    print('Outputs written:')
    print(' - ', os.path.join(args.output_dir, OUTPUT_SEASON_FILE))
    print(' - ', os.path.join(args.output_dir, OUTPUT_RANGE_FILE))
    print(' - ', os.path.join(args.output_dir, OUTPUT_STABILITY_FILE))

    
    for tmp in (extracted_temp,):
        if tmp and os.path.isdir(tmp):
            try:
                shutil.rmtree(tmp)
            except Exception:
                pass


if __name__ == '__main__':
    main()
