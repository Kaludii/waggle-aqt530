"""
This module interfaces with the Vaisala AQT530 and parses the output of the
instrument before uploading to Beehive via Waggle.

Updated to support interval-based averaging to reduce database load.
High-frequency raw data is cached locally in CSV files, and only averaged
values are published to Beehive at user-defined intervals.

To display currently available serial ports:
python -m serial.tools.list_ports
"""

import time
import serial
import argparse
import parse
import csv
import logging
import threading
import pandas as pd
import xarray as xr

from pathlib import Path
from datetime import datetime, timezone
from waggle.plugin import Plugin, get_timestamp


def parse_values(sample, **kwargs):
    """
    Parse the AQT530 ASCII data string into a dictionary.

    The AQT ASCII data contains variable names as the second to last
    comma separated value within the line.
    """
    if sample.startswith(b'20'):
        data = parse.search("{ti}," +
                            "{.1F}," +
                            "{.1F}," +
                            "{.1F}," +
                            "{.3F}," +
                            "{.3F}," +
                            "{.3F}," +
                            "{.3F}," +
                            "{.1F}," +
                            "{.1F}," +
                            "{.1F}," +
                            "{w}," +
                            "{d}\r\n" ,
                            sample.decode('utf-8')
                            )
        if data:
            # Parse the variable names from the datastring
            # Captured by the {w} flag
            parms = data['w'].split(':')
            # Convert the variables to floats
            strip = [float(var) for var in data]
            # Create a dictionary to match the parameters and variables
            ndict = dict(zip(parms, strip))
            # Add the AQT datetime to the dictionary
            ndict['datetime'] = data['ti']
            ndict['uptime'] = int(data['d'])
        else:
            ndict = None
    else:
        ndict = None

    return ndict


def secs_to_xr_freq(minutes):
    """Convert minutes to a string frequency for xarray resampling."""
    seconds = int(minutes) * 60
    if seconds <= 0:
        raise ValueError("minutes must be > 0")

    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}min"
    return f"{seconds}s"


def initialize_local_file(site, outdir, publish_names):
    """
    Generate the filename and header info for local CSV file.

    Parameters
    ----------
    site : str
        Site identifier for the deployment location.
    outdir : str
        Directory where to output files.
    publish_names : dict
        Dictionary of publish names and their metadata.

    Returns
    -------
    csv_path : Path
        Path to the initialized CSV file.
    """
    nout = (site +
            '.aqt530.' +
            datetime.now(timezone.utc).strftime("%Y%m%d.%H%M%S") +
            '.csv')
    csv_path = Path(outdir) / nout
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Initializing local CSV file at {csv_path}")
    with open(csv_path, mode='w', newline='', encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header rows matching WXT536 format
        header = ['Timestamp'] + [info[1] for info in publish_names.values()]
        units = ['UTC seconds'] + [info[2] for info in publish_names.values()]
        waggle_vars = ['Timestamp'] + [info[0] for info in publish_names.values()]
        short_names = ['time'] + list(publish_names.keys())

        csv_writer.writerow(header)
        csv_writer.writerow(units)
        csv_writer.writerow(waggle_vars)
        csv_writer.writerow(short_names)

    return csv_path


def publish_file(file_path):
    """Utilizing threading, publish raw CSV file to Beehive."""
    def upload_file(file_path):
        with Plugin() as plugin:
            plugin.upload_file(file_path, timestamp=get_timestamp())
            print(f"Published {file_path}")
    thread = threading.Thread(target=upload_file, args=(file_path,))
    thread.start()
    thread.join()


def publish_avg(args, file_path, publish_names):
    """
    Calculate a user-defined average from the local data files
    and publish to Beehive.

    Reads the accumulated raw CSV data, computes temporal mean for
    all measurement variables, and publishes averaged values.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    file_path : str or Path
        Path to the local CSV file containing raw observations.
    publish_names : dict
        Dictionary mapping short variable names to
        [waggle_name, description, units, sensor_key].
    """
    timestamp = get_timestamp()

    # Define the resampling frequency
    nfreq = secs_to_xr_freq(args.beehive_interval)

    try:
        df = pd.read_csv(file_path, skiprows=3, na_values=-9999)
        df["time"] = pd.to_datetime(
            df["time"], utc=True, errors="coerce"
        ).dt.tz_convert(None)
        df = df.set_index("time").sort_index()

        # Drop non-numeric housekeeping columns before computing mean
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if len(numeric_cols) == 0 or len(df) == 0:
            print(f"No valid data to average in {file_path}")
            return

        ds = xr.Dataset.from_dataframe(df[numeric_cols])
        ds = ds.assign_coords(time=pd.to_datetime(ds["time"].values))

        # Temporal mean for all numeric variables
        ds_mean = ds.resample(time=nfreq).mean()

        # Publish averaged values to Beehive
        with Plugin() as plugin:
            for name, info in publish_names.items():
                waggle_name = info[0]
                description = info[1]
                units = info[2]

                try:
                    value = float(ds_mean[name].round(decimals=3).data[-1])
                except (KeyError, IndexError):
                    continue

                if args.debug:
                    print(f'beehive-avg {timestamp} {waggle_name} '
                          f'{value} {units}')

                logging.info("beehive publishing avg %s %s units %s",
                             waggle_name, value, units)

                plugin.publish(waggle_name,
                               value=value,
                               meta={"units": units,
                                     "sensor": "vaisala-aqt530",
                                     "missing": "-9999.9",
                                     "description": description,
                                     "avg_frequency": nfreq},
                               scope="beehive",
                               timestamp=timestamp
                               )

        print(f"Published averaged data from {file_path}")

    except Exception as e:
        print(f"Error computing/publishing averages: {e}")
    finally:
        try:
            del df, ds, ds_mean
        except Exception:
            pass


def list_files(img_dir):
    """Lists all CSV files within a directory and their sizes in bytes."""
    dir_path = Path(img_dir)
    saved_files = sorted(list(dir_path.glob("*.csv")))
    if saved_files:
        print(f'\nUpdated local files within {dir_path}:')
        for sfile in saved_files:
            file_size = sfile.stat().st_size
            print(f"{sfile}: {file_size} bytes")


def main(args):
    """Main function for AQT530 interface and publishing"""

    # publish_names maps short variable names to:
    # [waggle_name, description, units, sensor_key]
    publish_names = {
        "T":    ["aqt.env.temp",
                 "Ambient Temperature",
                 "degrees Celsius",
                 "T"],
        "P":    ["aqt.env.pressure",
                 "Ambient Atmospheric Pressure",
                 "hPa",
                 "P"],
        "H":    ["aqt.env.humidity",
                 "Ambient Relative Humidity",
                 "percent relative humidity",
                 "H"],
        "NO2":  ["aqt.gas.no2",
                 "Nitrogen Dioxide Gas Concentration",
                 "ppm",
                 "NO2"],
        "CO":   ["aqt.gas.co",
                 "Carbon Monoxide Gas Concentration",
                 "ppm",
                 "CO"],
        "O3":   ["aqt.gas.ozone",
                 "Ozone Gas Concentration",
                 "ppm",
                 "O3"],
        "NO":   ["aqt.gas.no",
                 "Nitric Oxide Gas Concentration",
                 "ppm",
                 "NO"],
        "SO2":  ["aqt.gas.so2",
                 "Sulfur Dioxide Gas Concentration",
                 "ppm",
                 "SO2"],
        "H2S":  ["aqt.gas.h2s",
                 "Hydrogen Sulfide Gas Concentration",
                 "ppm",
                 "H2S"],
        "PM1":  ["aqt.particle.pm1",
                 "Particulate Matter less than 1 micron in diameter",
                 "microgram per cubic meter",
                 "PM1"],
        "PM2.5": ["aqt.particle.pm2.5",
                  "Particulate Matter less than 2.5 microns in diameter",
                  "microgram per cubic meter",
                  "PM2.5"],
        "PM10": ["aqt.particle.pm10",
                 "Particulate Matter less than 10 microns in diameter",
                 "microgram per cubic meter",
                 "PM10"],
    }

    with serial.Serial(args.device,
                       baudrate=args.baud_rate,
                       timeout=args.serial_timeout) as dev:
        try:
            print(f"Serial connection to {args.device} is open")
            last_timestamp = time.gmtime()

            # ---- Local File Initialization ----
            if args.beehive_interval > 0:
                print(f"Averaging mode enabled. Publishing averages every "
                      f"{args.beehive_interval} minutes to Beehive.")
                nfile_writer = initialize_local_file(
                    args.site, args.outdir, publish_names
                )

            if args.debug:
                list_files(args.outdir)
                print("\n")

            # --- Main AQT Interface Loop ----
            while True:

                # --- Check on Publish Interval ----
                if args.beehive_interval > 0:
                    current_timestamp = time.gmtime()

                    if (current_timestamp.tm_min % args.beehive_interval == 0
                            and current_timestamp.tm_min
                            != last_timestamp.tm_min):
                        # Publish averaged data to Beehive
                        publish_avg(args, nfile_writer, publish_names)
                        # Upload the raw CSV file
                        if nfile_writer:
                            print(f"Closing {nfile_writer}")
                            publish_file(nfile_writer)
                        # Initialize a new local file
                        nfile_writer = initialize_local_file(
                            args.site, args.outdir, publish_names
                        )
                        last_timestamp = current_timestamp

                # --- Read AQT Data ----
                # AQT outputs data automatically every ~1 minute
                line = dev.readline()

                if len(line) > 0:
                    timestamp = get_timestamp()
                    logging.debug("Read transmitted data")

                    # Parse the incoming data
                    sample = parse_values(line)

                    if sample:
                        if args.debug:
                            print(f"Parsed AQT sample: {sample}")

                        if args.beehive_interval > 0:
                            # --- Averaging mode: write to local CSV ---
                            with open(nfile_writer, mode='a',
                                      newline='',
                                      encoding="utf-8") as csvfile:
                                csv_writer = csv.writer(csvfile)
                                ts = datetime.now(
                                    timezone.utc
                                ).isoformat(timespec="seconds")
                                out_values = [
                                    str(sample.get(key, '-9999'))
                                    for key in publish_names.keys()
                                ]
                                csv_writer.writerow([ts, *out_values])
                                csvfile.flush()
                        else:
                            # --- Direct publish mode (legacy behavior) ---
                            with Plugin() as plugin:
                                for name, info in publish_names.items():
                                    waggle_name = info[0]
                                    description = info[1]
                                    units = info[2]
                                    sensor_key = info[3]

                                    try:
                                        value = sample[sensor_key]
                                    except KeyError:
                                        continue

                                    if args.debug:
                                        print(
                                            f'beehive {timestamp} '
                                            f'{waggle_name} {value} '
                                            f'{units} {type(value)}'
                                        )

                                    logging.info(
                                        "beehive publishing %s %s "
                                        "units %s type %s",
                                        waggle_name, value, units,
                                        str(type(value))
                                    )

                                    plugin.publish(
                                        waggle_name,
                                        value=value,
                                        meta={
                                            "units": units,
                                            "sensor": "vaisala-aqt530",
                                            "missing": "-9999.9",
                                            "description": description
                                        },
                                        scope="beehive",
                                        timestamp=timestamp
                                    )

        except KeyboardInterrupt:
            print(f"Program interrupted, closing serial port {args.device}")
        finally:
            if dev:
                dev.close()


if __name__ == '__main__':

    plugin_descript = (
        "Script for interfacing with the Vaisala AQT530 datastream."
        " Publishes data to Sage Beehive as immediate or averaged"
        " observations, while providing files of raw observations"
        " at user selected frequency."
    )

    plugin_usage = (
        "python app.py --debug --beehive-publish-interval 10"
    )

    parser = argparse.ArgumentParser(description=plugin_descript,
                                     usage=plugin_usage)

    parser.add_argument("--debug",
                        action="store_true",
                        dest='debug',
                        help="enable debug logs"
                        )
    parser.add_argument("--device",
                        type=str,
                        dest='device',
                        default="/dev/ttyUSB3",
                        help="serial device to use"
                        )
    parser.add_argument("--baudrate",
                        type=int,
                        dest='baud_rate',
                        default=115200,
                        help="baudrate to use"
                        )
    parser.add_argument("--serial-timeout",
                        type=float,
                        dest='serial_timeout',
                        default=65.0,
                        help="Serial read timeout in"
                             " seconds. AQT outputs data every ~60s, so"
                             " timeout should exceed that."
                        )
    parser.add_argument("--beehive-publish-interval",
                        default=10,
                        dest='beehive_interval',
                        type=int,
                        help="Interval in minutes to"
                             " publish averaged data to Beehive. Values > 0"
                             " enable averaging mode. Set to 0 for legacy"
                             " direct-publish behavior."
                        )
    parser.add_argument("--outdir",
                        type=str,
                        dest="outdir",
                        default=".",
                        help="Directory where to output CSV files"
                        )
    parser.add_argument("--site",
                        type=str,
                        default="crocus",
                        dest="site",
                        help="Site identifier for deployment location"
                        )
    args = parser.parse_args()

    main(args)
