import numpy as np
import pandas as pd
import os
from parse_imu_sbet import read_applanix_pospac_sbet
from parse_imu_gps import parse_imu_gps
from scipy import interpolate
from datetime import datetime
from datetime import timedelta

def string_to_seconds(string):

    date_format = '%Y/%m/%d %H:%M:%S.%f'
    date =  datetime.strptime(string, date_format)
    date_origin = datetime.strptime(f"{date.year}/01/01 00:00:00.0", date_format)

    seconds = timedelta(days=(date-date_origin).days,
                        hours= date.hour,
                        minutes= date.minute,
                        seconds= date.second,
                        microseconds=date.microsecond).total_seconds()

    return seconds

def seconds_to_string(seconds, year):
    if np.isnan(seconds):
        return np.nan
    date_origin = datetime(year, 1, 1)
    date = date_origin + timedelta(seconds=seconds)
    date_string = date.strftime('%Y/%m/%d %H:%M:%S.%f')
    return date_string

def main(applanix_sbet_file, imu_gps_file):
    """
    the point of all this is that because we have frameindex (timestamps field) in the imu_gps file
    we need to interpolate the applanix to that resolution to be able to use it correctly.
    """

    # reading the applanix file
    applanix_df = read_applanix_pospac_sbet(applanix_sbet_file, 2021, 7, 23)

    # read the imu_gps
    nano_df = parse_imu_gps(imu_gps_file)

    # now interpolating the nano_df to the same resolution as applanix_df based on time.
    year = int(nano_df["Gps_UTC_Date&Time"][0][:4])
    def interp(key):
        if key == "Gps_UTC_Date&Time":
            seconds = nano_df[key].map(string_to_seconds)
            y_new = np.interp(applanix_df["timestamp"].to_numpy(),
                      nano_df["timestamp_seconds"].to_numpy(),
                      seconds.to_numpy(),
                      left = np.nan,
                      right = np.nan)
            y_new = [seconds_to_string(i, year) for i in y_new]

        else:
            y_new = np.interp(applanix_df["timestamp"].to_numpy(),
                              nano_df["timestamp_seconds"].to_numpy(),
                              nano_df[key].to_numpy(),
                              left=np.nan,
                              right=np.nan)
        return y_new

    nano_interp_df = pd.DataFrame(None, columns = nano_df.keys())
    for key in list(nano_df.keys())[:-1]:
        nano_interp_df[key] = interp(key)
    nano_interp_df["timestamp_seconds"] = applanix_df["timestamp"]
    nano_interp_df["Status"][:] = 1

    # dropping rows that contain nan
    nano_interp_df.dropna(0)

    # limit the number of decimals
    nano_interp_df = nano_interp_df.round(decimals=12)

    nano_interp_df.to_csv(imu_gps_file.replace(".txt","_slinear.txt"), index=False, sep = "\t")

    print(f'Wrote file to {imu_gps_file.replace(".txt","_slinear.txt")}')


if __name__ == "__main__":
    main("/Volumes/Work/Projects/NURI/DATA/labsphere/IMUdata/applanix/sbet_1133.out",
           "/Volumes/Work/Projects/NURI/DATA/labsphere/IMUdata/nano/imu_gps.txt")



