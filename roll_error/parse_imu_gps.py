import numpy as np
import pandas as pd
import os
from datetime import datetime
from datetime import timedelta

def calculate_time_offset_from(string):

    date_format = '%Y/%m/%d %H:%M:%S.%f'
    date =  datetime.strptime(string, date_format)
    date_origin = datetime.strptime(f"{date.year}/01/01 00:00:00.0", date_format)

    seconds = timedelta(days=(date-date_origin).days,
                        hours= date.hour,
                        minutes= date.minute,
                        seconds= date.second,
                        microseconds=date.microsecond).total_seconds()

    return seconds

def parse_imu_gps(filename):
    df = pd.read_csv(filename, delimiter="\t")

    df["timestamp_seconds"] = df["Gps_UTC_Date&Time"].map(calculate_time_offset_from)
    return df