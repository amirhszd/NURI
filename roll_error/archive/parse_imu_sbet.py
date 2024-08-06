import math
import datetime
import numpy
import struct
import pandas as pd

def get_number_of_days_to_closest_sunday(date):
   date_origin = datetime.datetime(date.year, 1, 1)
   dow = date_origin.strftime("%A") # day of week

   if dow == "Sunday":
      dif = 0
   elif dow == "Monday":
      dif = 1
   elif dow == "Tuesday":
      dif = 2
   elif dow == "Wednesday":
      dif = 3
   elif dow == "Thursday":
      dif = 4
   elif dow == "Friday":
      dif = 5
   else:  # Assuming dow == "Saturday"
      dif = 6

   return (date.timetuple().tm_yday - 1) - dif




def read_applanix_pospac_sbet(filename, year, month, day):
   """
      Applanix PosPac SBET file description found at:
         http://vislab-ccom.unh.edu/~schwehr/rt/21-python-binary-files.html


      the file reads timestamp[idx], day of the week. so friday would be the fifth day.
      for the application I am doing, there should be a change in the timeformat because
      the current one is two days behind, i am commenting this section.
      axhcis
   """
   sbetFile = open(filename, 'rb')
   sbetData = sbetFile.read()
   numberBytes = len(sbetData)
   recordLength = 17 * 8
   numberRecords = numberBytes // recordLength

   date = datetime.datetime(year, month, day)
   days_from_sunday = get_number_of_days_to_closest_sunday(date)
   offsetSeconds = \
      datetime.timedelta(days=days_from_sunday).total_seconds()


   # weekNumber = datetime.datetime(year, month, day).isocalendar()[1]
   # # Determine the weekday (where Monday is 1 and Sunday is 7) to allow
   # # for the offset from the start of the year to be the same as that found
   # # using the typical GPS week (where Sunday is 1 and Saturday is 7)
   # weekDay = datetime.datetime(year, month, day).isocalendar()[2]
   # offsetSeconds = \
   #    datetime.timedelta(days=(weekNumber+(weekDay//7)-1)*7).total_seconds()
   timestamp = numpy.zeros(numberRecords, dtype=numpy.float64)
   latitude = numpy.zeros(numberRecords, dtype=numpy.float64)
   longitude = numpy.zeros(numberRecords, dtype=numpy.float64)
   altitude = numpy.zeros(numberRecords, dtype=numpy.float64)
   xVelocity = numpy.zeros(numberRecords, dtype=numpy.float64)
   yVelocity = numpy.zeros(numberRecords, dtype=numpy.float64)
   zVelocity = numpy.zeros(numberRecords, dtype=numpy.float64)
   roll = numpy.zeros(numberRecords, dtype=numpy.float64)
   pitch = numpy.zeros(numberRecords, dtype=numpy.float64)
   yaw = numpy.zeros(numberRecords, dtype=numpy.float64)
   wanderAngle = numpy.zeros(numberRecords, dtype=numpy.float64)
   xAcceleration = numpy.zeros(numberRecords, dtype=numpy.float64)
   yAcceleration = numpy.zeros(numberRecords, dtype=numpy.float64)
   zAcceleration = numpy.zeros(numberRecords, dtype=numpy.float64)
   xAngularRate = numpy.zeros(numberRecords, dtype=numpy.float64)
   yAngularRate = numpy.zeros(numberRecords, dtype=numpy.float64)
   zAngularRate = numpy.zeros(numberRecords, dtype=numpy.float64)

   for idx in range(numberRecords):
      recordStartByte = idx * recordLength
      # Time stamp is GPS time of week [seconds]
      timestamp[idx], \
      latitude[idx], \
      longitude[idx], \
      altitude[idx], \
      xVelocity[idx], \
      yVelocity[idx], \
      zVelocity[idx], \
      roll[idx], \
      pitch[idx], \
      yaw[idx], \
      wanderAngle[idx], \
      xAcceleration[idx], \
      yAcceleration[idx], \
      zAcceleration[idx], \
      xAngularRate[idx], \
      yAngularRate[idx], \
      zAngularRate[idx] = \
         struct.unpack('ddddddddddddddddd', \
                       sbetData[recordStartByte:recordStartByte+recordLength])

      # so the time stamp is the timestamp in seconds from the start of the week in terms of seconds
      # we add the timestamp from the start of the week to this.
      timestamp[idx] += offsetSeconds   # Convert to GPS time of year [seconds]
      latitude[idx] = math.degrees(latitude[idx])
      longitude[idx] = math.degrees(longitude[idx])
      roll[idx] = math.degrees(roll[idx])
      pitch[idx] = math.degrees(pitch[idx])
      yaw[idx] = math.degrees(yaw[idx])
   sbetFile.close()
   imu_gps = {'timestamp': timestamp,
              'latitude': latitude,
              'longitude': longitude,
              'altitude': altitude,
              'roll': roll,
              'pitch': pitch,
              'yaw': yaw}

   df = pd.DataFrame(imu_gps)

   return df


if __name__ == "__main__":
   filename = "/Volumes/Work/Projects/NURI/DATA/labsphere/IMUdata/applanix/sbet_1133.out"
   imu_gps = read_applanix_pospac_sbet(filename,
                             2021, 7, 23)

   print("ok")