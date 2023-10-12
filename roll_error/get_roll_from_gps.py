import numpy as np
import os
import pandas as pd


def get_roll_data_vs_time():
    data = np.loadtxt("/Volumes/Work/Projects/NURI/DATA/labsphere/IMUdata/nano/imu_gps.txt", skiprows=1, dtype = str)

    # first column is the roll and the
    df = pd.DataFrame(data, columns = ["Roll","Pitch","Yaw","Lat","Lon","Alt","Timestamp", "Gps_UTC_Date","Time", "Status", "Heading"])
    x = df["Timestamp"].to_numpy()[186339:190537]
    y = df["Roll"].to_numpy()[186339:190537]

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 1, figsize=(8, 5))
    plt.subplots_adjust(hspace=0)
    axes.scatter(x, y, color="k")
    plt.tight_layout(h_pad=0)
    plt.show()


def detrend_signal_polynomial():
    from obspy.signal.detrend import polynomial
    data = np.loadtxt("/Volumes/Work/Projects/NURI/DATA/labsphere/IMUdata/nano/FromWaterfall.csv", delimiter=",")
    fit = np.polyval(np.polyfit(data[:,0], data[:,1], deg=3), data[:,0])

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(8, 5))
    plt.subplots_adjust(hspace=0)
    axes[0].plot(data[:,0], data[:,1], color="k", label="Original Data")
    axes[0].plot(data[:,0], fit, color="red", lw=2, label="Fitted Trend")
    axes[0].legend(loc="best")
    axes[0].label_outer()
    axes[0].set_yticks(axes[0].get_yticks()[1:])

    axes[1].plot(data[:,1] - fit, color="k", label="Result")
    axes[1].legend(loc="best")
    axes[1].label_outer()
    axes[1].set_yticks(axes[1].get_yticks()[:-1])
    axes[1].set_xlabel("Samples")

    plt.tight_layout(h_pad=0)
    plt.show()

    detrended_data = np.array([data[:,0],data[:,1] - fit])

    return detrended_data


def interpolate_reverse_order():

    from_waterfall = "/Volumes/Work/Projects/NURI/DATA/labsphere/IMUdata/nano/FromWaterfall_0_tile.csv"
    data_from_waterfall = np.loadtxt(from_waterfall, skiprows=1, dtype = float, delimiter= ",")

    roll_from_gps = "/Volumes/Work/Projects/NURI/DATA/labsphere/IMUdata/nano/clipped_gps_for_0_tile.csv"
    data_roll_from_gps = np.loadtxt(roll_from_gps, skiprows=1, dtype=float, delimiter= ",")

    time_gps = data_roll_from_gps[:,0]
    time_waterfall_fake = np.linspace(time_gps[0], time_gps[-1], len(data_from_waterfall))

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 1, figsize=(8, 5))
    plt.subplots_adjust(hspace=0)
    axes.plot(time_waterfall_fake, data_from_waterfall[:,-1][::-1], color="k", label="From Waterfall Image")
    axes.plot(time_gps, data_roll_from_gps[:,-1]*5 +7.5, color="r", label="From Drone Roll (Omega)")
    axes.legend(loc="best")
    axes.label_outer()
    axes.set_yticks(axes.get_yticks()[1:])
    axes.set_xlabel("Timestamp")
    axes.set_ylabel("Roll | Along Track Line-to-Line Pixel Shift")
    plt.show()








if __name__ == "__main__":
    detrend_data = detrend_signal_polynomial("/Volumes/Work/Projects/NURI/DATA/labsphere/IMUdata/nano/FromWaterfall.csv")

