"""Implement the class RealData, which computes travel times from PeMS data."""
from importlib import resources
import os

import pandas as pd
import geopandas as gpd
import numpy as np


class RealData:
    """Elaborate PeMS data."""

    def __init__(self, route=101, way="N"):
        """Compute the travel times for each day.

        Args:
            route: the number of the highway the data will be taken
                from. It has to be one of 101, 85, 880, 87, 17,
                280, 237 and 680.

            way: the direction in which the traffic is considered.
                It has to be one of "N", "S" for the highways that go
                north to south, or one of "E", "W" for the highways
                that go west to east.
        """
        if way not in ["N", "S", "E", "W"]:
            raise ValueError(f"{way} is not a valid road direction")

        filename = resources.files("vickrey.data").joinpath(
            "cache_travel_times", f"speeds_{route}_{way}.csv"
        )

        # Check if the speeds for the route/way have already been
        # generated
        if os.path.isfile(filename):
            tts = pd.read_csv(filename, index_col=0, parse_dates=True)
            if tts.shape[1] != 1:
                raise ValueError(
                    "Found wrong csv file. Please regenerate cash"
                )
            self.travel_times = pd.to_timedelta(tts.iloc[:, 0])
        else:
            self.travel_times = self._generate_speeds(route, way, filename)

    def _generate_speeds(self, route, way, filename):
        """Compute the travel times at each available time point.

        Args:
            route, way: Data of the road to elaborate.
            filename: Name of the file to save

        Returns:
            travel_times: A pandas series containing the travel times,
                which has been saved in a file with the given filename.

        """
        # Import speed data
        speeds = pd.read_hdf(
            resources.files("vickrey.data").joinpath("pems-bay.h5")
        )
        # Convert mph to kph
        speeds *= 1.609344

        # Following code (commented) shows that 0s are wrong
        # measurements. They can thus be deleted.

        # speedz = speeds.loc[:, (speeds == 0).max()]
        # days = speedz[(speedz == 0).max(axis=1)].index.day_of_year
        # day0 = speedz.loc[:, (speedz[speedz.index.day_of_year == days[2]] == 0).max()].columns
        # plt.plot(speedz.loc[speedz.index.day_of_year == days[2], day0[5:10]])
        # plt.show()

        speeds = speeds.replace(0, np.nan).ffill()
        speeds = speeds.reindex(
            pd.date_range(speeds.index[0], speeds.index[-1], freq="5min"),
            method="ffill",
        )

        # Import station metadata
        st_data = pd.read_csv(
            resources.files("vickrey.data").joinpath(
                "stations_meta/d04_text_meta_2018_01_26.txt"
            ),
            sep="\t",
        ).set_index("ID")
        f_st_data = st_data.loc[
            speeds.columns
        ]  # Filtering the sensors for which data are available
        if route not in f_st_data.Fwy.unique():
            raise ValueError(f"{route} is not a valid highway name")

        g_data = gpd.GeoDataFrame(
            f_st_data,
            geometry=gpd.points_from_xy(
                f_st_data.Longitude, f_st_data.Latitude
            ),
            crs="EPSG:4326",
        )

        speeds_route = speeds[
            g_data[g_data.Fwy == route].sort_values("Longitude").index
        ]

        speeds_route_way = speeds_route.loc[
            :, g_data.loc[speeds_route.columns].Dir == way
        ]

        if speeds_route_way.empty:
            raise ValueError(f"Highway {route} does not go in direction {way}")

        # Since sensors are pretty close, it is fine to use euclidean
        # distance. If computing the actual travel distance was required, the
        # package OSMnx could do that.

        g_route_way = gpd.GeoDataFrame(
            g_data.loc[speeds_route_way.columns].geometry
        )

        # For going north or west, the order has to be inverted
        if way in "NW":
            g_route_way = g_route_way.iloc[::-1]
        g_route_way.to_crs(epsg=3310, inplace=True)
        g_route_way["pos_n"] = g_route_way.geometry.shift(-1)
        g_route_way["distance"] = g_route_way.geometry.distance(
            g_route_way.pos_n
        ).fillna(0)

        # cur_time is initialized to the initial time of each travel time
        # point, and will then be increased iteratively
        cur_time = speeds_route_way.index.to_series(name="cur")
        for i in g_route_way.index:
            # For finding the most likely travel time available, speeds for
            # the nearest datapoint (in time) are found by doing a "nearest"
            # merge (merge_asof). To do this, it is necessary to sort the
            # values with respect to the key (namely, the current time and not
            # the departure time)
            cur_time.sort_values(inplace=True)
            speeds_approx = pd.merge_asof(
                cur_time,
                speeds_route_way,
                left_on="cur",
                right_index=True,
                direction="nearest",
            ).set_index("cur")

            # The index will be the starting point for the computed speeds
            speeds_approx.index.name = "start"

            # Time taken to go trough a segment is computed by dividing its
            # length (in km) by the speed found by the loop sensor
            time_taken = (
                g_route_way.loc[i, "distance"] / 1000
            ) / speeds_approx.loc[cur_time, i]

            # Finally, current time is updated by increasing it by the
            # time taken
            cur_time = (
                cur_time + pd.to_timedelta(time_taken, "h").values
            ).rename("cur")

        # The final series is now sorted by its index
        arr_time = cur_time.sort_index()
        travel_times = arr_time - arr_time.index

        travel_times.to_csv(filename)
        return travel_times

    def tt_for_day(self, day):
        """Find the travel times for a given day.

        Args:
            day: Day for which the travel times are extracted

        Returns:
            tt_of_day: series containing the travel time for the given
                day
        """
        tt_minutes = self.travel_times.dt.seconds / 60
        times = tt_minutes[tt_minutes.index.day_of_year == day].index
        tt_of_day = tt_minutes[times]
        tt_of_day.index = (tt_of_day.index - tt_of_day.index[0]).seconds / 60
        tt_of_day.name = "{}, {} {} ({})".format(
            times[0].day_name(), times[0].month_name(), times[0].day, day
        )
        return tt_of_day
