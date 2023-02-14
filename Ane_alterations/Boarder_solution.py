''' Classes for Improvements upon existing framework, 2023->'''

import numpy as np
import math
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import NamedTuple, List, Union

class MapBoarderRegulator:      #Class with set up for Stopping simulation/plotting outside of map

    # Defines map boarders, route and start points
    def __init__(self, size, center, route):
        self.north_side = center[1] + size[1]
        self.south_side = center[1] - size[1]
        self.east_side = center[0] + size[0]
        self.west_side = center[0] - size[0]
        self.start_north = start_north
        self.start_east = start_east
        self.route = route


    # From models.py
    def load_waypoints(self, route):
        '''
        Reads the file containing the route and stores it as an
        array of north positions and an array of east positions
        '''
        self.data = np.loadtxt(route)
        self.north = []
        self.east = []
        for i in range(0, (int(np.size(self.data) / 2))):
            self.north.append(self.data[i][0])
            self.east.append(self.data[i][1])

    # Finds start point simulation
    def search_waypoints(self):

        # First route point in map
        if self.north_side >= self.north[0] <= self.south_side and self.east_side >= self.east[0] <= self.west_side:
            start_north = self.north[0]
            start_east = self.east[0]
            return start_north, start_east

        # First route point not in map
        else:
            point_pair_north = []
            point_pair_east = []

            # If point i outside of map and point i+1 inside map they are added to point pair list
            for i in range(len(self.north)):
                if self.north[i] not in range(self.south_side, self.north_side) or self.east[i] not in range(self.west_side, self.east_side) \
                and self.north[i+1] in range(self.south_side, self.north_side) and self.east[i+1] in range(self.west_side, self.east_side):
                    point_pair_north.extend([self.north[i], self.north[i+1]])
                    point_pair_east.extend([self.east[i], self.east[i+1]])
                    return

            # x and y coordinates
            x = [point_pair_east[0], point_pair_east[1]]
            y = [point_pair_north[0], point_pair_north[1]]

            # Calculate the coefficients a (slope, sometimes m is used) and b (y-interception)
            coefficients = np.polyfit(x, y, 1)
            a = coefficients[0]
            b = coefficients[1]

            # Compute the values of the line, y = ax + b
            line_eq = np.poly1d(coefficients)

            # List of possible starting points
            possible_start_points = []

            # Checking for intersections with boarder
            if line_eq == self.north_side: # y value equals north boarder and x value at intersection
                x_n = (self.north_side - b)/a
                possible_start_points.extend([x_n, self.north_side])

            elif line_eq == self.south_side: # y value equals south boarder and x value at intersection
                x_s = (self.south_side - b)/a
                possible_start_points.extend([x_s, self.south_side])

            elif (line_eq - b)/a == self.west_side: # x value equals west boarder and y value at intersection
                y_w = a*self.west_side + b
                possible_start_points.extend([self.west_side, y_w])

            elif (line_eq - b)/a == self.east_side: # x value equals west boarder and y value at intersection
                y_e = a * self.east_side + b
                possible_start_points.extend([self.east_side, y_e])

            # witch one is right is who ever is closer to point_pair[0]
            least_distance = 2000000
            dist = []
            for i in range(len(possible_start_points),2):
                distance = (possible_start_points[i] - x[i])**2 + (possible_start_points[i+1] - y[i])**2
                dist.append(distance)
                if distance <= least_distance:
                    least_distance = distance
            for i in range(len(dist)):
                if dist[i] == least_distance:
                    start_north = possible_start_points[i+1]
                    start_east = possible_start_points[i]
                    return start_north, start_east

