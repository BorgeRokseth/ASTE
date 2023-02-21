# Classes for Improvements upon existing framework, 2023->

import numpy as np


class MapBoarderRegulator:
    ''' Class with set up for Stopping simulation/plotting outside of map '''

    def __init__(self, shipping_lane, center, size):
        ''' Defines map boarders, and sets route and empty start points '''
        self.north_side = center[1] + size[1]
        self.south_side = center[1] - size[1]
        self.east_side = center[0] + size[0]
        self.west_side = center[0] - size[0]
        self.route = shipping_lane


    def load_waypoints(self): # From models.py, changed so route not redundant
        ''' Reads the file containing the route and stores it as an
        array of north positions and an array of east positions '''
        self.data = np.loadtxt(self.route)
        self.north = []
        self.east = []
        for i in range(0, (int(np.size(self.data) / 2))):
            self.north.append(self.data[i][0])
            self.east.append(self.data[i][1])
        #print(self.north, self.east)

    def intersection_on_y(self, y_value, line_slope, y_intercept):
        self.x_value = (y_value - y_intercept) / line_slope
        return self.x_value

    def intersection_on_x(self, x_value, line_slope, y_intercept):
        self.y_value = line_slope * x_value + y_intercept
        return self.y_value

    # Finds start point simulation
    def search_waypoints(self):
        ''' Searches through route points and finds if they are contained in the map. If not, finds intersection with map boarder and sets this as start point'''

        self.load_waypoints()  # Uncomment when running it properly

        # First route point in map
        if self.north[0] <= self.north_side and self.north[0] >= self.south_side and self.east[0]<= self.east_side and self.east[0] >= self.west_side:
            self.start_north = self.north[0]
            self.start_east = self.east[0]
            #print(self.start_north, self.start_east)
        # First route point not in map
        else:
            point_pair_north = []
            point_pair_east = []

            # If point i outside of map and point i+1 inside map they are added to point pair list
            for i in range(len(self.north)-1):
                if self.north[i] >= self.north_side or self.north[i] <= self.south_side or self.east[i] >= self.east_side or self.east[i] <= self.west_side \
                and self.north[i+1] <= self.north_side and self.north[i+1] >= self.south_side and self.east[i+1]<= self.east_side and self.east[i+1] >= self.west_side:
                    point_pair_north.extend([self.north[i], self.north[i+1]])
                    point_pair_east.extend([self.east[i], self.east[i+1]])
            #print(point_pair_north, point_pair_east)

            # x and y coordinates for calculation of line coefficients
            x = [point_pair_east[0], point_pair_east[1]]
            y = [point_pair_north[0], point_pair_north[1]]

            # Calculate the coefficients a (slope, sometimes m is used) and b (y-interception)
            coefficients = np.polyfit(x, y, 1)
            a = coefficients[0]
            b = coefficients[1]

            # Compute the values of the line, y = ax + b
            line_eq = np.poly1d(coefficients)
            #print(line_eq)
            # List of possible starting points
            possible_start_points = []


            # Checking for intersections with boarder
            y_list = [self.north_side, self.south_side]
            x_list = [self.east_side, self.west_side]

            for i in range(len(y_list)):
                self.intersection_on_y(y_list[i], line_slope=a, y_intercept=b)
                if  self.x_value <= self.east_side and self.x_value >= self.west_side:
                    possible_start_points.extend([ y_list[i],self.x_value])

            for i in range(len(x_list)):
                self.intersection_on_x(x_list[i], line_slope=a, y_intercept=b)
                if self.y_value <= self.north_side and self.y_value >= self.south_side:
                    possible_start_points.extend([ self.y_value,x_list[i]])
            #print(possible_start_points)

            # witch one is right is who ever is closer to point_pair[0]
            least_distance = 2000000
            dist = []
            for i in range(0,len(possible_start_points),2):
                distance = (possible_start_points[i+1] - x[0])**2 + (possible_start_points[i] - y[0])**2
                dist.append(distance)
                if distance <= least_distance:
                    least_distance = distance
            for i in range(len(dist)):
                if dist[i] == least_distance:
                    self.start_north = possible_start_points[i]
                    self.start_east = possible_start_points[i+1]
            #print(self.start_north, self.start_east)

#test=MapBoarderRegulator(shipping_lane='test_route.txt', center=[253536, 7045845], size=[19000, 15000])
#test.load_waypoints()
#test.search_waypoints()