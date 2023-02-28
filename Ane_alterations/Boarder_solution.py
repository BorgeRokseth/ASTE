# Classes for Improvements upon existing framework, 2023->
import math

import numpy as np


class MapBoarderRegulator:
    ''' Class with set up for Stopping simulation/plotting outside of map '''

    def __init__(self, shipping_lane, center, size):
        ''' Defines map boarders, and sets route and empty start points '''
        self.north_side = center[1] + size[1]/2
        self.south_side = center[1] - size[1]/2
        self.east_side = center[0] + size[0]/2
        self.west_side = center[0] - size[0]/2
        self.route = shipping_lane


    def load_waypoints(self):  # From models.py, changed so route not redundant
        ''' Reads the file containing the route and stores it as an
        array of north positions and an array of east positions '''
        self.data = np.loadtxt(self.route)
        self.north = []
        self.east = []
        for i in range(0, (int(np.size(self.data) / 2))):
            self.north.append(self.data[i][0])
            self.east.append(self.data[i][1])


    def intersection_on_y(self, y_value, line_slope, y_intercept):
        '''Equation for finding x value on line'''
        self.x_value = (y_value - y_intercept) / line_slope
        return self.x_value


    def intersection_on_x(self, x_value, line_slope, y_intercept):
        '''Equation for finding y value on line'''
        self.y_value = line_slope * x_value + y_intercept
        return self.y_value


    def convert_from_tan2code(self):
        '''Converts angle you get from math.atan2 (+-pi from horizontal axis right side) to coordinate system according
        to ShipSimplifiedPropulsion (relative to north axis with positive clockwise). If angle goes over pi it uses
        "opposite" angle ((2pi - angle)*(-1))'''
        ang = self.angle_tan*(-1)+(np.pi/2)
        if ang > np.pi:
            self.angle = (2*np.pi - ang)*(-1)
        else:
            self.angle = ang
        return self.angle


    # Finds start point simulation
    def search_waypoints(self):
        ''' Searches through route points and finds if they are contained in the map.
        If not, finds intersection with map boarder and route, and sets this as start point'''
        self.load_waypoints()


        # First route point inside map
        if self.north_side >= self.north[0] >= self.south_side and self.east_side >= self.east[0] >= self.west_side:
            self.start_north = float(self.north[0])
            self.start_east = float(self.east[0])

            # Counter to separate if and else cases
            self.counter = 0

            # Angle of ship
            delta_y = self.north[1] - self.north[0]
            delta_x = self.east[1] - self.east[0]
            self.angle_tan = math.atan2(delta_y, delta_x)
            self.convert_from_tan2code()


        # First route point not in map
        else:
            point_pair_north = []
            point_pair_east = []

            # If point i outside of map and point i+1 inside map they are added to point pair list
            self.counter = 0
            for i in range(len(self.north) - 1):
                self.counter += 1
                #print(i)
                if (self.north[i] > self.north_side or self.north[i] < self.south_side or self.east[i] > self.east_side or self.east[i] < self.west_side) \
                and self.north_side >= self.north[i + 1] >= self.south_side and self.east_side >= self.east[i + 1] >= self.west_side:
                    point_pair_north.extend([self.north[i], self.north[i + 1]])
                    point_pair_east.extend([self.east[i], self.east[i + 1]])
                    break


            # x and y coordinates for calculation of line coefficients
            x = [point_pair_east[0], point_pair_east[1]]
            y = [point_pair_north[0], point_pair_north[1]]

            # Angle of ship
            delta_y = point_pair_north[1] - point_pair_north[0]
            delta_x = point_pair_east[1] - point_pair_east[0]
            self.angle_tan = math.atan2(delta_y, delta_x)
            self.convert_from_tan2code()

            # Calculate the coefficients a (slope, sometimes m is used) and b (y-interception)
            coefficients = np.polyfit(x, y, 1)
            a = coefficients[0]
            b = coefficients[1]

            # Compute the values of the line, y = ax + b
            line_eq = np.poly1d(coefficients)

            # List of possible starting points
            possible_start_points = []

            # Checking for intersections with boarder
            y_list = [self.north_side, self.south_side]
            x_list = [self.east_side, self.west_side]

            for i in range(len(y_list)):
                self.intersection_on_y(y_list[i], line_slope=a, y_intercept=b)
                if self.x_value <= self.east_side and self.x_value >= self.west_side:
                    possible_start_points.extend([y_list[i], self.x_value])

            for i in range(len(x_list)):
                self.intersection_on_x(x_list[i], line_slope=a, y_intercept=b)
                if self.y_value <= self.north_side and self.y_value >= self.south_side:
                    possible_start_points.extend([self.y_value, x_list[i]])


            # witch one is right is who ever is closer to point_pair[0]
            least_distance = float('inf')  # If set too low it won't get overriden, prompting and attribute error: no attribute 'start_north'
            dist = []
            for i in range(0, len(possible_start_points), 2):
                distance = (possible_start_points[i + 1] - x[0]) ** 2 + (possible_start_points[i] - y[0]) ** 2
                dist.append(distance)
                if distance <= least_distance:
                    least_distance = distance
            for i in range(len(dist)):
                if dist[i] == least_distance:  #if & else statements to make sure we get correct par of points, since possible_start_points = 2*dist
                    if i % 2 == 0:
                        self.start_north = possible_start_points[i]
                        self.start_east = possible_start_points[i + 1]
                    else:
                        self.start_north = float(possible_start_points[i + 1])
                        self.start_east = float(possible_start_points[i + 2])
        return self.start_north, self.start_east

    def update_txt_file(self):
        '''Makes an updated file from input file with correct waypoints (starting at intersection point),
        name: updated_(original filename)'''
        self.search_waypoints()
        self.updated_file = open(f'updated_{self.route}', "w")  # w means txt files is overriden everytime script runs
        if self.counter == 0: # If first waypoint is in map the updated file is the same as original txt file
            for i in range(len(self.north)):
                self.updated_file.writelines([str(self.north[i]),' ', str(self.east[i]), '\n'])
        else:
            self.updated_wp = [str(self.start_north),' ', str(self.start_east), '\n']
            for i in range(self.counter, len(self.north)):
                self.updated_wp.extend([str(self.north[i]),' ', str(self.east[i]), '\n'])
            self.updated_file.writelines(self.updated_wp)
            self.updated_file.close()
        return self.updated_file



#test = MapBoarderRegulator(shipping_lane='own_ship_route.txt', center=[253536, 7045845], size=[19000, 15000])
#test.search_waypoints()
#print(test.counter)
#test.update_txt_file()
#print(test.start_north, test.start_east)
#print(test.angle)

