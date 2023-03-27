'''Collision avoidance from Set-based Line-of-sight (LOS) path following with collision avoidance
   for underactuated unmanned surface vessel, Moe & Pettersen (2016)

   Uses a set-based control approach for switching between path following mode and collision avoidance mode.

   Any modes can be used in combination with this, here the ship_in_transit_simulator has the path following mode and
   the collision avoidance mode is the one suggested in the article.
'''
import math
from typing import NamedTuple
import numpy as np




class InitialStates(NamedTuple):
    '''Desired initial velocity and heading'''
    init_desired_yaw_angle__psi_des: float
    init_desired_surge_velocity__u_des: float




class StaticValues(NamedTuple):
    '''Values that don't change during the simulation'''
    safe_radius__r0: float
    switch_radius__rm: float
    head_on_angle__alpha: float
    look_ahead_distance__delta: float
    mass_matrix__m: np.ndarray
    dampening_matrix__d: np.ndarray
    time_step: float
    max_rudder_angle: float
    max_shaft_speed: float
    surge_velocity_obstacle__u0: float
    collision_avoidance_speed: float




class UpdatedVariables(NamedTuple):
    '''Variables take from existing ship and obstacle that needs to be continuously updated'''
    north__y: float
    east__x: float
    yaw_angle__psi: float
    surge_speed__u: float
    sway_speed__v: float
    yaw_rate__r: float
    shaft_speed: float
    north_obstacle__yc: float
    east_obstacle__xc: float
    yaw_obstacle__psi_c: float



class Controllers(NamedTuple):
    surge: float
    yaw: float




class SetBasedGuidance:
    '''Set based guidance algorithm that chooses between path following and obstacle avoidance.
       Currently adjusted to accommodate stationary obstacles'''

    def __init__(self,initial: InitialStates, static: StaticValues):
        # Initialization for set based method
        self.path_following = 'path following'
        self.object_avoidance = 'object avoidance'
        self.last_mode = self.path_following
        self.lambda_d = -1
        self.delta = static.look_ahead_distance__delta
        self.r0 = static.safe_radius__r0
        self.rm = static.switch_radius__rm
        self.alpha = static.head_on_angle__alpha
        self.u0 = static.surge_velocity_obstacle__u0   # math.sqrt(self.xc_dot**2 + self.yc_dot**2)
        self.u_des_oa = static.collision_avoidance_speed

        # Necessary values for controllers
        self.time_step = static.time_step
        self.max_rudder_angle = static.max_rudder_angle
        self.max_shaft_speed = static.max_shaft_speed

        # Matrices
        self.m = static.mass_matrix__m
        self.d = static.dampening_matrix__d

        # Desired velocity and heading
        self.psi_des = initial.init_desired_yaw_angle__psi_des
        self.u_des = initial.init_desired_surge_velocity__u_des

        # Static values for path following
        self.init_u_des = initial.init_desired_surge_velocity__u_des
        self.init_psi_des = initial.init_desired_yaw_angle__psi_des



    def angle_correction(self, angle):
        '''Converts angle to an angle in [-pi, pi]'''
        nr_rounds = abs(angle) // 2*np.pi

        # If angles is over a full circle
        if angle > 2*np.pi:
            angle = angle - nr_rounds*2*np.pi

        elif angle < -2*np.pi:
            angle = angle + nr_rounds*2*np.pi

        # If angle is over half a circle (simulator does not handle this well)
        if angle > np.pi:
            angle = - (2*np.pi - angle)

        elif angle < -np.pi:
            angle = 2*np.pi + angle
        return angle

    def atan2_to_sim(self, angle):
        '''Converts angle you get from math.atan2 (+-pi from horizontal axis right side) to coordinate system according
        to ShipSimplifiedPropulsion (relative to north axis with positive clockwise). Corrects angles over pi'''
        angle = angle * (-1) + (np.pi / 2)

        # If angle is over half a circle (simulator does not handle this well)
        if angle > np.pi:
            angle = - (2 * np.pi - angle)

        elif angle < -np.pi:
            angle = 2 * np.pi + angle

        return angle


    def update_variables(self, updated: UpdatedVariables, cont: Controllers):

        # Updated states
        self.y = updated.north__y
        self.x = updated.east__x
        self.psi = updated.yaw_angle__psi
        self.u = updated.surge_speed__u
        self.v = updated.sway_speed__v
        self.r = updated.yaw_rate__r
        self.shaft_speed = updated.shaft_speed
        self.yc = updated.north_obstacle__yc
        self.xc = updated.east_obstacle__xc
        self.psi_c = updated.yaw_obstacle__psi_c


        # Controllers
        self.tau_u = cont.surge
        self.tau_r = cont.yaw

        # Vessel model fucntions
        under = (self.m[1][1] * self.m[2][2] - (self.m[1][2]) ** 2)
        self.fu_vt = (((self.m[1][1] * self.v) + (self.m[1][2] * self.r)) / self.m[0][0]) * self.r
        self.f_uvr = ((self.m[1][2] * self.d[1][1] - self.m[1][1] * (self.d[2][1] +\
                     (self.m[1][1] - self.m[0][0]) * self.u)) / under) * self.v + \
                     ((self.m[1][2] * (self.d[1][2] + self.m[0][0] * self.u) -\
                       self.m[1][1] * (self.d[2][2] + self.m[1][2] * self.u)) / under) * self.r
        self.xu = (((self.m[1][2]) ** 2 - self.m[0][0] * self.m[2][2]) / under) * self.u + \
                  ((self.d[2][2] * self.m[1][2] - self.d[1][2] * self.m[2][2]) / under)
        self.yu = ((self.m[1][1] * self.m[1][2] - self.m[0][0] * self.m[1][2]) / under) * self.u - \
                  ((self.d[1][1] * self.m[2][2] - self.d[2][1] * self.m[1][2]) / under)

        # Derived own ship states
        self.y_dot = math.sin(self.psi) * self.u + math.cos(self.psi) * self.v
        self.x_dot = math.cos(self.psi) * self.u - math.sin(self.psi) * self.v
        self.psi_dot = self.r
        self.u_dot = self.fu_vt - (self.d[0][0] / self.m[0][0]) * self.u + self.tau_u
        self.v_dot = self.xu * self.r + self.yu * self.v
        self.r_dot = self.f_uvr + self.tau_r


        # Derived obstacle states
        self.yc_dot = self.u0 * math.cos(self.psi_c)  # Y component of obstacle surge speed, zero for static obstacle
        self.xc_dot = self.u0 * math.sin(self.psi_c)  # X component of obstacle surge speed, zero for static obstacle

        # Calculation variables
        self.phi = math.atan2((self.y - self.yc),(self.x - self.xc))
        #self.phi = self.atan2_to_sim(self.phi)
        #self.phi = self.angle_correction(self.phi)

        self.vita_0 = 0  # set as zero for stationary object: math.atan2(self.yc_dot,self.xc_dot)
        # self.vita_0 = self.atan2_to_sim(self.vita_0)
        # self.vita_0 = self.angle_correction(self.vita_0)
        self.v0 = self.u0 * math.cos(self.psi - self.vita_0) # Zero when object is stationary

        self.rho = math.sqrt((self.x - self.xc) ** 2 + (self.y - self.yc) ** 2)
        self.sigma = self.rho
        self.sigma_min = min(self.rm, max(self.sigma, self.r0))
        self.sigma_max = math.inf
        self.sigma_dot = (2 * (self.x - self.xc) * (self.x_dot - self.xc_dot) + 2 * (self.y - self.yc) * (self.y_dot - self.yc_dot)) / \
                         (2 * math.sqrt((self.x - self.xc) ** 2 + (self.y - self.yc) ** 2))

        print(self.sigma_min, self.sigma, self.sigma_dot)

        self.omega = (np.pi / 2 + self.phi) - self.psi_c
        self.omega = self.atan2_to_sim(self.omega)
        self.omega = self.angle_correction(self.omega)



        self.e = self.r0 - self.rho
        # self.a = math.sqrt(self.u_des_oa**2 + self.v**2) - self.v0**2
        # self.b = -2*self.v0**2*self.e
        # self.c = -self.v0**2*(self.delta**2 + self.e**2)
        # if self.v0 >= 0:
        #     self.k = (-self.b + math.sqrt(self.b**2 - 4*self.a*self.c))/2*self.a
        # elif self.v0 < 0:
        #     self.k = (-self.b - math.sqrt(self.b**2 - 4*self.a*self.c))/2*self.a
        self.k = 0  # set to zero for stationary object




    def obstacle_avoidance_algorithm(self, lambda_d):

        self.psi_oa = self.phi + lambda_d * ((math.pi / 2) - math.atan((self.e + self.k)/self.delta)) - math.atan(self.v / self.u_des_oa)
        self.psi_oa = self.atan2_to_sim(self.psi_oa) # Comment out this for previous result
        self.psi_oa = self.angle_correction(self.psi_oa)
        #print(self.psi_oa)
        return self.psi_oa






    def tangent_cone(self):
        '''Decides when to switch between path following and object avoidance
           based of distance between objects and whether they are moving closer or further away
        '''
        if self.sigma_min < self.sigma < self.sigma_max:
            return True
        elif self.sigma <= self.sigma_min and self.sigma_dot >= 0:
            return True
        elif self.sigma <= self.sigma_min and self.sigma_dot < 0:
            return False
        elif self.sigma >= self.sigma_max and self.sigma_dot <= 0:
            return True
        else:
            return False





    def lambda_direction(self):
        '''Decides direction to turn when making a collision avoidance maneuver'''

        psi_oa_c = self.obstacle_avoidance_algorithm(lambda_d=1)
        psi_oa_cc = self.obstacle_avoidance_algorithm(lambda_d=-1)

        # Overtaking
        if self.omega < -112.5*np.pi/180 or self.omega > 112.5*np.pi/180:
            if abs(self.psi - psi_oa_cc) <= abs(self.psi - psi_oa_c):
                self.lambda_d = -1
                return self.lambda_d
            elif abs(self.psi - psi_oa_cc) > abs(self.psi - psi_oa_c):
                self.lambda_d = 1
                return self.lambda_d

        # Crossing from left
        elif self.alpha <= self.omega < 112.5*np.pi/180:
            self.lambda_d = -1
            return self.lambda_d

        # Crossing from right
        elif -112.5*np.pi/180 <= self.omega < self.alpha:
            self.lambda_d = -1
            return self.lambda_d

        # Head on
        elif -self.alpha <= self.omega < self.alpha:
            self.lambda_d = -1
            return self.lambda_d






    def set_based_guidance_algorithm(self):
        ''' While decides based on tangent cone which algorithm to use'''
        while self.tangent_cone() == True or self.tangent_cone() == False:
            a = self.tangent_cone()
            #print(a)
            if a:
                mode = self.path_following
                self.u_des = self.init_u_des
                self.psi_des = self.init_psi_des
                #print('pf_a')
            else:
                if self.last_mode == self.path_following:
                    self.lambda_direction()
                    #print(self.lambda_d)

                mode = self.object_avoidance
                self.psi_des = self.obstacle_avoidance_algorithm(self.lambda_d)
                self.u_des = self.u_des_oa
                #print(self.tau_u, self.tau_r)
                #print(self.u_des)
                #print('oa')
            self.last_mode = mode
            #print(self.u_des, self.psi_des)
            return self.u_des, self.psi_des












