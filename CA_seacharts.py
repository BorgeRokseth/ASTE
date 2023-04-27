import math
from typing import NamedTuple
import numpy as np
import os
import shapely.geometry as geo
import seacharts

from ship_in_transit_simulator.models import EnvironmentConfiguration, \
    HeadingByRouteController, HeadingControllerGains, \
    LosParameters, ShipConfiguration, ShipModelSimplifiedPropulsion, \
    SimplifiedPropulsionMachinerySystemConfiguration, SimulationConfiguration, SpecificFuelConsumptionBaudouin6M26Dot3, \
    SpecificFuelConsumptionWartila6L26, MachineryModeParams, MachineryMode, MachineryModes, \
    ThrottleFromSpeedSetPointSimplifiedPropulsion, HeadingByReferenceController


from Ane_alterations.Boarder_solution import MapBoarderRegulator
from Ane_alterations.Collision_avoidance import SetBasedGuidance, PathFollowingParameters, StaticValues, UpdatedVariables, Controllers


class TargetShip(NamedTuple):
    sim_time: float
    start_time: float
    ship_model: ShipModelSimplifiedPropulsion
    speed_controller: ThrottleFromSpeedSetPointSimplifiedPropulsion
    navigation_system: HeadingByRouteController


class TargetShipMaker:
    def __init__(self,
                 ship_configuration: ShipConfiguration,
                 environment: EnvironmentConfiguration,
                 machinery_config: SimplifiedPropulsionMachinerySystemConfiguration,
                 sea_lane: str
                 ) -> None:
        self.ship_config = ship_configuration
        self.env = environment
        self.machinery_config = machinery_config
        self.sea_lane = sea_lane

    def make_target_ship(self, start_time, initial_states: SimulationConfiguration) -> TargetShip:
        target_ship_model = ShipModelSimplifiedPropulsion(
            ship_config=self.ship_config,
            machinery_config=self.machinery_config,
            environment_config=self.env,
            simulation_config=initial_states
        )
        speed_controller = ThrottleFromSpeedSetPointSimplifiedPropulsion(kp=3, ki=0.02,
                                                                         time_step=initial_states.integration_step)
        heading_controller_gains = HeadingControllerGains(kp=7, kd=90, ki=0.01)
        los_guidance_parameters = LosParameters(
            radius_of_acceptance=600,
            lookahead_distance=500,
            integral_gain=0.002,
            integrator_windup_limit=4000
        )
        navigation_system = HeadingByRouteController(route_name=self.sea_lane,
                                                     heading_controller_gains=heading_controller_gains,
                                                     los_parameters=los_guidance_parameters,
                                                     time_step=initial_states.integration_step,
                                                     max_rudder_angle=self.machinery_config.max_rudder_angle_degrees * np.pi / 180)
        return TargetShip(
            sim_time=start_time + initial_states.simulation_time,
            start_time=start_time,
            ship_model=target_ship_model,
            speed_controller=speed_controller,
            navigation_system=navigation_system
        )


if __name__ == "__main__":

    size = 19000, 19000
    center = 253536, 7045845  # east, north UTM-zone 33
    files = ['Trondelag.gdb']
    enc = seacharts.ENC(files=files, border=True, center=center, size=size, new_data=False)

    time_step = 0.5
    # own_ship_route_name = "test_route.txt"      # Redundant, could be added directly into function on line 180
    time_between_snapshots = 60

    main_engine_capacity = 2160e3
    diesel_gen_capacity = 510e3
    hybrid_shaft_gen_as_generator = 'GEN'

    desired_speed_own_ship = 7
    desired_speed_target_ship = 9

    ship_config = ShipConfiguration(
        coefficient_of_deadweight_to_displacement=0.7,
        bunkers=20000,
        ballast=20000,
        length_of_ship=20,
        width_of_ship=8,
        added_mass_coefficient_in_surge=0.4,
        added_mass_coefficient_in_sway=0.4,
        added_mass_coefficient_in_yaw=0.4,
        dead_weight_tonnage=3850000,
        mass_over_linear_friction_coefficient_in_surge=130,
        mass_over_linear_friction_coefficient_in_sway=18,
        mass_over_linear_friction_coefficient_in_yaw=90,
        nonlinear_friction_coefficient__in_surge=2400,
        nonlinear_friction_coefficient__in_sway=4000,
        nonlinear_friction_coefficient__in_yaw=400
    )
    env_config = EnvironmentConfiguration(
        current_velocity_component_from_north=-0.5,
        current_velocity_component_from_east=-0.2,
        wind_speed=5,
        wind_direction=0
    )

    pto_mode_params = MachineryModeParams(
        main_engine_capacity=main_engine_capacity,
        electrical_capacity=0,
        shaft_generator_state=hybrid_shaft_gen_as_generator
    )
    pto_mode = MachineryMode(params=pto_mode_params)

    mso_modes = MachineryModes([pto_mode, pto_mode])

    fuel_curves_me = SpecificFuelConsumptionWartila6L26()
    fuel_curves_dg = SpecificFuelConsumptionBaudouin6M26Dot3()

    machinery_config = SimplifiedPropulsionMachinerySystemConfiguration(
        hotel_load=200e3,
        machinery_modes=mso_modes,
        machinery_operating_mode=1,
        specific_fuel_consumption_coefficients_me=fuel_curves_me.fuel_consumption_coefficients(),
        specific_fuel_consumption_coefficients_dg=fuel_curves_dg.fuel_consumption_coefficients(),
        max_rudder_angle_degrees=30,
        rudder_angle_to_yaw_force_coefficient=500e3,
        rudder_angle_to_sway_force_coefficient=50e3,
        thrust_force_dynamic_time_constant=30
    )

    initial_states_first_ship = SimulationConfiguration(
        initial_north_position_m=7039562.36,
        initial_east_position_m=251040.06,
        initial_yaw_angle_rad=30 * np.pi / 180,
        initial_forward_speed_m_per_s=10,
        initial_sideways_speed_m_per_s=0,
        initial_yaw_rate_rad_per_s=0,
        integration_step=time_step,
        simulation_time=10000,
    )

    initial_states_second_ship = SimulationConfiguration(
        initial_north_position_m=7039562.36,
        initial_east_position_m=251040.06,
        initial_yaw_angle_rad=30 * np.pi / 180,
        initial_forward_speed_m_per_s=7,
        initial_sideways_speed_m_per_s=0,
        initial_yaw_rate_rad_per_s=0,
        integration_step=time_step,
        simulation_time=10000,
    )

    ship_factory = TargetShipMaker(ship_configuration=ship_config,
                                   environment=env_config,
                                   machinery_config=machinery_config,
                                   sea_lane="sea_lane_route.txt"
                                   )

    initial_states_third_ship = SimulationConfiguration(
        initial_north_position_m=7048136.48,
        initial_east_position_m=261421.6,
        initial_yaw_angle_rad=-90 * np.pi / 180,
        initial_forward_speed_m_per_s=9,
        initial_sideways_speed_m_per_s=0,
        initial_yaw_rate_rad_per_s=0,
        integration_step=time_step,
        simulation_time=10000,
    )

    test_target_factory = TargetShipMaker(ship_configuration=ship_config,
                                          environment=env_config,
                                          machinery_config=machinery_config,
                                          sea_lane="try_route.txt"
                                          )

    first_target_ship = ship_factory.make_target_ship(start_time=0, initial_states=initial_states_first_ship)
    second_target_ship = ship_factory.make_target_ship(start_time=200, initial_states=initial_states_second_ship)
    third_target_ship = test_target_factory.make_target_ship(start_time=0, initial_states=initial_states_third_ship)
    list_of_target_ships = [first_target_ship, second_target_ship, third_target_ship]

    test_own = MapBoarderRegulator(
        shipping_lane='own_ship_path_ane.txt',
        center=center,
        size=size
    )
    test_own.search_waypoints()
    test_own.update_txt_file()

    initial_states_own_ship = SimulationConfiguration(
        initial_north_position_m=7048337.46  ,#test_own.start_north,
        initial_east_position_m=248431.66, #test_own.start_east,
        initial_yaw_angle_rad=90*np.pi/180,#test_own.angle,  # in rads, formula: degrees*np.pi/180
        initial_forward_speed_m_per_s=7,
        initial_sideways_speed_m_per_s=0,
        initial_yaw_rate_rad_per_s=0,
        integration_step=time_step,
        simulation_time=2000,
    )

    own_ship = ShipModelSimplifiedPropulsion(
        ship_config=ship_config,
        machinery_config=machinery_config,
        environment_config=env_config,
        simulation_config=initial_states_own_ship
    )

    own_ship_speed_controller = ThrottleFromSpeedSetPointSimplifiedPropulsion(
        kp=3,
        ki=0.02,
        time_step=initial_states_own_ship.integration_step
    )
    own_ship_heading_controller_gains = HeadingControllerGains(
        kp=7,
        kd=90,
        ki=0.01
    )
    own_ship_los_guidance_parameters = LosParameters(
        radius_of_acceptance=600,
        lookahead_distance=500,
        integral_gain=0.002,
        integrator_windup_limit=4000
    )
    own_ship_navigation_system = HeadingByRouteController(
        route_name='test_route.txt',
        heading_controller_gains=own_ship_heading_controller_gains,
        los_parameters=own_ship_los_guidance_parameters,
        time_step=initial_states_own_ship.integration_step,
        max_rudder_angle=machinery_config.max_rudder_angle_degrees * np.pi / 180
    )

    time_since_snapshot = time_between_snapshots
    snap_shot_id = 0
    # Lists for storing ships and trails
    ship_snap_shots = []


    # List of lists for keeping distances between ships
    dist_ships = []
    for i in range(len(list_of_target_ships)):
        dist_ships.append([])

    def collision_index(list_of_lists,own_north, own_east, target_north, target_east):
        ''' Requires an index to keep track of which list to append to, makes lists of distances between ships and
            reconstructed them to an array after loop with np.column_stack.
        '''

        distance = math.sqrt((own_north - target_north) ** 2 + \
                             (own_east - target_east) ** 2)
        list_of_lists[index].append(distance)

        # Collision index
        if distance < 100:
            collision_index =  distance / 100
            print(f'Collision index:{collision_index}')


    # List of lists for keeping track of distance to ground
    dist_ground = []
    folder = 'data\shapefiles'
    for i in range(len(os.listdir(folder))):
        dist_ground.append([])
    # print(os.listdir(folder))

    def grounding_index(list_of_lists, own_north, own_east):
        ''' Makes lists of distances to the nearest ground, lists are reconstructed after loop with np.column_stack'''
        geo_own = geo.Point(own_east, own_north)

        for i in range(len(os.listdir(folder))):
            if i == 0:
                distance = geo_own.distance(enc.land.geometry)
                list_of_lists[i].append(int(distance))


                # if distance < 50:
                #     print('ship grounded into land')

            elif 1 <= i <= 9:
                if i == 1:
                    nr = 0
                elif i == 2:
                    nr = 5
                elif i == 3:
                    nr = 10
                elif i == 4:
                    nr = 20
                if i == 5:
                    nr = 50
                elif i == 6:
                    nr = 100
                elif i == 7:
                    nr = 200
                elif i == 8:
                    nr = 350
                elif i == 9:
                    nr = 500
                distance = geo_own.distance(enc.seabed[nr].geometry)
                list_of_lists[i].append(int(distance))

                # if distance < 50 and (i == 1 or i == 2):
                #     print(f'Ship grounded in sea bed at {nr} meters')

            elif i == 10:
                distance = geo_own.distance(enc.shore.geometry)
                list_of_lists[i].append(int(distance))

                # if distance < 50:
                #     print('Ship grounded into shore')


    # Control system setup for collision avoidance
    heading_controller_gains = HeadingControllerGains(kp=4, kd=90, ki=0.01)
    heading_controller = HeadingByReferenceController(
        gains=heading_controller_gains, time_step=time_step,
        max_rudder_angle=machinery_config.max_rudder_angle_degrees * np.pi / 180
    )


    # Collision avoidance parameters and object third target ship
    path_following_parameters = PathFollowingParameters(path_desired_yaw_angle__psi_des=third_target_ship.ship_model.yaw_angle,
                               path_desired_surge_velocity__u_des=third_target_ship.ship_model.forward_speed)

    static_values_ca = StaticValues(safe_radius__r0=300,
                                    switch_radius__rm=500,
                                    head_on_angle__alpha=15 * np.pi / 180,
                                    look_ahead_distance__delta=200,
                                    mass_matrix__m=third_target_ship.ship_model.mass_matrix(),
                                    dampening_matrix__d=third_target_ship.ship_model.linear_damping_matrix(),
                                    time_step=time_step,
                                    max_rudder_angle=third_target_ship.ship_model.ship_machinery_model.rudder_ang_max * np.pi / 180,
                                    collision_avoidance_speed=desired_speed_target_ship)

    target_ship_with_ca = SetBasedGuidance(path=path_following_parameters, static=static_values_ca)

    cont_params = Controllers(surge=0, yaw=0)


    # Collision avoidance parameters and object own ship
    own_path_following_parameters = PathFollowingParameters(
        path_desired_yaw_angle__psi_des=own_ship.yaw_angle,
        path_desired_surge_velocity__u_des=own_ship.forward_speed)

    own_static_values_ca = StaticValues(safe_radius__r0=300,
                                    switch_radius__rm=500,
                                    head_on_angle__alpha=15 * np.pi / 180,
                                    look_ahead_distance__delta=200,
                                    mass_matrix__m=own_ship.mass_matrix(),
                                    dampening_matrix__d=own_ship.linear_damping_matrix(),
                                    time_step=time_step,
                                    max_rudder_angle=own_ship.ship_machinery_model.rudder_ang_max * np.pi / 180,
                                    collision_avoidance_speed=desired_speed_own_ship)

    own_ship_with_ca = SetBasedGuidance(path=own_path_following_parameters, static=own_static_values_ca)

    while own_ship.int.time <= own_ship.int.sim_time:
        global_time = own_ship.int.time

        # Measure position and speed
        own_ship_north_position = own_ship.north
        own_ship_east_position = own_ship.east
        own_ship_heading = own_ship.yaw_angle
        own_ship_speed = own_ship.forward_speed

        # Check for grounding
        if own_ship.int.time % 400 == 0:
            grounding_index(dist_ground, own_ship.north, own_ship.east)


        # Check if own ship goes out of map and simulation should end
        if own_ship_north_position > test_own.north_side or own_ship_north_position < test_own.south_side \
                or own_ship_east_position > test_own.east_side or own_ship_east_position < test_own.west_side:
            own_ship.int.time = own_ship.int.sim_time + 1

        # Update variables
        own_dynamic_values_ca = UpdatedVariables(north__x=own_ship.north,
                                             east__y=own_ship.east,
                                             yaw_angle__psi=own_ship.yaw_angle,
                                             surge_speed__u=own_ship.forward_speed,
                                             sway_speed__v=own_ship.sideways_speed,
                                             yaw_rate__r=own_ship.yaw_rate,
                                             north_obstacle__xc=third_target_ship.ship_model.north,
                                             east_obstacle__yc=third_target_ship.ship_model.east,
                                             yaw_obstacle__psi_c=third_target_ship.ship_model.yaw_angle,
                                             surge_velocity_obstacle__u0=third_target_ship.ship_model.forward_speed)

        # Run the set based algorithm
        own_ship_with_ca.update_variables(updated=own_dynamic_values_ca, cont=cont_params)
        own_ship_with_ca.set_based_guidance_algorithm()

        # Update parameters
        if own_ship_with_ca.last_mode == own_ship_with_ca.path_following:
            # Find appropriate rudder angle and engine throttle (assume perfect measurements)
            rudder_angle = own_ship_navigation_system.rudder_angle_from_route(
                north_position=own_ship.north,
                east_position=own_ship.east,
                heading=own_ship.yaw_angle
            )
            throttle = own_ship_speed_controller.throttle(
                speed_set_point=desired_speed_own_ship,
                measured_speed=own_ship.forward_speed,
            )
            #print(f'rudder_angle={rudder_angle} for pathfinding')

        elif own_ship_with_ca.last_mode == own_ship_with_ca.object_avoidance:
            rudder_angle = heading_controller.rudder_angle_from_heading_setpoint(
                heading_ref=own_ship_with_ca.psi_des,
                measured_heading=own_ship.yaw_angle
            )
            throttle = own_ship_speed_controller.throttle(
                speed_set_point=desired_speed_own_ship,
                measured_speed=own_ship.forward_speed,
            )
            # print(f'rudder_angle={rudder_angle} for object avoidance')

        # Update controller parameters
        cont_param = Controllers(surge=throttle, yaw=rudder_angle)

        # Update and integrate differential equations for current time step
        own_ship.store_simulation_data(throttle)
        own_ship.update_differentials(engine_throttle=throttle, rudder_angle=rudder_angle)
        own_ship.integrate_differentials()

        # Add ownship snapshot to chart
        if time_since_snapshot > time_between_snapshots:
            snap_shot_id += 1
            ship_snap_shots.append((snap_shot_id, int(own_ship_east_position), int(own_ship_north_position),
                                    int(own_ship.yaw_angle * 180 / np.pi), "green"))
            time_since_snapshot = 0
        time_since_snapshot += own_ship.int.dt

        # Check if we should launch target ship
        index = 0
        for target_ship in list_of_target_ships:
            if global_time >= target_ship.start_time:
                if global_time <= target_ship.sim_time:

                    # Update parameters
                    if target_ship == third_target_ship:
                        dynamic_values_ca = UpdatedVariables(north__x=third_target_ship.ship_model.north,
                                                             east__y=third_target_ship.ship_model.east,
                                                             yaw_angle__psi=third_target_ship.ship_model.yaw_angle,
                                                             surge_speed__u=third_target_ship.ship_model.forward_speed,
                                                             sway_speed__v=third_target_ship.ship_model.sideways_speed,
                                                             yaw_rate__r=third_target_ship.ship_model.yaw_rate,
                                                             north_obstacle__xc=own_ship.north,
                                                             east_obstacle__yc=own_ship.east,
                                                             yaw_obstacle__psi_c=own_ship.yaw_angle,
                                                             surge_velocity_obstacle__u0=own_ship.forward_speed)

                        # Run the set based algorithm
                        target_ship_with_ca.update_variables(updated=dynamic_values_ca, cont=cont_params)
                        target_ship_with_ca.set_based_guidance_algorithm()

                        if target_ship_with_ca.last_mode == target_ship_with_ca.path_following:
                            # Find appropriate rudder angle and engine throttle (assume perfect measurements)
                            rudder_angle = target_ship.navigation_system.rudder_angle_from_route(
                                north_position=target_ship.ship_model.north,
                                east_position=target_ship.ship_model.east,
                                heading=target_ship.ship_model.yaw_angle
                            )
                            throttle = target_ship.speed_controller.throttle(
                                speed_set_point=desired_speed_target_ship,
                                measured_speed=target_ship.ship_model.forward_speed,
                            )
                            # print(f'rudder_angle={rudder_angle} for pathfinding')

                        elif target_ship_with_ca.last_mode == target_ship_with_ca.object_avoidance:
                            rudder_angle = heading_controller.rudder_angle_from_heading_setpoint(
                                heading_ref=target_ship_with_ca.psi_des,
                                measured_heading=target_ship.ship_model.yaw_angle
                            )
                            throttle = target_ship.speed_controller.throttle(
                                speed_set_point=desired_speed_target_ship,
                                measured_speed=target_ship.ship_model.forward_speed,
                            )
                            # print(f'rudder_angle={rudder_angle} for object avoidance')

                        # Update controller parameters
                        cont_param = Controllers(surge=throttle, yaw=rudder_angle)
                    else:
                        # Find appropriate rudder angle and engine throttle (assume perfect measurements)
                        rudder_angle = target_ship.navigation_system.rudder_angle_from_route(
                            north_position=target_ship.ship_model.north,
                            east_position=target_ship.ship_model.east,
                            heading=target_ship.ship_model.yaw_angle
                        )
                        throttle = target_ship.speed_controller.throttle(
                            speed_set_point=desired_speed_target_ship,
                            measured_speed=target_ship.ship_model.forward_speed,
                        )

                    # Check if target ship goes outside map and should be terminated
                    if target_ship.ship_model.north > test_own.north_side or target_ship.ship_model.north < test_own.south_side \
                            or target_ship.ship_model.east > test_own.east_side or target_ship.ship_model.east < test_own.west_side:
                        list_of_target_ships.remove(target_ship)

                    # Update and integrate differential equations for current time step
                    target_ship.ship_model.store_simulation_data(throttle)
                    target_ship.ship_model.update_differentials(engine_throttle=throttle, rudder_angle=rudder_angle)
                    target_ship.ship_model.integrate_differentials()

            # Collision index
            if own_ship.int.time % 60 == 0:
                collision_index(dist_ships, own_ship.north, own_ship.east, target_ship.ship_model.north, target_ship.ship_model.east)

            if target_ship == third_target_ship:
                # Add targetships snapshot to chart
                if time_since_snapshot > time_between_snapshots:
                    snap_shot_id += 1
                    ship_snap_shots.append(
                        (snap_shot_id, int(target_ship.ship_model.east), int(target_ship.ship_model.north),
                         int(target_ship.ship_model.yaw_angle * 180 / np.pi), "red"))
                    time_since_snapshot = 0
                time_since_snapshot += own_ship.int.dt

            index += 1



        own_ship.int.next_time()

    # Reordering distance array collision index
    # dist_ships = np.column_stack((dist_ships))

    # Reordering distance array grounding index
    # dist_ground = np.vstack(dist_ground)

    # print(dist_ships)
    # print(dist_ground)

    enc.add_vessels(*ship_snap_shots)
    enc.add_hazards(depth=5)

    #print(shapely.ops.nearest_points(own_ship.ship_drawings[0], enc.land))

    #enc.save_image("Image to show basic collision index")
    #enc.fullscreen_mode(True)
    enc.show_display()