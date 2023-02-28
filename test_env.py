
from os import environ
from typing import NamedTuple
import numpy as np

import seacharts

from ship_in_transit_simulator.models import EngineThrottleFromSpeedSetPoint, EnvironmentConfiguration, HeadingByRouteController, HeadingControllerGains,\
    LosParameters, ShipConfiguration, ShipModelSimplifiedPropulsion, \
    SimplifiedPropulsionMachinerySystemConfiguration, SimulationConfiguration, SpecificFuelConsumptionBaudouin6M26Dot3, \
    SpecificFuelConsumptionWartila6L26, MachineryModeParams, MachineryMode, MachineryModes, ThrottleControllerGains, ThrottleFromSpeedSetPointSimplifiedPropulsion

from Ane_alterations.Boarder_solution import MapBoarderRegulator

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
        speed_controller = ThrottleFromSpeedSetPointSimplifiedPropulsion(kp=3, ki=0.02, time_step=initial_states.integration_step)
        heading_controller_gains = HeadingControllerGains(kp=4, kd=90, ki=0.01)
        los_guidance_parameters = LosParameters(
            radius_of_acceptance=600,
            lookahead_distance=500,
            integral_gain=0.002,
            integrator_windup_limit=4000
        )
        navigation_system = HeadingByRouteController(route_name=self.sea_lane, heading_controller_gains=heading_controller_gains, 
            los_parameters=los_guidance_parameters, time_step=initial_states.integration_step, max_rudder_angle=self.machinery_config.max_rudder_angle_degrees * np.pi/180)
        return TargetShip(
            sim_time=start_time + initial_states.simulation_time,
            start_time=start_time,
            ship_model=target_ship_model,
            speed_controller=speed_controller,
            navigation_system=navigation_system
        )


if __name__ == "__main__":

    size = 19000, 19000
    center = 253536, 7045845 # east, north UTM-zone 33
    files = ['Trondelag.gdb']
    enc = seacharts.ENC(files=files, border=True, center=center,size=size, new_data=False)


    time_step = 0.5
    #own_ship_route_name = "test_route.txt"      # Redundant, could be added directly into function on line 180
    time_between_snapshots = 60

    main_engine_capacity = 2160e3
    diesel_gen_capacity = 510e3
    hybrid_shaft_gen_as_generator = 'GEN'

    desired_speed_own_ship = 7

    ship_config = ShipConfiguration(
        coefficient_of_deadweight_to_displacement=0.7,
        bunkers=200000,
        ballast=200000,
        length_of_ship=80,
        width_of_ship=16,
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
        initial_yaw_angle_rad=45 * np.pi / 180,
        initial_forward_speed_m_per_s=10,
        initial_sideways_speed_m_per_s=0,
        initial_yaw_rate_rad_per_s=0,
        integration_step=time_step,
        simulation_time=10000,
    )

    initial_states_second_ship = SimulationConfiguration(
        initial_north_position_m=7039562.36,
        initial_east_position_m=251040.06,
        initial_yaw_angle_rad=45 * np.pi / 180,
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
    first_target_ship = ship_factory.make_target_ship(start_time=0, initial_states=initial_states_first_ship)
    second_target_ship = ship_factory.make_target_ship(start_time=200, initial_states=initial_states_second_ship)
    list_of_target_ships = [first_target_ship, second_target_ship]


    test_own = MapBoarderRegulator(
        shipping_lane='own_ship_path_ane.txt',
        center=center,
        size=size
    )
    test_own.search_waypoints()
    test_own.update_txt_file()


    initial_states_own_ship = SimulationConfiguration(
        initial_north_position_m=test_own.start_north,
        initial_east_position_m=test_own.start_east,
        initial_yaw_angle_rad=test_own.angle, # in rads, formula: degrees*np.pi/180
        initial_forward_speed_m_per_s=7,
        initial_sideways_speed_m_per_s=0,
        initial_yaw_rate_rad_per_s=0,
        integration_step=time_step,
        simulation_time=10000,
    )
    print(test_own.angle)

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
        kp=4,
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
        route_name='updated_own_ship_path_ane.txt',
        heading_controller_gains=own_ship_heading_controller_gains,
        los_parameters=own_ship_los_guidance_parameters,
        time_step=initial_states_own_ship.integration_step,
        max_rudder_angle=machinery_config.max_rudder_angle_degrees * np.pi/180
    )

    time_since_snapshot = time_between_snapshots
    snap_shot_id = 0
    # Lists for storing ships and trails
    ship_snap_shots = []

    # List for storing crashes
    #crashes=[]


    while own_ship.int.time <= own_ship.int.sim_time:
        global_time = own_ship.int.time

        # Measure position and speed
        own_ship_north_position = own_ship.north
        own_ship_east_position = own_ship.east
        own_ship_heading = own_ship.yaw_angle
        own_ship_speed = own_ship.forward_speed

        # Check if own ship goes out of map and simulation should end
        if own_ship_north_position > test_own.north_side or own_ship_north_position < test_own.south_side \
        or own_ship_east_position > test_own.east_side or own_ship_east_position < test_own.west_side:
            own_ship.int.time = own_ship.int.sim_time + 1

        # Find appropriate rudder angle and engine throttle
        rudder_angle = own_ship_navigation_system.rudder_angle_from_route(
            north_position=own_ship_north_position,
            east_position=own_ship_east_position,
            heading=own_ship_heading
        )
        throttle = own_ship_speed_controller.throttle(
            speed_set_point=desired_speed_own_ship,
            measured_speed=own_ship_speed,
        )

        # Update and integrate differential equations for current time step
        own_ship.store_simulation_data(throttle)
        own_ship.update_differentials(engine_throttle=throttle, rudder_angle=rudder_angle)
        own_ship.integrate_differentials()

        # Add ownship snapshot to chart
        if time_since_snapshot > time_between_snapshots:
            snap_shot_id += 1
            ship_snap_shots.append((snap_shot_id, int(own_ship_east_position), int(own_ship_north_position), int(own_ship.yaw_angle*180/np.pi), "green"))
            time_since_snapshot = 0
        time_since_snapshot += own_ship.int.dt


        # Check if we should launch target ship
        for target_ship in list_of_target_ships:
            if  global_time >= target_ship.start_time:
                if global_time <= target_ship.sim_time:
                    # Find appropriate rudder angle and engine throttle (assume perfect measurements)
                    rudder_angle = target_ship.navigation_system.rudder_angle_from_route(
                        north_position=target_ship.ship_model.north,
                        east_position=target_ship.ship_model.east,
                        heading=target_ship.ship_model.yaw_angle
                    )
                    throttle = target_ship.speed_controller.throttle(
                        speed_set_point=5,
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


            # Add targetships snapshot to chart
            if time_since_snapshot > time_between_snapshots:
                snap_shot_id += 1
                ship_snap_shots.append((snap_shot_id, int(target_ship.ship_model.east), int(target_ship.ship_model.north),
                                        int(target_ship.ship_model.yaw_angle * 180 / np.pi), "red"))
                time_since_snapshot = 0
            time_since_snapshot += own_ship.int.dt


        own_ship.int.next_time()




    enc.add_vessels(*ship_snap_shots)
    enc.add_hazards(depth=5)

    #enc.save_image("Test_own_ship_route_1")
    #enc.fullscreen_mode(True)
    enc.show_display()





