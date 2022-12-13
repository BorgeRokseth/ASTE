
from os import environ
from typing import NamedTuple
import numpy as np

from ship_in_transit_simulator.models import EngineThrottleFromSpeedSetPoint, EnvironmentConfiguration, HeadingByRouteController, HeadingControllerGains, LosParameters, ShipConfiguration, ShipModelSimplifiedPropulsion, \
SimplifiedPropulsionMachinerySystemConfiguration, SimulationConfiguration, SpecificFuelConsumptionBaudouin6M26Dot3, \
    SpecificFuelConsumptionWartila6L26, MachineryModeParams, MachineryMode, MachineryModes, ThrottleControllerGains, ThrottleFromSpeedSetPointSimplifiedPropulsion


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
    time_step = 0.5
    own_ship_route_name = "own_ship_route.txt"

    main_engine_capacity = 2160e3
    diesel_gen_capacity = 510e3
    hybrid_shaft_gen_as_generator = 'GEN'

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
        current_velocity_component_from_north=-2,
        current_velocity_component_from_east=-2,
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
        initial_north_position_m=0,
        initial_east_position_m=0,
        initial_yaw_angle_rad=45 * np.pi / 180,
        initial_forward_speed_m_per_s=7,
        initial_sideways_speed_m_per_s=0,
        initial_yaw_rate_rad_per_s=0,
        integration_step=time_step,
        simulation_time=3600,
    )

    initial_states_second_ship = SimulationConfiguration(
        initial_north_position_m=10,
        initial_east_position_m=0,
        initial_yaw_angle_rad=45 * np.pi / 180,
        initial_forward_speed_m_per_s=7,
        initial_sideways_speed_m_per_s=0,
        initial_yaw_rate_rad_per_s=0,
        integration_step=time_step,
        simulation_time=3600,
    )

    initial_states_own_ship = SimulationConfiguration(
        initial_north_position_m=100,
        initial_east_position_m=0,
        initial_yaw_angle_rad=0 * np.pi / 180,
        initial_forward_speed_m_per_s=7,
        initial_sideways_speed_m_per_s=0,
        initial_yaw_rate_rad_per_s=0,
        integration_step=time_step,
        simulation_time=3600,
    )


    ship_factory = TargetShipMaker(ship_configuration=ship_config, environment=env_config, machinery_config=machinery_config, sea_lane="ship_in_transit_simulator/examples/route.txt")
    first_target_ship = ship_factory.make_target_ship(start_time=100, initial_states=initial_states_first_ship)
    second_target_ship = ship_factory.make_target_ship(start_time=400, initial_states=initial_states_second_ship)
    list_of_target_ships = [first_target_ship, second_target_ship]

    own_ship = ShipModelSimplifiedPropulsion(
        ship_config=ship_config,
        machinery_config=machinery_config,
        environment_config=env_config,
        simulation_config=initial_states_own_ship
    )

    speed_controller = ThrottleFromSpeedSetPointSimplifiedPropulsion(kp=3, ki=0.02, time_step=initial_states_own_ship.integration_step)
    own_ship_heading_controller_gains = HeadingControllerGains(kp=4, kd=90, ki=0.01)
    own_ship_los_guidance_parameters = LosParameters(
        radius_of_acceptance=600,
        lookahead_distance=500,
        integral_gain=0.002,
        integrator_windup_limit=4000
    )
    own_ship_navigation_system = HeadingByRouteController(route_name=own_ship_route_name, heading_controller_gains=own_ship_heading_controller_gains, 
        los_parameters=own_ship_los_guidance_parameters, time_step=initial_states_own_ship.integration_step, max_rudder_angle=machinery_config.max_rudder_angle_degrees * np.pi/180)



