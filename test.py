import numpy as np

from ship_in_transit_simulator.models import HeadingByReferenceController, HeadingControllerGains

heading_controller_gains = HeadingControllerGains(kp=4, kd=90, ki=0.01)
heading_controller = HeadingByReferenceController(gains=heading_controller_gains,
                                                  time_step=0.5,
                                                  max_rudder_angle= 30*np.pi/180)

heading_ref = 90*np.pi/180
yaw = 90*np.pi/180

angle = heading_controller.rudder_angle_from_heading_setpoint(heading_ref=heading_ref, measured_heading=yaw)
print(angle)

