std_msgs/Header header
uint8   type          # Packet Type, 0xE2 = Sonar 8 Bit
uint32  time          # Ping time   [100 us]
uint8   frequency_id  # Freq Id, currently 0x00
int16   gain
uint16  decimation
float32 max_duration  # Max travel time of outermost bins (s)
uint8[] port_channel
uint8[] starboard_channel
uint8[] extra_channel
# Interferometric PHASE, 16-bit signed (+-32768 <-> +-pi), split high/low byte.
#
# ALIGNED 2026-08-29 WITH THE REAL DRIVER. This file previously declared
#     port_channel_angle_high / _low, starboard_channel_angle_high / _low
# with extra_channel LAST. The real interferometric side scan we own -- the ISSS stick,
# recorded 2024-12-10 at Askoe -- publishes the names and the ORDER below; its own inline
# definition is in campaigns/2024-12-10_Asko/isss/2024-12-10-12-01-07.bag.
#
# Both differences mattered:
#   * NAME: nothing written for one could read the other.
#   * ORDER: ROS serialisation is positional, so with extra_channel last the four arrays
#     decoded one field out of step. That is exactly what happened to sim run 7
#     (sam_mk2_02_51_1787981003): the Unity publisher had already been aligned to the real
#     driver, this file had not, and the recorded phase arrays are shifted by one and are
#     not usable. Port/starboard intensity are ahead of the divergence and are unaffected.
#   * QUANTITY: it is raw, wrapped PHASE, not a finished elevation angle. Measured on the
#     real record: coherence 0.79 (weak echo) to 0.90 (strong), phase noise 0.57 -> 0.46 rad,
#     deterministic across-swath structure only ~0.24 rad -- i.e. noise-dominated.
#
# REBUILD REQUIRED: anything compiled against smarc_msgs must be rebuilt before this takes
# effect (`colcon build --packages-select smarc_msgs` on vm1 and on the hull). Until then a
# publisher and a subscriber can disagree about this message silently -- SETTLED section 3c.
uint8[] port_channel_phase_15_8
uint8[] port_channel_phase_7_0
uint8[] starboard_channel_phase_15_8
uint8[] starboard_channel_phase_7_0
