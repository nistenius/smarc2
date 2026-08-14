#! /bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/tmux_layout.sh"

# LOCAL_* comes from the environment (bashrc), or -- on Data Cube machines -- from the
# GENERATED machine env file (gen_deployment_artifacts.py --env, single source:
# deployments.yaml + fleet.yaml). Environment wins; the real vehicle (orin) has no such
# file and is untouched by this. 2026-08-11 config-architecture session.
DC_MACHINE_ENV="${XDG_CONFIG_HOME:-$HOME/.config}/data-cube/unity_bridge.env"
if [[ -z "${LOCAL_ROBOT_NAME:-}" && -f "$DC_MACHINE_ENV" ]]; then
    set -a; source "$DC_MACHINE_ENV"; set +a
    echo "machine config: $DC_MACHINE_ENV (${DC_DEPLOYMENT:-?}/${DC_MACHINE_ID:-?})"
fi
if [[ -z "${LOCAL_ROBOT_NAME:-}" || -z "${LOCAL_MQTT_BROKER_IP:-}" || -z "${LOCAL_MQTT_BROKER_PORT:-}" ]]; then
    echo "ERROR: LOCAL_ROBOT_NAME / LOCAL_MQTT_BROKER_IP / LOCAL_MQTT_BROKER_PORT not set."
    echo "Set them in your bashrc, or generate the machine env file:"
    echo "    gen_deployment_artifacts.py --deployment <id> --machine <id> --env"
    exit 1
fi
ROBOT_NAME=$LOCAL_ROBOT_NAME
MQTT_BROKER_IP=$LOCAL_MQTT_BROKER_IP
MQTT_BROKER_PORT=$LOCAL_MQTT_BROKER_PORT
SSS_SAVE_PATH=/home/orin/sss_auto_save

SESSION=${ROBOT_NAME}_bringup
# check if there is already a tmux session with this name
if tmux has-session -t $SESSION 2>/dev/null; then
    # SAM_BRINGUP_IDEMPOTENT=1: VC / unity_bridge re-triggers bringup every power cycle without
    # requiring a manual tmux kill first (Data Cube Vehicle Control path).
    if [[ "${SAM_BRINGUP_IDEMPOTENT:-0}" == "1" ]]; then
        echo "tmux session $SESSION already running — idempotent success."
        exit 0
    fi
    echo "There is already a tmux session named $SESSION."
    echo "Please close it before launching this script."
    echo "Exiting."
    exit 1
fi


if [[ "$(whoami)" == "orin" ]]; then
    USE_SIM_TIME=False
    REALSIM=real
    MQTT_BROKER_IP=20.240.40.232
    MQTT_BROKER_PORT=1884
    # On the vehicle a health fault is terminal until a human clears it.
    LATCH_FAULTS=True
    # Hardware: estimator keeps its long-standing config and ignores message covariances.
    DR_SENSOR_ARGS="config_file:=sam.yaml use_sensor_covariance:=false"
    DR_ENV=""
    # Hardware keeps the long-standing 5-sample window; changing it is a team decision.
    RATE_WINDOW_SIZE=5
    DR_STRATEGY=FixedLagSmoothing
else
    USE_SIM_TIME=True
    REALSIM=simulation
    # Sim (2026-08-08, sim fidelity work): Unity sensors now publish measured noise and real
    # covariances. sam.yaml's sigmas (DVL 1e-5, GPS 1 mm) were tuned for the old noiseless sim
    # and are wildly overconfident against noisy data; sim.yaml is sane, and
    # use_sensor_covariance:=true takes DVL/GPS sigmas from the messages themselves.
    DR_SENSOR_ARGS="config_file:=sim.yaml use_sensor_covariance:=true"
    # The VM's GTSAM lives in /usr/local; without this the loader picks apt's 4.2 and the
    # estimator dies with undefined symbol GPSFactorArm (2026-08-07 session note).
    # PREPEND, don't replace: clobbering LD_LIBRARY_PATH hides the ROS libs themselves
    # (librcl_action.so ImportError). Escaped so it expands in the tmux pane, not here.
    DR_ENV="LD_LIBRARY_PATH=/usr/local/lib:\$LD_LIBRARY_PATH "
    # In the sim, every Unity editor stop/play drops all topics at once. With latching that
    # left sam_rate_health_node stuck on VEHICLE_HEALTH_ERROR forever, and wasp_bt rejects
    # every start-tst while health_status != READY -- so a single editor restart silently
    # blocked all missions until someone restarted the node by hand. Recover instead.
    LATCH_FAULTS=False
    # Unity editor hitches pause every topic together; a 5-sample window cannot tell that
    # apart from a dead sensor. 20 samples spans ~0.7 s at 30 Hz, which rides out a hitch
    # while still catching a sensor that actually stopped.
    RATE_WINDOW_SIZE=20
    # 2026-08-08: with FixedLagSmoothing(100 s) the initial prior ages out of the window.
    # Underwater the sim has NO absolute constraint (no GPS fix, no compass factor), so once
    # the prior is gone yaw and horizontal position are unobservable -- the graph becomes
    # underdetermined and GTSAM throws IndeterminantLinearSystemException and kills the node.
    # That is what ended the first honest run. ISAM2 keeps every variable (and therefore the
    # prior) while still updating incrementally. Override with DR_STRATEGY=... to compare.
    DR_STRATEGY=${DR_STRATEGY:-ISAM2}
fi

# Variables for wasp_bt.launch and wasp_mqtt_agent.launch
AGENT_TYPE=subsurface
PULSE_RATE=0.5 # Hz


# create a tmux session with a name
tmux -2 new-session -d -x 220 -y 60 -s "$SESSION"


if [[ $USE_SIM_TIME == "False" ]]; then
    SAM_CORE_CMD="ros2 launch sam_drivers sam_core.launch robot_name:=$ROBOT_NAME"
    SAM_PAYLOADS_CMD="ros2 launch sam_drivers sam_payloads.launch sss_out_file:=$SSS_SAVE_PATH/ high_freq:=true robot_name:=$ROBOT_NAME use_sim_time:=$USE_SIM_TIME"
    SAM_UWCOMMS_CMD="ros2 launch sam_drivers sam_uwcomms.launch robot_name:=$ROBOT_NAME use_sim_time:=$USE_SIM_TIME"
    tmux_make_layout "$SESSION" core "
    col(
        var(SAM_CORE_CMD),
        var(SAM_PAYLOADS_CMD),
        var(SAM_UWCOMMS_CMD)
    )"
fi

DESCRIPTION_CMD="ros2 launch sam_description sam_description.launch robot_name:=$ROBOT_NAME"
DR_CMD="${DR_ENV}ros2 launch hydrobatic_localization state_estimator.launch robot_name:=$ROBOT_NAME use_sim_time:=$USE_SIM_TIME use_motion_model:=false inference_strategy:=$DR_STRATEGY kf_interval_hz:=10 $DR_SENSOR_ARGS init_from_ground_truth:=false"

tmux_make_layout "$SESSION" dr "
col(
    3:var(DR_CMD),
    1:var(DESCRIPTION_CMD)
)"

BT_CMD="ros2 launch wasp_bt wasp_bt.launch robot_name:=$ROBOT_NAME agent_type:=$AGENT_TYPE pulse_rate:=$PULSE_RATE use_sim_time:=$USE_SIM_TIME"
# Controller selection (Session C, 2026-08-09): SAM_DIVE_LAUNCH picks the dive
# controller launch file.
#
# Default changed to the BLEND controller 2026-08-13, Ivan's call: "this has been the one
# working best up till now." The stock PID switches hard at |depth_error| <= 0.5 m and
# commands rpm_u_neutral above it -- so descending to a 2 m setpoint it never turns the
# props at all, and around the threshold it flips between modes and stutters. Measured on
# the rig that day: depth oscillating 1.25-1.67 m against a 2 m target, thruster1/2_rpm
# both 0, the vehicle diving and surfacing without ever driving forward. The blend
# controller blends on measured surge instead of switching (Session C, three matched runs:
# depth mean 1.53 -> 2.05 m, RMS 0.49 -> 0.19 m, prop cycling gone, 12.9 -> 7.0 min).
#
# THIS FILE IS THE DEFAULT FOR THE REAL BRINGUP. Setting it anywhere else does nothing --
# data-cube's vehicle_services only supplies context for nodes the supervisor itself
# launches, which in observe mode is none. Changing it there first, as was tried, is inert.
#   SAM_DIVE_LAUNCH=pid_wp_following ... sam_bringup.sh   -> back to stock, for A/B
DIVE_LAUNCH="${SAM_DIVE_LAUNCH:-blend_pid_wp_following}"
CONTROLLER_CMD="ros2 launch sam_diving_controller ${DIVE_LAUNCH}.launch robot_name:=$ROBOT_NAME use_sim_time:=$USE_SIM_TIME"
# EMERGENCY_ACTION_CMD="ros2 launch sam_emergency_action sam_emergency_action.launch robot_name:=$ROBOT_NAME"
# HEALTH_FAKER_CMD replaced by the real sam_health_checker (uncommented per request 2026-07-24):
# HEALTH_FAKER_CMD="ros2 topic pub /sam/smarc/vehicle_health std_msgs/msg/Int8 data:\ 0\ "
# rate_window_size: the 2026-08-07 session concluded sim needs 20 and hardware keeps 5, but
# nothing ever passed it -- so every sim run since has been aborting on single late messages.
# At the default 5 the window spans 0.17 s at 30 Hz: one hiccup reads as a 50 % rate loss,
# health goes ERROR for a tick, the BT raises the emergency flag and cancels the mission.
# That is exactly what killed the 2026-08-08 noisy-sensor run at t+10 s (IMU measured
# 23.98 Hz against a 20 Hz nominal, i.e. healthy on average and faulting on jitter).
HEALTH_CHECKER_CMD="ros2 launch sam_health_checker sam_rate_health_checker.launch robot_name:=$ROBOT_NAME use_sim_time:=$USE_SIM_TIME latch_faults:=$LATCH_FAULTS rate_window_size:=$RATE_WINDOW_SIZE"
DISCOVERY_SERVER_CMD="export ZENOH_CONFIG_OVERRIDE='listen/endpoints=[\"tcp/0.0.0.0:7447\"]' && ros2 run rmw_zenoh_cpp rmw_zenohd"
tmux_make_layout "$SESSION" bt+cont "
row(
    var(BT_CMD),
    col(
        2:var(CONTROLLER_CMD),
        1:var(HEALTH_CHECKER_CMD),
        1:var(DISCOVERY_SERVER_CMD)
    )
)"

SMARC_PUB_CMD="ros2 launch sam_smarc_publisher default.launch robot_name:=$ROBOT_NAME"
# Agent identity from the fleet registry when this machine has one (generated env file,
# LOCAL_WARAPS_AGENT -- see data-cube gen_deployment_artifacts.py). Without it the launch
# falls back to its historical $USER_<robot_name> default, so the real vehicle (orin, which
# sets no LOCAL_WARAPS_AGENT) behaves exactly as before. 2026-08-11.
AGENT_NAME_ARG=""
if [[ -n "${LOCAL_WARAPS_AGENT:-}" ]]; then
    AGENT_NAME_ARG=" agent_name:=$LOCAL_WARAPS_AGENT"
fi
MQTT_BRIDGE_CMD="ros2 launch str_json_mqtt_bridge waraps_bridge.launch broker_addr:=$MQTT_BROKER_IP broker_port:=$MQTT_BROKER_PORT robot_name:=$ROBOT_NAME domain:=subsurface context:=isee realsim:=$REALSIM use_sim_time:=$USE_SIM_TIME$AGENT_NAME_ARG"
# HEALTH_CHECKER_CMD="ros2 launch sam_health_checker sam_rate_health_checker.launch robot_name:=$ROBOT_NAME use_sim_time:=$USE_SIM_TIME"
# UTILS_CMD="ros2 launch smarc_bringups utilities.launch robot_name:=$ROBOT_NAME"

# SAM 2.2 perception (2026-08-09): Sonar 3D-15 + RealSense D435i. In sim this pane is a
# rate monitor of the Unity-published topics (one glance answers "is sim data flowing?");
# on hardware it launches the real drivers remapped to the sim-identical topic names.
# Disable with SAM_PERCEPTION=0 (e.g. when flying the old sam_auv_v1 prefab, whose
# missing perception topics would just be SILENT log noise).
if [[ $USE_SIM_TIME == "True" ]]; then PERCEPTION_SIM=true; else PERCEPTION_SIM=false; fi
PERCEPTION_CMD="ros2 launch sam_perception sam_perception.launch robot_name:=$ROBOT_NAME use_sim_time:=$USE_SIM_TIME simulation:=$PERCEPTION_SIM"

if [[ "${SAM_PERCEPTION:-1}" == "1" ]]; then
    tmux_make_layout "$SESSION" utils "
row(
    var(MQTT_BRIDGE_CMD),
    col(
        var(SMARC_PUB_CMD),
        var(PERCEPTION_CMD)
    )
)"
else
    tmux_make_layout "$SESSION" utils "
row(
    var(MQTT_BRIDGE_CMD),
    var(SMARC_PUB_CMD)
)"
fi





# Set default window
tmux select-window -t $SESSION:0
# Attach for interactive use (default). Set SAM_BRINGUP_ATTACH=0 when launched from
# Data Cube unity_bridge / VC so the script can run without a TTY and leave the
# session detached (nodes keep running in tmux).
if [[ "${SAM_BRINGUP_ATTACH:-1}" == "1" ]]; then
    tmux -2 attach-session -t $SESSION
fi
