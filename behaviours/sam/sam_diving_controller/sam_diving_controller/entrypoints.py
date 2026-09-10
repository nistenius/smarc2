# my_pkg/entrypoints.py
from __future__ import annotations

from .dive_runner import Components, Rates, run_mode

from .ParamUtils import DivingModelParam
from .DiveSub import DiveSub
from .DivePub import DivePub
# Unused, analyticalsamsim imports smarc_modelling as a
# pure py package, which is a PITA when using PID?
# from .SimPub import SimPub
# from .AnalyticalSAMSim import AnalyticalSAMSim
from .ConveniencePub import ConveniencePub

from .controllers.DiveControllerPID import DiveControllerPID
from .controllers.DiveControllerBlendPID import DiveControllerBlendPID
from .controllers.DiveControllerCascadePID import DiveControllerCascadePID

from .controllers.DiveControllerJoyPID import DiveControllerJoyPID

from .ActionServerDiveSub import DiveActionServerSub, HydropointServer, MPCPathServer, PIDPathServer
from smarc_action_base.smarc_action_base import ActionType
from smarc_msgs.action import BaseAction
from smarc_msgs.msg import Topics as SMaRCTopics


# --- Builders (wire-up per mode) ---

def _build_main(node, rates: Rates) -> Components:

    dive_pub = DivePub(node)
    dive_sub = DiveSub(node, dive_pub)

    dive_controller = DiveControllerMPC(node, dive_pub, dive_sub, rates.dive_controller)
    # dive_controller = DiveControllerPID(node, dive_pub, dive_sub, rates.dive_controller)

    return Components(dive_pub=dive_pub, dive_controller=dive_controller, dive_sub=dive_sub)


def _build_joy_depth(node, rates: Rates) -> Components:

    param = DivingModelParam(node).get_param()
    dive_sub = DiveSub(node, param)
    dive_pub = DivePub(node, dive_sub, param)
    dive_controller = DiveControllerJoyPID(node, dive_pub, dive_sub, param, rates.dive_controller)

    return Components(
        dive_pub=dive_pub,
        dive_controller=dive_controller,
        dive_sub=dive_sub,
        dive_pub_update=dive_pub.joy_update,  # <- special case handled cleanly
    )


#def _build_sim_sam(node, rates: Rates) -> Components:
#
#    param = DivingModelParam(node).get_param()
#    action_type = ActionType(BaseAction)
#    heartbeat_topic = SMaRCTopics.WARA_PS_ACTION_SERVER_HB_TOPIC
#
#    dive_sub = HydropointServer(node, "go_to_hydropoint", action_type, param, heartbeat_topic)
#    dive_pub = SimPub(node, dive_sub, param)
#    dive_controller = AnalyticalSAMSim(node, dive_pub, dive_sub, param, rates.dive_controller)
#    convenience_pub = ConveniencePub(node, dive_sub, dive_controller)
#
#    return Components(dive_pub=dive_pub, dive_controller=dive_controller,
#                      dive_sub=dive_sub, convenience_pub=convenience_pub)
#

def _build_pid_wp_following(node, rates: Rates) -> Components:

    param = DivingModelParam(node).get_param()
    action_type = ActionType(BaseAction)
    heartbeat_topic = SMaRCTopics.WARA_PS_ACTION_SERVER_HB_TOPIC

    dive_sub = DiveActionServerSub(node, "auv_depth_move_to", action_type, param, heartbeat_topic)
    dive_pub = DivePub(node, dive_sub, param)
    dive_controller = DiveControllerPID(node, dive_pub, dive_sub, param, rates.dive_controller)
    convenience_pub = ConveniencePub(node, dive_sub, dive_controller)

    return Components(
        dive_pub=dive_pub,
        dive_controller=dive_controller,
        dive_sub=dive_sub,
        convenience_pub=convenience_pub,
    )

def _build_blend_pid_wp_following(node, rates: Rates) -> Components:
    """
    Session C (2026-08-09): the blend-allocation controller as a SEPARATE server, so
    BT/MC can switch controllers per mission (or mid-mission) and A/B them.

    The action name is a parameter. Default is the same "auv_depth_move_to" the stock
    PID serves, so swapping the executable in the bringup launch is a drop-in A/B with
    no BT change. To run BOTH servers side by side for BT-level switching, launch this
    one with dive_action_name:=auv_depth_move_to_blend and give the BT a client for it.
    """
    param = DivingModelParam(node).get_param()
    action_type = ActionType(BaseAction)
    heartbeat_topic = SMaRCTopics.WARA_PS_ACTION_SERVER_HB_TOPIC

    node.declare_parameter('dive_action_name', 'auv_depth_move_to')
    action_name = node.get_parameter('dive_action_name').get_parameter_value().string_value

    dive_sub = DiveActionServerSub(node, action_name, action_type, param, heartbeat_topic)
    dive_pub = DivePub(node, dive_sub, param)
    dive_controller = DiveControllerBlendPID(node, dive_pub, dive_sub, param, rates.dive_controller)
    convenience_pub = ConveniencePub(node, dive_sub, dive_controller)

    return Components(
        dive_pub=dive_pub,
        dive_controller=dive_controller,
        dive_sub=dive_sub,
        convenience_pub=convenience_pub,
    )

def _build_cascade_pid_wp_following(node, rates: Rates) -> Components:
    """
    Session C part 3 (2026-08-09): triple-cascade controller (cross-track ILOS ->
    heading -> yaw-rate; depth -> pitch -> pitch-rate; speed-nested VBS/LCG).
    Same server pattern as the blend controller; tuning in config/cascade_pid.yaml.
    """
    param = DivingModelParam(node).get_param()
    action_type = ActionType(BaseAction)
    heartbeat_topic = SMaRCTopics.WARA_PS_ACTION_SERVER_HB_TOPIC

    node.declare_parameter('dive_action_name', 'auv_depth_move_to')
    action_name = node.get_parameter('dive_action_name').get_parameter_value().string_value

    dive_sub = DiveActionServerSub(node, action_name, action_type, param, heartbeat_topic)
    dive_pub = DivePub(node, dive_sub, param)
    dive_controller = DiveControllerCascadePID(node, dive_pub, dive_sub, param, rates.dive_controller)
    convenience_pub = ConveniencePub(node, dive_sub, dive_controller)

    return Components(
        dive_pub=dive_pub,
        dive_controller=dive_controller,
        dive_sub=dive_sub,
        convenience_pub=convenience_pub,
    )

def _build_pid_trajectory_tracking(node, rates: Rates) -> Components:
    param = DivingModelParam(node).get_param()
    action_type = ActionType(BaseAction)
    heartbeat_topic = SMaRCTopics.WARA_PS_ACTION_SERVER_HB_TOPIC

    dive_sub = PIDPathServer(node, "auv_trajectory_tracking", action_type, param)
    dive_pub = DivePub(node, dive_sub, param)
    dive_controller = DiveControllerPID(node, dive_pub, dive_sub, param, rates.dive_controller)
    convenience_pub = ConveniencePub(node, dive_sub, dive_controller)

    return Components(dive_pub=dive_pub, dive_controller=dive_controller,
                      dive_sub=dive_sub, convenience_pub=convenience_pub)

def _build_mpc_wp_following(node, rates: Rates) -> Components:

    param = DivingModelParam(node).get_param()
    action_type = ActionType(BaseAction)
    heartbeat_topic = SMaRCTopics.WARA_PS_ACTION_SERVER_HB_TOPIC

    dive_sub = HydropointServer(node, "go_to_hydropoint", action_type, param, heartbeat_topic)
    dive_pub = DivePub(node, dive_sub, param)
    dive_controller = DiveControllerMPC(node, dive_pub, dive_sub, param,
                                        ref_is_trajectory=False,
                                        rate=rates.dive_controller,)
    convenience_pub = ConveniencePub(node, dive_sub, dive_controller)

    return Components(
        dive_pub=dive_pub,
        dive_controller=dive_controller,
        dive_sub=dive_sub,
        convenience_pub=convenience_pub,
    )


def _build_mpc_trajectory_tracking(node, rates: Rates) -> Components:

    param = DivingModelParam(node).get_param()
    action_type = ActionType(BaseAction)

    dive_sub = MPCPathServer(node, "auv_trajectory_tracking", action_type, param)
    dive_pub = DivePub(node, dive_sub, param)
    dive_controller = DiveControllerMPC(
        node, dive_pub, dive_sub, param,
        ref_is_trajectory=True,
        rate=rates.dive_controller,
    )
    convenience_pub = ConveniencePub(node, dive_sub, dive_controller)

    return Components(dive_pub=dive_pub, dive_controller=dive_controller,
                      dive_sub=dive_sub, convenience_pub=convenience_pub)


def _build_mpc_and_pid_wp_following(node, rates: Rates) -> Components:
    """BOTH SERVERS, ONE `DivePub`, BEHIND AN ARBITER — option M1 of strategy §5.3.

    WHY IT EXISTS. The close-inspection orbit wants the ordinary lawnmower legs on
    `auv_depth_move_to` (the PID/blend waypoint server the team flies) AND the turbo-turn ring
    on `auv_trajectory_tracking` (the MPC, which is the group's published method and has run on
    the real Orin). `diving_node` has only ever run ONE controller family per bringup, so this
    is the first time the two would share a process — and sharing a process means sharing a
    writer, which is ADR-004 invariant 12's question.

    `OneWriterArbiter` is the answer: exactly one server may hold the writer at a time, the
    other is REFUSED BY NAME, and the refusal says who is holding it and for how long. It is
    pure python and is driven exhaustively in
    `test/test_one_writer_across_the_two_servers.py`.

    **NOT A DEFAULT, AND NOT FLOWN.** No launch file references this entry point, nothing on the
    hull or the rig calls it, and this session did not run it: acados is not installed on this
    machine and is UNMEASURED on vm1 (SETTLED §3ad), so the MPC half cannot even be imported
    here. The import is therefore inside the function, exactly as `mpc_wp_following` does it,
    and the REFUSAL is what an operator gets on a machine without acados rather than an
    ImportError from somewhere confusing. Whether the two families can actually share a process
    is rung R1 and is measured, not argued.
    """
    from .controllers.DiveControllerMPC import DiveControllerMPC
    from .one_writer_arbiter import OneWriterArbiter

    param = DivingModelParam(node).get_param()
    action_type = ActionType(BaseAction)
    heartbeat_topic = SMaRCTopics.WARA_PS_ACTION_SERVER_HB_TOPIC

    arbiter = OneWriterArbiter(("auv_depth_move_to", "auv_trajectory_tracking"),
                               now=lambda: node.get_clock().now().nanoseconds * 1e-9)

    wp_server = DiveActionServerSub(node, "auv_depth_move_to", action_type, param,
                                    heartbeat_topic)
    mpc_server = MPCPathServer(node, "auv_trajectory_tracking", action_type, param)
    # ONE DivePub. That is the whole point: two publishers on one actuator path is the state the
    # arbiter exists to make impossible, and building a second one here would make the arbiter
    # decorative.
    dive_pub = DivePub(node, wp_server, param)
    for srv in (wp_server, mpc_server):
        setattr(srv, "one_writer_arbiter", arbiter)
    node.get_logger().info(
        "mpc_and_pid_wp_following: BOTH auv_depth_move_to and auv_trajectory_tracking are "
        "served from this process, over ONE DivePub, behind a one-writer arbiter. UNFLOWN — "
        "no launch file references this entry point (strategy §5.3 M1, rung R1).")

    dive_controller = DiveControllerMPC(node, dive_pub, mpc_server, param,
                                        ref_is_trajectory=True, rate=rates.dive_controller)
    convenience_pub = ConveniencePub(node, wp_server, dive_controller)
    return Components(dive_pub=dive_pub, dive_controller=dive_controller,
                      dive_sub=wp_server, convenience_pub=convenience_pub)


# --- Console-script entry points (module-level functions) ---
def main():
    run_mode(node_name="DivingNode", build=_build_main)

def joy_depth():
    run_mode(node_name="JoyDivingNode", build=_build_joy_depth)

#def sim_sam():
#    run_mode(node_name="ActionServerDivingNode", build=_build_sim_sam)

def pid_wp_following():
    run_mode(node_name="PidWpFollowingNode", 
             build=_build_pid_wp_following,
             log_banner="PID Waypoint Following")

def blend_pid_wp_following():
    run_mode(node_name="BlendPidWpFollowingNode",
             build=_build_blend_pid_wp_following,
             log_banner="Blend PID Waypoint Following")

def cascade_pid_wp_following():
    run_mode(node_name="CascadePidWpFollowingNode",
             build=_build_cascade_pid_wp_following,
             log_banner="Cascade PID Waypoint Following")

def pid_trajectory_tracking():
    run_mode(node_name="PidTrajectoryTrackingNode",
             build=_build_pid_trajectory_tracking,
             log_banner="PID Trajectory Tracking")

def mpc_wp_following():
    from .controllers.DiveControllerMPC import DiveControllerMPC
    run_mode(node_name="MpcWpFollowingNode",
             build=_build_mpc_wp_following,
             log_banner="MPC Waypoint Following")

def mpc_trajectory_tracking():
    from .controllers.DiveControllerMPC import DiveControllerMPC
    run_mode(node_name="MpcTrajectoryTracking",
             build=_build_mpc_trajectory_tracking,
             log_banner="MPC Trajectory tracking")

def mpc_and_pid_wp_following():
    """Option M1 (strategy §5.3). ADDITIVE, NON-DEFAULT, REFERENCED BY NO LAUNCH FILE."""
    run_mode(node_name="MpcAndPidWpFollowingNode",
             build=_build_mpc_and_pid_wp_following,
             log_banner="MPC + PID waypoint following, one writer (UNFLOWN)")
