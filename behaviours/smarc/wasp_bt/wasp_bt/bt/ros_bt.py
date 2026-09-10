#!/usr/bin/python3

import json
from std_msgs.msg import String

import operator
import typing

import py_trees as pt
from py_trees.composites import Selector as Fallback
from py_trees.composites import Sequence, Parallel
from py_trees.blackboard import Blackboard
from py_trees.decorators import Inverter
from py_trees.common import Status, ParallelPolicy
from py_trees.behaviours import Running, Success, Failure
from smarc_msgs.msg import Topics as SMaRCTopics

from ..vehicles.vehicle import IVehicleStateContainer
from ..vehicles.sensor import SensorNames
from .i_has_vehicle_container import HasVehicleContainer
from .i_has_clock import HasClock

from wasp_bt.waraps.waraps_task_handler import (WaraPSTaskHandler, HasWaraPSTaskHandler,
                                                WaraPSTaskStates, BT_PROVIDED_TASKS)

from smarc_action_base.smarc_action_base import ActionType
from wasp_bt.bt.client import BTActionClient
from smarc_msgs.action import BaseAction



from .conditions import C_TaskIs,\
                        C_TaskStatus,\
                        C_AbortedPreviousTask,\
                        C_NoEmergencyAbortSignalDetected,\
                        C_VehicleHealthStatus,\
                        C_HealthNodeAlive,\
                        C_HasHeardFromVehicleHealth,\
                        C_MissionNotInError,\
                        C_MissionJustEnded

from .actions import A_Abort,\
                     A_Heartbeat,\
                     A_ActionClient,\
                    A_JustChillFor,\
                    A_ClearTaskQueue,\
                    A_TaskAbortedFlagReset, \
                    A_Chilling,\
                    A_WaitForData,\
                    A_ClearCurrentTask

from .farm_inspection import (A_FarmInspection, A_SurfaceAndReport,
                              A_EndOfMissionSurface, FARM_INSPECTION_TASK)
from .target_inspection import (A_CloseInspection, A_ResumeAtDiversionPoint,
                                C_TargetCandidatePending)

class BT(HasVehicleContainer, HasClock, HasWaraPSTaskHandler):
    def __init__(self,
                 vehicle_container:IVehicleStateContainer,
                 task_handler:WaraPSTaskHandler,
                 now_seconds_func: typing.Callable,
                 get_ready_action: BTActionClient = None,
                 emergency_action: BTActionClient = None,
                 action_client_list: typing.List[BTActionClient] = None,
                 bt_health_timeout: float = 10.0
                 ):
        """
        vehicle_container: An object that has a field "vehicle_state" which
            returns a vehicles.vehicle.IVehicleState type of object.
            SAMAuv, ROSVehicle, etc. should all fit this
        """
        self._vehicle_container = vehicle_container
        self._task_handler = task_handler
        self._bt = None
        self.action_client_list = action_client_list
        self._now_seconds_func = now_seconds_func
        self.emergency_action = emergency_action
        self.get_ready_action = get_ready_action

        self._last_state_str = ""

        self._bt_health_timeout = bt_health_timeout

        # Keep action client instances alive across tree rebuilds so in-flight
        # goals and cancel handles remain reachable.
        self._action_client_cache: typing.Dict[str, BTActionClient] = {}
        
        # Add tracking for dynamic tree updates
        self._last_available_tasks = []
        self._task_handler_node = None  # Reference to the task handler node in the tree

    @property
    def vehicle_container(self) -> IVehicleStateContainer:
        return self._vehicle_container
    
    @property
    def mqtt_interactor(self) -> typing.Any:
        return self._task_handler
    
    @property
    def now_seconds(self) -> int:
        return self._now_seconds_func()
    
    def _liveliness_tree(self):
        liveliness_tree = Parallel("P_Liveliness", policy=ParallelPolicy.SuccessOnAll(synchronise=False) , children=[
            A_WaitForData(self, SensorNames.VEHICLE_HEALTHY),
            A_WaitForData(self, SensorNames.POSITION)
            # Maybe add other sensors too, depth, altitude?
        ])

        return liveliness_tree
    
    def _health_tree(self):
        """
        A tree that checks the health status of the vehicle
        """

        health_checks = Fallback("F_Health_Handler", memory=False, children=[
            Sequence("S_Health_Status", memory=False, children=[
                # C_HasHeardFromVehicleHealth(self._task_handler),  # check if the vehicle health returns SUCCESS (Vehicle is ready)
                C_HealthNodeAlive(self._task_handler, timeout=self._bt_health_timeout),  # check if the last heartbeat was within 10 seconds
                Fallback("F_Health_Checks", memory=False, children=[
                    C_VehicleHealthStatus(self._task_handler, desired_status = SMaRCTopics.VEHICLE_HEALTH_READY),
                    C_VehicleHealthStatus(self._task_handler, desired_status = SMaRCTopics.VEHICLE_HEALTH_WAITING),
                ]),
            ]),
            A_Abort(self._task_handler),
        ])

        return health_checks

    def _handle_emergency_tree(self):
        """
        A tree that handles emergency situations, such as aborting the mission
        """

        emergency_children = [C_NoEmergencyAbortSignalDetected(self._task_handler)]
        if self.emergency_action is not None:
            # if there is an emergency action given to us
            # first, check if the action client is available
            availability_check = self.emergency_action._setup(num_iters=3)
            if not availability_check:
                # if the action client is not available, we cannot run it
                # we can just chill -- but say EMERGENCY PARKED, not "idle" (#29). This branch is
                # only reached when the no-emergency check has already FAILED.
                emergency_children.append(A_Chilling(self, A_Chilling.ROLE_EMERGENCY_PARKED))
            else:
                # if the action client is available, we can run it
                emergency_children.append(
                    A_ActionClient(
                        self.emergency_action, 
                        bt = self,
                        task_handler = self._task_handler
                    )
                )
        else:
            # if there is no emergency action, we can just chill -- again, EMERGENCY PARKED (#29)
            emergency_children.append(A_Chilling(self, A_Chilling.ROLE_EMERGENCY_PARKED))

        return Fallback("F_HandleEmergency", memory=False, children=emergency_children)
                    
    def _one_task_tree(self, task_name: str, action_client: BTActionClient):
        """
        A tree that handles a single task type, such as move-to or depth-move-to
        """
        task_tree = Sequence(f"S_{task_name}", memory=False, children=[
            C_MissionNotInError(self._task_handler),  # Fail immediately if mission is in ERROR
            C_TaskIs(self._task_handler, task_name),
            Fallback("F_StatusCheck", memory=False, children=[
                C_TaskStatus(self._task_handler, WaraPSTaskStates.STARTED.value),
                C_TaskStatus(self._task_handler, WaraPSTaskStates.RESUMED.value),
                C_TaskStatus(self._task_handler, WaraPSTaskStates.RUNNING.value),
            ]),

            # run the action client
            A_ActionClient(
                action_client,
                bt = self,
                task_handler = self._task_handler
            ),
            # when done, clear the task queue
            A_ClearCurrentTask(self._task_handler),
        ])

        return task_tree

    def _bt_provided_task_tree(self, task_name: str, action_client: BTActionClient):
        """The subtree for a task the TREE executes itself, e.g. `auv-farm-inspection`.

        Structurally identical to `_one_task_tree` — same guards, same status gate, same
        clear-when-done — with the single `A_ActionClient` swapped for the behaviour that
        streams many goals through THAT SAME CLIENT. Ending with `A_SurfaceAndReport` is
        invariant 5b: the tree is the only thing that knows a mission is over, and today it
        just falls through to `A_Chilling` while the controller holds its last depth.

        `memory=True` on the inner sequence, unlike `_one_task_tree`: these two children are
        SEQUENTIAL PHASES of one task, not a re-evaluated guard chain. Without memory, a
        SUCCESS from the inspection would be re-run from the top on the next tick and the
        vehicle would fly the whole survey again.
        """
        robot_name = self._task_handler.wara_ps_dict["name"]
        node = self._task_handler._node
        return Sequence(f"S_{task_name}", memory=False, children=[
            C_MissionNotInError(self._task_handler),
            C_TaskIs(self._task_handler, task_name),
            Fallback("F_StatusCheck", memory=False, children=[
                C_TaskStatus(self._task_handler, WaraPSTaskStates.STARTED.value),
                C_TaskStatus(self._task_handler, WaraPSTaskStates.RESUMED.value),
                C_TaskStatus(self._task_handler, WaraPSTaskStates.RUNNING.value),
            ]),
            Sequence(f"S_{task_name}_phases", memory=True, children=[
                A_FarmInspection(action_client, self, self._task_handler, node, robot_name),
                A_SurfaceAndReport(action_client, self, self._task_handler, node, robot_name),
            ]),
            A_ClearCurrentTask(self._task_handler),
        ])

    def _get_ready_tree(self, task_name: str, action_client: BTActionClient):
        """
        A tree that handles a single task type, such as move-to or depth-move-to
        """


        # check if the action server is alive (setup like below)
        status = action_client._setup(num_iters=1, timeout = 0.5)
        if not status:
            # if the action server is not alive, we cannot run the action
            self._task_handler._node.get_logger().warn(f"Action server for {task_name} is not alive")

            # remove the action from the available tasks in the WaraPSTaskHandler so that it doesn't show up in the task handler tree
            removed = self._task_handler.remove_available_task(task_name=task_name)
            if removed:
                self._task_handler._node.get_logger().info(
                    f"Removed unavailable task '{task_name}' from available tasks"
                )
            
            return None

        ready_tree = Sequence(f"S_{task_name}", memory=False, children=[

            C_MissionNotInError(self._task_handler),  # Fail immediately if mission is in ERROR

            Fallback("F_CanIGetReady?", memory=False, children = [
                C_VehicleHealthStatus(self._task_handler, desired_status = SMaRCTopics.VEHICLE_HEALTH_WAITING),
                C_VehicleHealthStatus(self._task_handler, desired_status = SMaRCTopics.VEHICLE_HEALTH_READY),
            ]),

            C_TaskIs(self._task_handler, task_name),
            Fallback("F_StatusCheck", memory=False, children=[
                C_TaskStatus(self._task_handler, WaraPSTaskStates.STARTED.value),
                C_TaskStatus(self._task_handler, WaraPSTaskStates.RESUMED.value),
                C_TaskStatus(self._task_handler, WaraPSTaskStates.RUNNING.value),
            ]),

            # run the action client
            A_ActionClient(
                action_client,
                bt = self,
                task_handler = self._task_handler
            ),
            # when done, clear the task queue
            A_ClearCurrentTask(self._task_handler),
        ])

        return ready_tree


    def _target_inspection_tree(self, action_client: BTActionClient):
        """The adaptive close inspection, as a HIGHER-PRIORITY SIBLING of the mission tree.

        THIS IS THE PREEMPTION, and it is the whole reason the subtree sits where it sits.
        `F_Task_Handler` is a Fallback with `memory=False`, so its children are re-evaluated in
        priority order on every tick. The moment `C_TargetCandidatePending` succeeds, this
        subtree returns RUNNING, py_trees invalidates everything below it — including the
        `A_ActionClient` flying the operator's waypoint — and terminates it with
        `Status.INVALID`. `smarc_action_base/bt_action_client_action.py:110-136` handles exactly
        that ("Preempted by higher priority in tree, cancelling goal") and then calls
        `get_ready()`, so when this subtree stops matching the task tree ticks again,
        `initialise()` re-sends the goal from `get_current_task_params()` — the same waypoint,
        still `tasks_executing[0]`, still queued.

        THE PATH THAT MUST NOT BE USED, measured 2026-09-09 (SETTLED §3ad and its correction):
        cancelling the goal from inside the inspection behaviour. `actions.py:322-534` lists
        `ActionClientState.CANCELLED` among `A_ActionClient`'s FAILURE states; driven, that value
        is intercepted at the top of `update()` (`get_ready()`, RUNNING) so a clean side-cancel
        does NOT delete the leg — but a cancel that FAILS lands the client in `REJECTED`, which
        DOES reach the failure branch and `clear_current_task()`, and the lawnmower silently loses
        a waypoint. Preemption by tree priority never depends on how a cancel resolves.

        `Fallback([A_CloseInspection, Success])`: an inspection that REFUSES must still fly the
        resume legs. A refusal is a mission phase ending, not a reason to leave the vehicle at
        the ring; the refusal itself is already reported by the behaviour and recorded on the
        latch as `not_inspected` with its reason.

        `memory=True` on the phase sequence, as `_bt_provided_task_tree` uses it and for the
        same reason: these are SEQUENTIAL PHASES of one diversion, not a re-evaluated guard
        chain. `memory=False` on the outer sequence, so the mission-error gate and the latch are
        re-read every tick and a cleared latch drops this subtree out of the way immediately.
        """
        robot_name = self._task_handler.wara_ps_dict["name"]
        node = self._task_handler._node
        return Sequence("S_TargetInspection", memory=False, children=[
            C_MissionNotInError(self._task_handler),
            # The policy gate is inside this condition: it only latches a candidate when the
            # mission carried an `adaptive` block, so a mission without one can never divert.
            C_TargetCandidatePending(self._task_handler, node, robot_name),
            Sequence("S_TargetInspection_phases", memory=True, children=[
                Fallback("F_InspectOrResumeAnyway", memory=False, children=[
                    A_CloseInspection(action_client, self, self._task_handler, node, robot_name),
                    Success(name="A_InspectionRefused_ResumeAnyway"),
                ]),
                A_ResumeAtDiversionPoint(action_client, self, self._task_handler, node,
                                         robot_name),
            ]),
        ])

    def _task_handler_tree(self, action_client_list: typing.List[BTActionClient] = None):
        """
        Fallback root node, connecting together sequences of {is the current action a certain kind of action? If so, run the corresponding action server}
        """

        def _remember_action_client(action_client: BTActionClient):
            self._action_client_cache[action_client.get_action_name()] = action_client
            return action_client

        def _get_or_create_action_client(ros_task_name: str, action_type: ActionType):
            cached_action_client = self._action_client_cache.get(ros_task_name)
            if cached_action_client is not None:
                return cached_action_client

            return _remember_action_client(
                BTActionClient(self._task_handler._node, ros_task_name, action_type)
            )

        task_children = [
            # check if the previous task was aborted, if so, reset the flag
            Sequence("S_BreatheAfterAbort", memory=False, children=[
                C_AbortedPreviousTask(self._task_handler),
                A_TaskAbortedFlagReset(self._task_handler),
            ]),
        ]

        if self.get_ready_action is not None:
            get_ready_tree = self._get_ready_tree("get-ready", self.get_ready_action)
            if get_ready_tree is not None:
                task_children.append(get_ready_tree)

        # create a action_client_list from the heartbeats of action clients, as stored by the WaraPSTaskHandler

        tasks_available = self._task_handler.get_available_tasks()

        ros_task_names = []

        for i in range(len(tasks_available)):
            # we will wait for the next task to be available
            ros_task_name = tasks_available[i]["ros_name"]
            
            # only append to the list of available task if it's not the emergency task. We don't want emergency to be available to the user in the task handler tree.
            if "emergency" not in ros_task_name and "ready" not in ros_task_name:
                ros_task_names.append(ros_task_name)
            
        # self._task_handler._node.get_logger().info(f"Available tasks: {ros_task_names}")

        # if action_client_list is None, we will use the action clients from the WaraPSTaskHandler
        if action_client_list == None:
            action_type = ActionType(BaseAction)
            
            action_client_list = [
                _get_or_create_action_client(ros_task_name, action_type)
                for ros_task_name in ros_task_names
            ]
        else:
            for action_client in action_client_list:
                _remember_action_client(action_client)

        mission_children = [
            C_VehicleHealthStatus(self._task_handler, desired_status=SMaRCTopics.VEHICLE_HEALTH_READY)
        ]

        mission_task_children = []

        # we will append the task trees to this list programmatically
        if action_client_list is not None:
            self._task_handler._node.get_logger().info(f"Action clients: {[ac.get_action_name() for ac in action_client_list]}")

            for action_client in action_client_list:

                # first, check if the corresponding action server is available
                availability_check = action_client._setup(num_iters = 1, timeout = 0.5)

                if not availability_check:
                    # if the action client is not available, skip it
                    continue

                # if the action client is available, we can proceed

                # parse the action client name to get the task name according to the WaraPS naming convention
                task_name = action_client.get_action_name().split('/')[-1].replace("_", "-")

                task_tree = self._one_task_tree(task_name, action_client)
                mission_task_children.append(task_tree)

                # Tasks the TREE provides on top of this same action server. They reuse this
                # client instance — one client, one server, one writer (SETTLED §1c and
                # invariant 12) — and appear only while the handler lists them as available,
                # which happens only while the provider's heartbeat is arriving.
                available_names = [t["name"] for t in self._task_handler.get_available_tasks()]
                for provided, provider in BT_PROVIDED_TASKS.items():
                    if provider != task_name or provided not in available_names:
                        continue
                    mission_task_children.append(
                        self._bt_provided_task_tree(provided, action_client))

        # make a fallback out of mission_task_children
        mission_task_fallback = Fallback("F_Tasks", memory=False, children=mission_task_children)

        # add mission_task_fallback to the mission_children
        mission_children.append(mission_task_fallback)

        # construct the mission tree
        mission_tree = Sequence("S_Mission", memory=False, children=mission_children)


        # THE ADAPTIVE CLOSE INSPECTION GOES IN AHEAD OF THE MISSION TREE (2026-09-09).
        #
        # Order IS the mechanism (see `_target_inspection_tree`): a Fallback child listed before
        # `S_Mission` has higher priority, and py_trees' own invalidation is what preempts the
        # running waypoint goal. Appending it after the mission tree would make it unreachable
        # while any task matched, which is every moment of a mission.
        #
        # It is built only when there is a client to stream through — the SAME cached client the
        # ordinary `auv-depth-move-to` task uses. One client, one server (SETTLED §1c).
        if action_client_list:
            task_children.append(self._target_inspection_tree(action_client_list[0]))

        # add the mission tree to task handler
        task_children.append(mission_tree)

        # INVARIANT 5b, THE ORDINARY PATH (2026-08-29). Between "the mission stopped matching"
        # and "idle" there is one thing the vehicle owes: coming up.
        #
        # `A_SurfaceAndReport` has existed since 2026-08-17 but only inside the
        # `auv-farm-inspection` subtree. Every OTHER mission -- every plain waypoint run --
        # ended by falling straight through to `A_Chilling` with the diving controller still
        # holding its last commanded depth, indefinitely. Recovery needs the vehicle visible,
        # and surfacing is how it resets accumulated DR error.
        #
        # It also silently broke recording: `bridge_node`'s end-of-run rule needs *mission
        # ended AND SURFACED AND idle*, where "surfaced" is the controller's own
        # `ctrl/neutral_handoff` verdict and not a depth reading. No verdict, no grace, no
        # stop -- so the bag was never closed and never had a `metadata.yaml`. Seen twice on
        # 2026-08-29, and the hand workaround (killing the recorder) destroyed a flown mission,
        # because `bridge_node` owns that recorder and restarts it on the same path.
        #
        # Ordering is the safety argument. This sits AFTER the mission tree, so it is reached
        # only when no task matches; the condition is an EDGE (a mission ran, then stopped), so
        # a vehicle that has never flown does not blow its tank on power-up; and it latches on
        # the TASK HANDLER, which outlives the rebuilds this subtree undergoes whenever an
        # action server's heartbeat changes. Once attempted -- confirmed, timed out, or held by
        # the protective stop -- it stops matching and idle resolves to `A_Chilling` as before.
        #
        # `memory=True`: the two children are sequential phases of one recovery, not a
        # re-evaluated guard chain. Same reason `_bt_provided_task_tree` uses it.
        if action_client_list:
            surface_client = action_client_list[0]
            robot_name = self._task_handler.wara_ps_dict["name"]
            task_children.append(Sequence("S_EndOfMissionSurface", memory=True, children=[
                C_MissionJustEnded(self._task_handler),
                A_EndOfMissionSurface(surface_client, self, self._task_handler,
                                      self._task_handler._node, robot_name),
            ]))

        # add the chill task. THIS one is genuine idle -- the resting state of a healthy vehicle
        # with no mission, and the tip the mission gate looks for before an upload (#29).
        task_children.append(A_Chilling(self, A_Chilling.ROLE_IDLE))

        task_handler = Fallback("F_Task_Handler", memory=False, children=task_children)
                                
        return task_handler

    def _update_task_handler_tree(self):
        """
        Check if available tasks have changed and rebuild the task handler subtree if needed.
        Returns True if tree was updated, False otherwise.
        """
        if self._bt is None or self._task_handler_node is None:
            return False
            
        # Get current available tasks
        current_tasks = self._task_handler.get_available_tasks()
        current_task_names = [task["ros_name"] for task in current_tasks 
                             if "emergency" not in task["ros_name"] and "ready" not in task["ros_name"]]
        
        # Check if tasks have changed
        if set(current_task_names) != set(self._last_available_tasks):
            self._task_handler._node.get_logger().info(
                f"Tasks changed from {self._last_available_tasks} to {current_task_names}. Rebuilding tree..."
            )
            
            # Rebuild the task handler tree
            new_task_handler = self._task_handler_tree(self.action_client_list)
            
            # Find the task handler node in the tree and replace it
            # This assumes F_Task_Handler is a direct child of S_Root
            root = self._bt.root
            for i, child in enumerate(root.children):
                if child.name == "F_Task_Handler":
                    # Replace the old node with the new one
                    root.children[i] = new_task_handler
                    # Setup the new subtree
                    new_task_handler.setup_with_descendants()
                    self._task_handler_node = new_task_handler
                    break
            
            self._last_available_tasks = current_task_names
            return True
            
        return False

    def setup(self) -> bool:

        children = [
            A_Heartbeat(self),
            self._handle_emergency_tree(),
            # self._liveliness_tree(),
            self._health_tree(),
            
            # self._safety_tree(), # should look at julian safety node topic

            # add the mission tree
            self._task_handler_tree(self.action_client_list),
            # self._run_tree()
        ]

        # clean out Nones   
        children = [c for c in children if c is not None]

        # make the sequence tree
        root = Sequence("S_Root", memory=False, children=children)

        self._bt = pt.trees.BehaviourTree(root)
        
        # Store reference to task handler node for dynamic updates
        for child in root.children:
            if child.name == "F_Task_Handler":
                self._task_handler_node = child
                break
        
        # Initialize tracking for available tasks
        tasks_available = self._task_handler.get_available_tasks()
        self._last_available_tasks = [task["ros_name"] for task in tasks_available 
                                     if "emergency" not in task["ros_name"] and "ready" not in task["ros_name"]]
        
        return self._bt.setup()



    def tick(self):
        # Update tree structure before ticking if tasks changed
        self._update_task_handler_tree()
        self._bt.tick()


def wasp_bt():
    from ..vehicles.smarc_vehicle import GenericSMaRCVehicle
    from ..vehicles.vehicle import VehicleState, UnderwaterVehicleState
    from wasp_bt.bt.client import BTActionClient
    from smarc_msgs.msg import Topics

    from smarc_action_base.smarc_action_base import (
        ActionFeedback,
        ActionResult,
        ActionType,
        SMARCActionClient,
    )
    from smarc_msgs.action import BaseAction

    import rclpy, sys
    import uuid

    rclpy.init(args=sys.argv)
    node = rclpy.create_node("wasp_bt_executor")

    def ros_seconds() -> int:
        nonlocal node
        secs, _ = node.get_clock().now().seconds_nanoseconds()
        return int(secs)
    
    def ros_seconds_float() -> float:
        nonlocal node
        secs, nsecs = node.get_clock().now().seconds_nanoseconds()
        return float(secs) + float(nsecs) * 1e-9


    bt_status_pub = node.create_publisher(String, Topics.BT_STATUS_TOPIC, qos_profile=10)

    
    # agent = SAMAuv(node)
    agent = GenericSMaRCVehicle(node, UnderwaterVehicleState)
    action_type = ActionType(BaseAction)
    
    # get-ready action client: None if does not exist, else
    get_ready_action_client = BTActionClient(node, "get_ready", action_type)
    # get_ready_action_client = None

    # emergency action
    emergency_action_client = BTActionClient(node, "emergency_action", action_type)
    # emergency_action_client = None

    # list of remaining action clients
    action_client_list = [
        BTActionClient(node, "move_to", action_type),
        BTActionClient(node, "auv_depth_move_to", action_type),
        BTActionClient(node, "cruise_depth_at_heading", action_type),
        # ADD NEW ACTION CLIENTS HERE

    ]

    # action_client_list = None
    # Declare and get parameters with defaults
    node.declare_parameter("agent_type", "air")
    node.declare_parameter("pulse_rate", 1.0) # Hz
    node.declare_parameter("domain", "simulation")

    agent_type = node.get_parameter("agent_type").value
    levels = ["sensor", "direct_execution", "tst_execution"]
    pulse_rate = node.get_parameter("pulse_rate").value
    robot_name = node.get_parameter("robot_name").value if node.has_parameter("robot_name") else "sam0"

    agent_waraps_dict = {
            "agent-type": agent_type,
            "agent-uuid": None, # there is a callback in the WaraPSTaskHandler that will read this from the lvl1 WaraPSVehicle
            "levels": levels,
            "name": robot_name,
            "pulse_rate": pulse_rate,
        }
    
    # declare the parameter for printing bt (mode)
    node.declare_parameter("bt_log_mode", "verbose") # can be "verbose" or "compact"
    bt_log_mode = node.get_parameter("bt_log_mode").value

    
    # start_offset = 5.0
    node.declare_parameter("bt_launch_delay", 5.0) # seconds
    bt_launch_delay = node.get_parameter("bt_launch_delay").value

    # get the BT timeout ros parameter
    node.declare_parameter("bt_health_timeout", 15.0) # seconds
    bt_health_timeout = node.get_parameter("bt_health_timeout").value

    # timeout for considering action-server heartbeats stale in task discovery
    node.declare_parameter("task_liveliness_timeout", 10.0) # seconds
    task_liveliness_timeout = node.get_parameter("task_liveliness_timeout").value


    wara_ps_task_handler = WaraPSTaskHandler(
        node,
        agent_waraps_dict,
        start_offset=bt_launch_delay,
        task_liveliness_timeout=task_liveliness_timeout,
    )
    bt = BT(vehicle_container = agent,
            task_handler    = wara_ps_task_handler,
            get_ready_action = get_ready_action_client,
            emergency_action = emergency_action_client,
            # action_client_list = action_client_list,
            # the commented out line above means that you're listening for available tasks from the WaraPSTaskHandler. You can also provide a list of action clients to use if you like.
            now_seconds_func  = ros_seconds_float,
            bt_health_timeout        = bt_health_timeout
            )
    # bt.setup()
    need_bt_setup = False
    is_bt_setup = False

    bt_tip = None
    old_bt_tip = None

    bt_str = ""
    def print_bt(mode: str = "verbose"): # can be "verbose" or "compact"
        nonlocal bt, bt_str, node, action_client_list, agent, wara_ps_task_handler, bt_tip, old_bt_tip

        if mode == "verbose":
            new_str = pt.display.ascii_tree(bt._bt.root, show_status=True)
            if new_str != bt_str:
                s = f"\nBT::\n{new_str}\n"
                s+= f"WARA PS Task Handler::\n{wara_ps_task_handler}\n"
                node.get_logger().info(s)
                bt_str = new_str
            return
        elif mode == "compact":
            # print a compact version of the BT
            # log that you're here
            # node.get_logger().info("Printing compact BT...")
            new_str = pt.display.ascii_tree(bt._bt.root, show_status=True)
            if new_str != bt_str:

                new_tip = bt._bt.root.tip()
                if  old_bt_tip is None or new_tip!= old_bt_tip:
                    old_bt_tip = new_tip
                    s = f"\nBT::\n{new_str}\n"
                    bt_str = new_str

                    s+= f"WARA PS Task Handler::\n{wara_ps_task_handler}\n"
                    node.get_logger().info(s)
            return

            

    def update():
        nonlocal bt, is_bt_setup, need_bt_setup

        if not is_bt_setup and need_bt_setup:
            bt.setup()
            is_bt_setup = True
            need_bt_setup = False

        if is_bt_setup:
            bt.tick()
            print_bt(mode=bt_log_mode)
    
    def check_tree_updates():
        """Periodically check if available tasks have changed (separate from tick for efficiency)"""
        nonlocal bt, is_bt_setup
        if is_bt_setup:
            # The actual update happens in tick(), this is just for logging purposes
            # or you could call bt._update_task_handler_tree() here if you want less frequent checks
            pass
        
    node.create_timer(0.1, update)
    node.create_timer(5.0, check_tree_updates)  # Optional: Add explicit periodic check
    # node.create_timer(0.5, print_bt)

    def publish_bt_tip():
        nonlocal bt, node, wara_ps_task_handler

        # publish the BT tip to the WaraPS task handler
        if is_bt_setup:
            bt_tip = bt._bt.root.tip()
            if bt_tip is not None:
                # parse the tip to a string
                tip_str = f"{bt_tip.name} ({bt_tip.status})"
                # publish the tip to the WaraPS task handler
                wara_ps_task_handler.publish_bt_tip(tip_str)
        else:
            node.get_logger().info("BT is not setup yet, cannot publish tip.")
    
    node.create_timer(1, publish_bt_tip) 
    
    def pub_bt_status():
        nonlocal bt_status_pub, bt, node, is_bt_setup
        
        if not is_bt_setup:
            # node.get_logger().warn("BT is not setup yet, cannot publish status.")
            return
        # publish the BT status to the BT_STATUS_TOPIC
        bt_status_pub.publish(String(data=pt.display.ascii_tree(bt._bt.root, show_status=True)))

    # create a timer to publish the BT status to BT_STATUS_TOPIC
    status_str_timer = node.create_timer(0.1,pub_bt_status)

    start_time = None

    def wara_ps_lvl_2_comms():
        nonlocal wara_ps_task_handler, need_bt_setup, start_time, node

        # get the current time
        now_time = ros_seconds_float()
        if start_time is None:
            start_time = now_time
        # task execution info
        wara_ps_task_handler.lvl_2_heartbeat(now_time)
        # tst execution info
        wara_ps_task_handler.lvl_3_heartbeat(now_time)

        if not is_bt_setup:
            # if the BT is not ticking, we can start it
            if now_time - start_time > bt_launch_delay: # give 5 seconds for living action servers to provide a heartbeat to the WaraPSTaskHandler object
                need_bt_setup = True
            else:
                node.get_logger().info(f"Launching WaraPS BT in {bt_launch_delay - (now_time - start_time):.2f} seconds...")

    node.create_timer(1.0/wara_ps_task_handler.wara_ps_dict["pulse_rate"], wara_ps_lvl_2_comms)

    rclpy.spin(node)