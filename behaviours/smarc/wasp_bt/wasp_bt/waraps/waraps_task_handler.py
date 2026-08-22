from typing import Type
from rclpy.node import Node
from std_msgs.msg import String, Int8, Empty
from smarc_msgs.msg import Topics
from wasp_bt.vehicles.sensor import Sensor, SensorNames
import json
from copy import deepcopy
import enum
from std_srvs.srv import Trigger

# TODO: move this to a common place
class WaraPSTaskStates(enum.Enum):
    """
    The states of the WARAPS task
    """
    STARTED = "started"
    RUNNING = "running"
    PAUSED = "paused"
    RESUMED = "resumed"
    ENOUGH = "enough"
    ABORTED = "aborted"
    ERROR = "error"
    FINISHED = "finished"


    def __str__(self):
        return self.name
    
class WaraPSCommandSignals(enum.Enum):
    """
    The signals that can be sent to the WaraPS task
    """
    ABORT = "$abort"
    ENOUGH = "$enough"
    PAUSE = "$pause"
    CONTINUE = "$continue"
    CANCEL_ABORT = "$cancel_abort" 

    def __str__(self):
        return self.name

class HasWaraPSTaskHandler:
    """
    This class is used to mark a class as having an MQTT interactor. This is used to make sure that the class has the methods that are needed for the MQTT interactor to work.
    """
    def __init__(self):
        self._task_handler = None
        self._wara_ps_dict = None
        self._robot_name = None

    @property
    def wara_ps_task_handler(self):
        """
        Returns the WaraPSTaskHandler object that is used to handle the MQTT interactor.
        """
        return self._wara_ps_task_handler

    @wara_ps_task_handler.setter
    def wara_ps_task_handler(self, value):
        """
        Sets the WaraPSTaskHandler object that is used to handle the MQTT interactor.
        """
        self._wara_ps_task_handler = value

    @property
    def wara_ps_dict(self):
        """
        Returns the WaraPS dictionary that is used to handle the MQTT interactor.
        """
        return self._wara_ps_dict
    
    @wara_ps_dict.setter
    def wara_ps_dict(self, value):
        """
        Sets the WaraPS dictionary that is used to handle the MQTT interactor.
        """
        self._wara_ps_dict = value
        self._robot_name = value["name"] if value else None

#: WARA-PS tasks the BEHAVIOUR TREE provides itself, mapped to the action server whose
#: pipeline each one streams through.
#:
#: Every other available task appears because an ACTION SERVER published a heartbeat
#: (`_action_hb_callback`). `auv-farm-inspection` has no server of its own on purpose: it is
#: a sequence of ordinary `auv_depth_move_to` goals chosen by the farm planner, and giving it
#: a second server on that action name is precisely the defect that stopped every mission at
#: waypoint 1 for a week (SETTLED §1c, the duplicate bringup).
#:
#: It is therefore registered as available **exactly when its provider's heartbeat arrives**,
#: with the SAME `ros_name`, and it ages out through the same liveliness timeout. That is
#: consumer-side evidence, not a flag: if the diving controller's action server dies, the farm
#: task disappears from `tasks-available` on its own, because the thing it needs is gone.
BT_PROVIDED_TASKS = {
    "auv-farm-inspection": "auv-depth-move-to",
}


class WaraPSTaskHandler:
    def __init__(
        self,
        node:Node,
        wara_ps_dict:Type[dict],
        start_offset:float=5.0,
        task_liveliness_timeout: float = 10.0,
    ):
        """
        A class to handle the parts of the BT that need to interact with MQTT. This will later double up as the Mission Command and Updator.

        It is the job of this interactor to listen and publish to the relevant ROS topics connected to the MQTT bridge, and handle WARA-PS actions.
        """

        # private: only this class should access this
        self._node = node

        # public: outsiders can access this        
        self._wara_ps_dict = wara_ps_dict
        self._robot_name = wara_ps_dict["name"]
        self.start_offset = start_offset
        self._task_liveliness_timeout = max(1.0, float(task_liveliness_timeout))

        self.tasks_available = []
        self.past_tasks = []
        self.tasks_executing = []
        # Mission-clock progress state (see _mission_progress / mission_timer_state). All None
        # or 0 until a start-tst is accepted: nothing here may make a display claim a mission
        # exists, and an absent estimate must render as "--" rather than as a number.
        self._mission_wp_total = 0
        self._mission_past_base = 0
        self._mission_wp_done_seen = 0
        self._mission_last_wp_at = None
        # Flat, and all-None until something has actually flown: a display must be able to say
        # "this vehicle has never flown" rather than draw an empty run as a finished one.
        self._last_mission_summary = {"last_elapsed_s": None, "last_limit_s": None,
                                      "last_wp_done": None, "last_wp_total": None}

        self.aborted_flag = False
        self.emergency_flag = False
        # Who aborted, and why, for whoever asks after the fact. None until something does --
        # never a placeholder string, because an unrecorded cause must report as unrecorded and
        # never be guessed (Data Cube spec invariant 4b).
        self.last_abort_origin = None
        self.last_abort_detail = None
        self.health_status = Topics.VEHICLE_HEALTH_ERROR
        
        self.health_last_time = None

        self.mission_start_time = None
        self.mission_timeout = None

        self.mission_status = None

        self.mission_command = None

        
        # Publishers for Level 2 WARA-PS topics
        self._wara_ps_direct_execution_info_pub = node.create_publisher(String, Topics.
        WARA_PS_DIRECT_EXECUTION_INFO_TOPIC, 10)

        # Publishers for Level 1 WARA-PS topic: executing_tasks
        self._wara_ps_task_list_pub = node.create_publisher(String, Topics.WARA_PS_SENSOR_EXECUTING_TASKS_TOPIC, 10)

        self._wara_ps_exec_response_pub = node.create_publisher(String, Topics.WARA_PS_EXEC_RESPONSE_TOPIC, 10)
        self._wara_ps_exec_feedback_pub = node.create_publisher(String, Topics.WARA_PS_EXEC_FEEDBACK_TOPIC, 10)


        # Publishers for Level 3 WARA-PS topics
        self._wara_ps_tst_exec_info_pub = node.create_publisher(String, Topics.WARA_PS_TST_EXEC_INFO_TOPIC, 10)

        self._wara_ps_tst_response_pub = node.create_publisher(String, Topics.WARA_PS_TST_RESPONSE_TOPIC, 10)
        self._wara_ps_tst_feedback_pub = node.create_publisher(String, Topics.WARA_PS_TST_FEEDBACK_TOPIC, 10)

        # publishers for bt head
        self._wasp_bt_tip_pub = node.create_publisher(String, Topics.WARA_PS_SENSOR_BT_TOPIC, 10)

        # THE MISSION CLOCK, PUBLISHED (Ivan, 2026-08-18).
        #
        # `mission_start_time` and `mission_timeout` decide whether a mission lives or dies, and
        # until today they existed ONLY as attributes on this object. Nothing outside could see
        # them, so a 130 m plan was aborted at 301 s eight times over several weeks and read as
        # a vehicle fault every time. Ivan, immediately after the first mission that survived:
        # "for next round of HUD it would be useful with timer, total mission passed, countdown
        # to end". A limit that can end a mission has to be observable while the mission runs.
        #
        # RELATIVE topic on purpose: the node already carries the robot namespace, so this
        # resolves to /<robot>/ctrl/mission_timer and a second vehicle gets its own without any
        # string surgery. Deliberately a plain diagnostic channel rather than a WARA-PS one, so
        # MC, the Unity HUD and a bag can all read it without knowing the WARA-PS envelope.
        # JSON in a String for the same reason: a new .msg would need a rebuild in every
        # consumer before anything could read it.
        self._mission_timer_pub = node.create_publisher(String, "ctrl/mission_timer", 10)


        # Subscriptions for WARA-PS command topics
        self._wara_ps_exec_command_sub = node.create_subscription(String, Topics.WARA_PS_EXEC_COMMAND_TOPIC, self._exec_command_cb, 10)

        self._wara_ps_tst_command_sub = node.create_subscription(String, Topics.WARA_PS_TST_COMMAND_TOPIC, self._tst_command_cb, 10)

        # Subscriptions to action Server topics
        self._wara_ps_action_server_sub = node.create_subscription(String, Topics.WARA_PS_ACTION_SERVER_HB_TOPIC, self._action_hb_callback, 10)

        # Subscriptions for WARA-PS heartbeat topics
        self._level_1_heartbeat_sub = node.create_subscription(String, Topics.WARA_PS_HEARTBEAT_TOPIC, self._read_level_1_heartbeat_cb, 1)

        # subscribe to ABORT topic
        self._wara_ps_abort_sub = node.create_subscription(String, Topics.WARA_PS_ABORT_TOPIC, self._bigredbutton_cb, 10)

        # subscribe to SMARC-wide abort topic
        self._smarc_abort_sub = node.create_subscription(Empty, Topics.ABORT_TOPIC, self._emptybigredbutton_cb, 10)

        # subscribe to smarc health topic
        self._vehicle_health_sub = node.create_subscription(Int8, Topics.VEHICLE_HEALTH_TOPIC, self._vehicle_health_cb, 10)


        if "direct_execution" in self._wara_ps_dict["levels"]:
            self._direct_execution_info_data = {
                "name": self._wara_ps_dict["name"],
                "rate": self._wara_ps_dict["pulse_rate"],
                "type": "DirectExecutionInfo",
                "stamp": "",
                # "tasks-available": self._wara_ps_dict["tasks-available"],
                "tasks-available": [], # empty list, read from relevant topic in callback for action server subscriptions
                "tasks-executing": self.tasks_executing,
            }

        # Add the reset emergency service
        self._reset_emergency_srv = self._node.create_service(
            Trigger,
            "reset_emergency",
            self._reset_emergency_cb
        )

    # read only task_handler.wara_ps_dict
    @property
    def wara_ps_dict(self):
        """
        Returns the WaraPS dictionary that is used to handle the MQTT interactor.
        """
        return self._wara_ps_dict
    
    
    def lvl_2_heartbeat(self, now_time):
        """
        This method is called to publish the level 2 heartbeat.
        """
        # find now_time from the stamp in the heartbeat data
        self._direct_execution_info_data["stamp"] = now_time


        # naming convention change
        list_of_running_tasks = deepcopy(self.tasks_executing)

        # for every dict in this list, rename the key "name" to "task-name"
        for i in range(len(list_of_running_tasks)):
            list_of_running_tasks[i]["task-name"] = list_of_running_tasks[i]["task"]["name"]
            # remove "task" param from dict
            list_of_running_tasks[i].pop("task", None)
            # remove "status" param from dict
            # list_of_running_tasks[i].pop("status", None)

        # update tasks executing
        self._direct_execution_info_data["tasks-executing"] = list_of_running_tasks

        # drop tasks that have not been seen for a while
        popped_indices = []
        for i in range(len(self.tasks_available)):
            # self._node.get_logger().info(f"Checking task {i} with name {self.tasks_available[i]['name']}")
            task = self.tasks_available[i]
            # log (now_time - task["last_seen"])
            if float(now_time - task["last_seen"]) > self._task_liveliness_timeout:
                # remove the task from the list of available tasks
                popped_indices.append(i)
                self._node.get_logger().info(f"Removed task {task['name']} from available at time {now_time}, last seen at {task['last_seen']}")

        # remove the tasks from the list of available tasks
        for i in reversed(popped_indices):
            self.tasks_available.pop(i)


        self._direct_execution_info_data["tasks-available"] = self.tasks_available


        # publish the heartbeat data
        msg = String()
        msg.data = json.dumps(self._direct_execution_info_data)
        self._wara_ps_direct_execution_info_pub.publish(msg)
        # self._node.get_logger().info('Published Direct Execution Info message')

        # publish executing tasks
        msg = String()
        msg.data = json.dumps(self.tasks_executing)
        self._wara_ps_task_list_pub.publish(msg)
        
        return True    
    
    def lvl_3_heartbeat(self, now_time):
        """
        This method is called to publish the level 3 heartbeat.
        It is used to update the WaraPS dictionary with the latest data.
        """
        # find now_time from the stamp in the heartbeat data
        self._direct_execution_info_data["stamp"] = now_time
        self._direct_execution_info_data["type"] = "TSTExecutionInfo"

        # publish the heartbeat data
        msg = String()
        msg.data = json.dumps(self._direct_execution_info_data)
        self._wara_ps_tst_exec_info_pub.publish(msg)
        # self._node.get_logger().info('Published TST Execution Info message')

        # Published EVERY tick, including when no mission is running: a consumer must be able to
        # tell "no mission" from "this node has stopped talking". Absent is not empty.
        self._publish_mission_timer()

        # A FINISHED MISSION STOPS BEING TIMED, WHATEVER ELSE IS TRUE (2026-08-19, measured).
        #
        # This used to live inside the guard below, and mission #37's bag is what that cost:
        # `ctrl/mission_timer` read `elapsed_s 8001.4, limit_s 2995.0, fraction 2.668, state:
        # running` while the vehicle sat surfaced and idle -- 6,856 s after the task queue
        # emptied. The reasoning that finds it is entirely in that one line: `state: running`
        # with a non-None `limit_s` means the first TWO clauses of the guard were satisfied, so
        # only the third can have been false, so `emergency_flag` was True. And it was: at
        # +1145.1, the instant the vehicle surfaced to end the mission, `smarc/vehicle_health`
        # went 0 -> 2 with `Fault detected: low altitude! Current altitude: 0.28, Min altitude:
        # 0.5` -- the surfaced vehicle reading its own depth as bottom clearance, which is the
        # 2026-08-18 "altitude meant two things" defect arriving through the health checker
        # instead of through min_altitude. The flag latches (only _reset_emergency_cb clears it),
        # so from that moment the timer could neither fire NOR STOP.
        #
        # RETIREMENT IS NOT ENFORCEMENT and must not share its guard. Whether an already-aborted
        # mission still needs a timeout is arguable -- it is the guard below, and it is Ivan's
        # call (a mission that ended is not a mission that needs a second abort). Whether a
        # mission that has ENDED should keep being timed is not arguable at all.
        if self.mission_start_time is not None and self.tasks_executing == []:
            self.mission_start_time = None
            self.mission_timeout = None

        # ABORT IF MISSION TIMOUT HAS BEEN EXCEEDED
        # do this only if there is a mission running
        if self.mission_start_time is not None and self.mission_timeout is not None and self.emergency_flag is False:
            elapsed = self.current_time() - self.mission_start_time
            if elapsed > self.mission_timeout:
                # THROUGH _apply_abort, NOT AROUND IT (2026-08-18). This used to set
                # `emergency_flag = True` inline, which aborted the mission correctly and told
                # nobody why: last_abort_origin stayed empty, so MC's readiness row read
                # "cause not recorded" and the operator was left guessing at a vehicle fault
                # when the real answer was "the plan needed more time than it was given".
                # The detail carries both numbers so the log says how badly, not just that.
                self._apply_abort(
                    self.ABORT_ORIGIN_MISSION_TIMEOUT,
                    f"the mission ran out of time: {elapsed:.0f} s elapsed against a "
                    f"{self.mission_timeout:.0f} s limit. The plan may simply need longer than "
                    f"it was allowed -- check the mission timeout before blaming the vehicle",
                    respond=False)
                # The WARA-PS feedback stays, unchanged in shape: no abort path may get quieter.
                abort_msg = {
                    "agent-uuid": self._wara_ps_dict["agent-uuid"],
                    "com-uuid": "",
                    "response": "mission timeout exceeded",
                    "response-to": ""
                }
                msg = String()
                msg.data = json.dumps(abort_msg)
                self._wara_ps_tst_feedback_pub.publish(msg)
                # Retire the timers. A timeout is an EVENT, not a state: having fired once, it must
                # not stay armed. Leaving them set meant the deadline was still expired after
                # reset_emergency cleared the flag, so the very next tick raised it again -- and
                # since the guard above skips this whole block while the flag is up, nothing could
                # ever clear them. One timed-out mission disabled the vehicle permanently, and the
                # service written to recover from exactly that could not (2026-08-06).
                self.mission_start_time = None
                self.mission_timeout = None
                return False
            # (The "no tasks are executing -> retire the timers" branch used to be here. It moved
            #  ABOVE this guard on 2026-08-19 -- see the note there. Left as a comment rather than
            #  as an `if` that can no longer be reached: by the time control gets this far, the
            #  earlier retirement has already run this tick, so a copy here would be dead code
            #  that still reads like the thing doing the work.)

        return True
    
    def _read_level_1_heartbeat_cb(self, data: String):
        """
        This method is called to read the level 1 heartbeat.
        It is used to update the WaraPS dictionary with the latest data.
        """
        # parse the command
        if self._wara_ps_dict['agent-uuid'] is not None:
            return
        try:
            hb_data = json.loads(data.data)
        except (json.JSONDecodeError, TypeError) as e:
            self._node.get_logger().error(f"Failed to decode JSON from heartbeat data: {e}")
            return

        # a heartbeat without an agent-uuid is useless to us; ignore it
        if not isinstance(hb_data, dict) or hb_data.get("agent-uuid") is None:
            self._node.get_logger().warn("Received Level 1 heartbeat without an 'agent-uuid'; ignoring")
            return

        # update the WaraPS dictionary with the heartbeat data
        # log
        self._node.get_logger().info(f"Received Level 1 heartbeat. Copying agent-uuid: {hb_data['agent-uuid']}")
        self._wara_ps_dict["agent-uuid"] = hb_data["agent-uuid"]

        # unregister the heartbeat subscriber
        self._node.destroy_subscription(self._level_1_heartbeat_sub)

    def _action_hb_callback(self, data: String):
        # this function is called when a new action server heartbeat is received

        # get the current time
        now_time = self._node.get_clock().now().to_msg().sec + self._node.get_clock().now().to_msg().nanosec * 1e-9

        # parse the command
        action_name = data.data
        # self._node.get_logger().info(f"Received action server heartbeat: {action_name}")

        # action name is the name of the action server, a ros topic ish. We want to get rid of the namespacing and just hold on to the last part of the name. Further, we want to replace the "_" with "-" in this last part of the name
        parsed_action_name = action_name.split("/")[-1]
        parsed_action_name = parsed_action_name.replace("_", "-")

        #TODO: remove this hacky shit, unless on drone
        # parsed_action_name = "move-to"

        # if this action server is not already in the list of available tasks, add it
        if parsed_action_name not in [task["name"] for task in self.tasks_available] and "emergency" not in parsed_action_name: # don't want to make emergency action triggerable by user
            # add the action server to the list of available tasks
            task_dict = {
                "name": parsed_action_name,
                "signals": [
                    WaraPSCommandSignals.ABORT.value,
                    WaraPSCommandSignals.ENOUGH.value, 
                    WaraPSCommandSignals.PAUSE.value, 
                    WaraPSCommandSignals.CONTINUE.value
                ],
                "last_seen": now_time,
                "ros_name": action_name,
            }
            self.tasks_available.append(task_dict)

            # log last seen time
            self._node.get_logger().info(f"Found new action server: {action_name} at {now_time}")
            # NOTE (2026-08-17): this branch used to `return` here. The return was redundant —
            # the block below is the `else` of this `if` and could never run after it — and it
            # skipped the BT-provided-task refresh at the bottom on the FIRST heartbeat, which
            # is the one that matters when a stack has just come up.

        # if this action server is already in the list of available tasks, update the last seen time
        else:
            # update the last seen time
            for i in range(len(self.tasks_available)):
                task = self.tasks_available[i]
                if task["name"] == parsed_action_name:
                    # update the last seen time
                    self.tasks_available[i]["last_seen"] = now_time
                    break
            # log last seen time
            # self._node.get_logger().info(f"Updated action server: {action_name} at {now_time}")

        # Tasks the tree itself provides on top of this action server (see BT_PROVIDED_TASKS).
        # Refreshed on every heartbeat, so they live and die with their provider.
        self._refresh_bt_provided_tasks(parsed_action_name, action_name, now_time)

    def _refresh_bt_provided_tasks(self, provider_task_name, provider_ros_name, now_time):
        """Add/refresh any BT-provided task whose provider just reported in.

        The derived task carries the provider's OWN `ros_name`, which is what makes
        `ros_bt` hand it the cached action client rather than build a second one.
        """
        for name, provider in BT_PROVIDED_TASKS.items():
            if provider != provider_task_name:
                continue
            for task in self.tasks_available:
                if task["name"] == name:
                    task["last_seen"] = now_time
                    break
            else:
                self.tasks_available.append({
                    "name": name,
                    "signals": [
                        WaraPSCommandSignals.ABORT.value,
                        WaraPSCommandSignals.ENOUGH.value,
                        WaraPSCommandSignals.PAUSE.value,
                        WaraPSCommandSignals.CONTINUE.value,
                    ],
                    "last_seen": now_time,
                    "ros_name": provider_ros_name,
                    # Marked, so nothing downstream mistakes it for a server that exists.
                    "provided_by": "behaviour_tree",
                })
                self._node.get_logger().info(
                    f"Behaviour-tree task '{name}' is available: its provider "
                    f"'{provider}' reported in on {provider_ros_name}")


    def _send_exec_response(self, com_uuid, response):
        """
        Publishes a simple response on the exec response topic. Centralising the
        response shape keeps command handlers from crashing on missing keys.
        """
        response_msg = {
            "agent-uuid": self._wara_ps_dict["agent-uuid"],
            "com-uuid": com_uuid,
            "response": response,
            "response-to": com_uuid,
        }
        msg = String()
        msg.data = json.dumps(response_msg)
        self._wara_ps_exec_response_pub.publish(msg)

    def _send_tst_response(self, com_uuid, response):
        """
        Publishes a simple response on the TST response topic. Centralising the
        response shape keeps command handlers from crashing on missing keys.
        """
        response_msg = {
            "agent-uuid": self._wara_ps_dict["agent-uuid"],
            "com-uuid": com_uuid,
            "response": response,
            "response-to": com_uuid,
        }
        msg = String()
        msg.data = json.dumps(response_msg)
        self._wara_ps_tst_response_pub.publish(msg)

    def _exec_command_cb(self, data: String):
        # this function is called when a new command is received from the MQTT broker
        # parse the command
        try:
            command = json.loads(data.data)
        except (json.JSONDecodeError, TypeError) as e:
            self._node.get_logger().error(f"The received command is not a valid JSON: {e}")
            return

        # commands must be JSON objects, otherwise we cannot index into them
        if not isinstance(command, dict):
            self._node.get_logger().error("Invalid command: expected a JSON object")
            return

        self._node.get_logger().info(f"Received command: {command}")

        # a command without a 'command' key is meaningless; bail out gracefully
        if command.get("command") is None:
            self._node.get_logger().error("Invalid command: missing 'command' key")
            return

        # Dispatch under a safety net so a single malformed command can never crash the node
        try:
            self._handle_exec_command(command)
        except Exception as e:
            self._node.get_logger().error(f"Error while handling exec command '{command.get('command')}': {e}")
        return

    def _handle_exec_command(self, command: dict):
        command_type = command["command"]
        com_uuid = command.get("com-uuid", "")

        # Refuse starts or signals if emergency flag is up
        if (self.emergency_flag) and command_type in ["start-task"]:
            self._send_exec_response(com_uuid, "rejected: emergency flag is up")
            self._node.get_logger().warn("Rejected start command due to emergency flag.")
            return

        # refuse start or signal if health status is not ok
        if (self.health_status != Topics.VEHICLE_HEALTH_READY) and command_type in ["start-task"]:
            self._send_exec_response(com_uuid, "rejected: vehicle health status is not ok")
            self._node.get_logger().warn(f"Rejected start command due to vehicle health status: {self.health_status}.")
            return

        # handle ping command
        if command_type == "ping":
            self._send_exec_response(com_uuid, "pong")
            self._node.get_logger().info('Published Ping response message')

        # handle signal-task command
        elif command_type == "signal-task":
            # check if the command is valid
            if "task-uuid" not in command:
                self._node.get_logger().error("Invalid signal-task command: missing 'task-uuid' key")
                self._send_exec_response(com_uuid, "task not found")
                return

            signal = command.get("signal")
            status_msg = "task not found"

            if command["task-uuid"] not in [task["task-uuid"] for task in self.tasks_executing]:
                self._node.get_logger().error("Invalid signal-task command: task not found in executing tasks")
                status_msg = "task not in current tasks"

            else: # if the task is found in executing tasks
                status_msg = "ok"
                # what is the signal asking for? options: enough, pause, continue, abort

                if signal == WaraPSCommandSignals.ABORT.value:
                    # abort the task
                    for task in self.tasks_executing:
                        if task["task-uuid"] == command["task-uuid"]:
                            task["status"] = WaraPSTaskStates.ABORTED.value
                            break
                elif signal == WaraPSCommandSignals.ENOUGH.value:
                    # enough of the task
                    for task in self.tasks_executing:
                        if task["task-uuid"] == command["task-uuid"]:
                            task["status"] = WaraPSTaskStates.ENOUGH.value
                            break
                elif signal == WaraPSCommandSignals.PAUSE.value:
                    # pause the task
                    for task in self.tasks_executing:
                        if task["task-uuid"] == command["task-uuid"]:
                            task["status"] = WaraPSTaskStates.PAUSED.value
                            break
                elif signal == WaraPSCommandSignals.CONTINUE.value:
                    # continue the task
                    for task in self.tasks_executing:
                        if task["task-uuid"] == command["task-uuid"] and task["status"] == WaraPSTaskStates.PAUSED.value:
                            task["status"] = WaraPSTaskStates.RESUMED.value
                            break

            valid_signals = [s.value for s in WaraPSCommandSignals]
            if signal not in valid_signals:
                self._node.get_logger().error("Invalid signal-task command: invalid signal")
                status_msg = "invalid signal"

            if signal in [WaraPSCommandSignals.ABORT.value, WaraPSCommandSignals.ENOUGH.value]:
                # remove the task from the executing tasks list
                for i in range(len(self.tasks_executing)):
                    task = self.tasks_executing[i]
                    if task["task-uuid"] == command["task-uuid"]:
                        self.past_tasks.append(task)
                        self.tasks_executing.pop(i)
                        self.aborted_flag = True
                        break

            self._send_exec_response(com_uuid, status_msg)
            self._node.get_logger().info('Published Signal Task response message')

        # handle query-task command
        elif command_type == "query-task":
            # check if the command is valid
            if "task-uuid" not in command:
                self._node.get_logger().error("Invalid query-task command: missing 'task-uuid' key")
                self._send_exec_response(com_uuid, "task not found")
                return
            
            # check if the task is valid
            status_msg = "task not found"
            
            for task in self.tasks_executing:
                if task["task-uuid"] == command["task-uuid"]:
                    status_msg = task["status"]
                    break
            
            self._send_exec_response(com_uuid, status_msg)
            self._node.get_logger().info('Published Query Task response message')

        # handle start-task command
        elif command_type == "start-task":
            # check that the task is present and well-formed
            task = command.get("task")
            if not isinstance(task, dict) or "name" not in task:
                self._node.get_logger().error("Invalid start-task command: missing or malformed 'task'")
                self._send_exec_response(com_uuid, "task not found")
                return

            if "task-uuid" not in command:
                self._node.get_logger().error("Invalid start-task command: missing 'task-uuid' key")
                self._send_exec_response(com_uuid, "task not found")
                return

            task_uuid = command["task-uuid"]

            # check if the task is available
            if task["name"] not in [t["name"] for t in self.tasks_available]:
                if task["name"] != "custom-task":
                    # Not a recognised task and not a custom task: reject
                    self._node.get_logger().error("Invalid start-task command: task not available")
                    self._send_exec_response(com_uuid, "task not available")
                    return

                # Custom task handling: a "custom-task" carries the real action
                # name inside params["action-name"]. Resolve it so the task can
                # be matched against the available tasks.
                self._node.get_logger().info("WARNING: Custom task started.")
                try:
                    task["name"] = task["params"]["action-name"]
                except Exception as e:
                    self._node.get_logger().error(f"Failed to extract action name from custom task params: {e}")
                    self._send_exec_response(com_uuid, "task not available")
                    return

                # check if the resolved action name is available
                if task["name"] not in [t["name"] for t in self.tasks_available]:
                    self._node.get_logger().error("Invalid start-task command: custom task action name not available")
                    self._send_exec_response(com_uuid, "task not available")
                    return

                self._node.get_logger().info(f"Starting custom task: {task['name']}")

            # the task name now resolves to an available task
            if any(t["task-uuid"] == task_uuid for t in self.tasks_executing):
                self._node.get_logger().error("Invalid start-task command: task already executing")
                self._send_exec_response(com_uuid, "task already executing")
                return

            # add the task to the executing tasks list
            task_dict = {
                "task-uuid": task_uuid,
                "task": task,
                "status": WaraPSTaskStates.STARTED.value,
                "description": task.get("description", ""),
            }
            self.tasks_executing.append(task_dict)

            # publish the feedback
            feedback_msg = {
                "agent-uuid": self._wara_ps_dict["agent-uuid"],
                "com-uuid": com_uuid,
                "task-uuid": task_uuid,
                "task": task,
                "status": WaraPSTaskStates.STARTED.value,
            }
            msg = String()
            msg.data = json.dumps(feedback_msg)
            self._wara_ps_exec_response_pub.publish(msg)
            self._node.get_logger().info('Published Start Task response message')

        return
    
    def _tst_command_cb(self, data: String):
        # This function is called when a new TST command is received from the MQTT broker
        try:
            command = json.loads(data.data)
        except (json.JSONDecodeError, TypeError) as e:
            self._node.get_logger().error(f"The received TST command is not a valid JSON: {e}")
            return

        # commands must be JSON objects, otherwise we cannot index into them
        if not isinstance(command, dict):
            self._node.get_logger().error("Invalid TST command: expected a JSON object")
            return

        self._node.get_logger().info(f"Received TST command: {command}")

        # a command without a 'command' key is meaningless; bail out gracefully
        if command.get("command") is None:
            self._node.get_logger().error("Invalid TST command: missing 'command' key")
            return

        # Dispatch under a safety net so a single malformed command can never crash the node
        try:
            self._handle_tst_command(command)
        except Exception as e:
            self._node.get_logger().error(f"Error while handling TST command '{command.get('command')}': {e}")
        return

    def _handle_tst_command(self, command: dict):
        command_type = command["command"]
        com_uuid = command.get("com-uuid", "")

        # Refuse starts or signals if emergency flag is up
        if self.emergency_flag and command_type in ["start-tst"]:
            self._send_tst_response(com_uuid, "rejected: emergency flag is up")
            self._node.get_logger().warn("Rejected start TST command due to emergency flag.")
            return

        # Refuse starts or signals if health status is not ok
        if (self.health_status != Topics.VEHICLE_HEALTH_READY) and command_type in ["start-tst"]:
            self._send_tst_response(com_uuid, "rejected: vehicle health status is not ok")
            self._node.get_logger().warn(f"Rejected start TST command due to vehicle health status: {self.health_status}.")
            return

        # handle signal-unit command
        if command_type == "signal-unit":
            if "unit" not in command:
                self._node.get_logger().error("Invalid signal-unit command: missing 'unit' key")
                return

            signal = command.get("signal")
            status_msg = "ok"
            if signal == WaraPSCommandSignals.ABORT.value:
                for task in self.tasks_executing:
                    task["status"] = WaraPSTaskStates.ABORTED.value
            elif signal == WaraPSCommandSignals.ENOUGH.value:
                for task in self.tasks_executing:
                    task["status"] = WaraPSTaskStates.ENOUGH.value
            elif signal == WaraPSCommandSignals.PAUSE.value:
                for task in self.tasks_executing:
                    task["status"] = WaraPSTaskStates.PAUSED.value
            elif signal == WaraPSCommandSignals.CONTINUE.value:
                for task in self.tasks_executing:
                    task["status"] = WaraPSTaskStates.RESUMED.value
            elif signal == WaraPSCommandSignals.CANCEL_ABORT.value:
                self.emergency_flag = False
                self.mission_start_time = None
                self.mission_timeout = None

            valid_signals = [s.value for s in WaraPSCommandSignals]
            if signal not in valid_signals:
                self._node.get_logger().error("Invalid signal-tst command: invalid signal")
                status_msg = "invalid signal"

            if signal in [WaraPSCommandSignals.ABORT.value, WaraPSCommandSignals.ENOUGH.value]:
                for i in range(len(self.tasks_executing)):
                    task = self.tasks_executing[0]
                    self.past_tasks.append(task)
                    self.tasks_executing.pop(0)
                
                # raise aborted flag
                self.aborted_flag = True

                if signal == WaraPSCommandSignals.ABORT.value:
                    self.emergency_flag = True

            self._send_tst_response(com_uuid, status_msg)
            self._node.get_logger().info('Published TST Signal Task response message')

        elif command_type == "start-tst": 
            # '''
            # {"receiver":"shekharu_lolo","tst":{"common-params":{"execunit":"/shekharu_lolo","node-uuid":"e5bcb11a-2c8f-48cc-94c1-747c88ab516e"},"params":{},"children":[{"description":"1","task-uuid":"03acd059-73d2-412f-8d75-f3fd2b9efac0","params":{"waypoint":{"latitude":58.850523629300554,"longitude":17.674904712183004,"target_depth":10.0,"min_altitude":5.0,"rpm":1000.0,"timeout":1000.0}},"name":"auv-depth-move-to"},{"description":"2","task-uuid":"c83ff631-8b63-4c69-a260-9008782ee41a","params":{"waypoint":{"latitude":58.850628267523796,"longitude":17.675200365495684,"target_depth":15.0,"min_altitude":5.0,"rpm":1000.0,"timeout":1000.0}},"name":"auv-depth-move-to"}],"tst-uuid":"0536c8e2-0d23-45e0-9434-eed663b14ec0","description":"Lolo Test","name":"seq"},"command":"start-tst","com-uuid":"fe38f852-7ff4-4f4a-bd78-62011e0fca00","sender":"UnityGUI"}
            # '''
            # check if the command is valid
            tst = command.get("tst")
            if not isinstance(tst, dict):
                self._node.get_logger().error("Invalid start-tst command: missing or malformed 'tst' key")
                self._send_tst_response(com_uuid, "task not found")
                return

            # set mission command
            self.mission_command = command

            # extract the mission timeout from "params" key in tst
            params = tst.get("params")
            if isinstance(params, dict) and "timeout" in params:
                self.mission_timeout = params["timeout"]
            else:
                self.mission_timeout = 1800 # default mission timeout
                self._node.get_logger().info(f"No timeout provided. Mission timeout set to {self.mission_timeout} seconds")

            # try to cast the timeout to a float, if it fails log an error and reject the command
            try:
                self.mission_timeout = float(self.mission_timeout)
            except (ValueError, TypeError) as e:
                self._node.get_logger().error(f"Invalid mission timeout value: {e}")
                self._send_tst_response(com_uuid, "Rejected: Mission timeout value should be a float representing seconds")
                return

            # extract the list of tasks from the command. They're the children of the tst key
            tasks = tst.get("children")
            if not isinstance(tasks, list):
                self._node.get_logger().error("Invalid start-tst command: missing or malformed 'children' key in 'tst'")
                self._send_tst_response(com_uuid, "task not found")
                return

            common_params = tst["common-params"] if isinstance(tst.get("common-params"), dict) else {}

            # inject common params into each tasks params
            tasks_to_start = []
            for task in tasks:
                # each child must be a well-formed task object
                if not isinstance(task, dict) or "name" not in task or "task-uuid" not in task:
                    self._node.get_logger().error("Invalid start-tst command: malformed task in 'children'")
                    self._send_tst_response(com_uuid, "Rejected: malformed task in mission")
                    return

                if not isinstance(task.get("params"), dict):
                    task["params"] = {}
                # merge common params into task params
                task["params"].update(common_params)

                # Custom task handling: a "custom-task" carries the real action
                # name inside params["action-name"]. Rewrite the task name so it
                # can be matched against the available tasks, mirroring the
                # start-task (single task) flow.
                if task["name"] == "custom-task":
                    self._node.get_logger().info("WARNING: Custom task started (TST).")
                    try:
                        task["name"] = task["params"]["action-name"]
                    except Exception as e:
                        self._node.get_logger().error(f"Failed to extract action name from custom task params: {e}")
                        self._send_tst_response(com_uuid, "Rejected: Custom task missing 'action-name' in params")
                        return

                # add the task to the executing tasks list
                task_dict = {
                    "task-uuid": task["task-uuid"],
                    "task": task,
                    "status": WaraPSTaskStates.STARTED.value,
                    "description": task.get("description", ""),
                }

                # check that the tasks are all available on the vehicle
                if task["name"] not in [t["name"] for t in self.tasks_available]:
                    self._node.get_logger().error(f"Invalid start-tst command: task {task['name']} not available")
                    self._send_tst_response(com_uuid, f"Rejected: Task {task['name']} not available")
                    return
                
                tasks_to_start.append(task_dict)
                
            # A start-tst REPLACES the mission. It does not add to it.
            #
            # This was `extend` onto whatever was already in tasks_executing, with no
            # clear anywhere on this path -- so a start-tst was appended behind the
            # previous mission's leftover legs and the vehicle flew those first. Measured
            # 2026-08-13: an earlier mission left legs 2/3/4 queued; the next mission was
            # appended behind them; the vehicle flew the OLD waypoints, which crossed land,
            # and drove into the dry-dock wall. From the operator's seat this reads as
            # "the navigation is wrong", which is the most expensive possible way to
            # present a queue that was never emptied.
            #
            # Cleared HERE, after every validation above has passed, so a rejected or
            # malformed start-tst cannot wipe a mission that is legitimately running --
            # each early return above leaves the queue untouched.
            if self.tasks_executing:
                self._node.get_logger().warn(
                    f"start-tst replaces {len(self.tasks_executing)} task(s) still queued "
                    f"from a previous mission: "
                    f"{[t.get('description') for t in self.tasks_executing]}")
                for task in self.tasks_executing:
                    task["status"] = WaraPSTaskStates.ABORTED.value
                    self.past_tasks.append(task)
                self.tasks_executing = []
            self.tasks_executing.extend(tasks_to_start)
            # start mission timer
            self.mission_start_time = self.current_time()
            # ...and the progress baseline for the mission clock. `past_tasks` accumulates
            # across missions, so "how many of THIS mission are done" is measured against where
            # the list stood when this one was accepted -- not against its length, which would
            # report the previous run's waypoints as already flown.
            self._mission_wp_total = len(tasks_to_start)
            self._mission_past_base = len(self.past_tasks)
            self._mission_wp_done_seen = 0
            self._mission_last_wp_at = self.mission_start_time

            # Publish acknowledgment that TST was accepted and queued
            self._send_tst_response(com_uuid, "accepted")
            self._node.get_logger().info(f"Published TST acceptance response for command {com_uuid}")

        return        
    
    def _vehicle_health_cb(self, data: Int8):
        """
        This method is called when a new vehicle health message is received.
        It is used to update the WaraPS dictionary with the latest data.
        """
        # log the time of the last health status update
        self.health_last_time = self.current_time()

        vehicle_health_status = data.data
        
        # update the health status
        if vehicle_health_status == Topics.VEHICLE_HEALTH_READY:
            self.health_status = Topics.VEHICLE_HEALTH_READY
            # self._node.get_logger().info("Vehicle health status: OK")
        elif vehicle_health_status == Topics.VEHICLE_HEALTH_WAITING:
            self.health_status = Topics.VEHICLE_HEALTH_WAITING
            # self._node.get_logger().warn("Vehicle health status: WARNING")
        elif vehicle_health_status == Topics.VEHICLE_HEALTH_ERROR:
            self.health_status = Topics.VEHICLE_HEALTH_ERROR
            # self._node.get_logger().error("Vehicle health status: ERROR")
        


    def clear_task_queue(self):
        """
        Clears the task queue.
        """
        self.tasks_executing = []

    def clear_current_task(self):
        """
        Clears the current task.
        """
        if len(self.tasks_executing) > 0:

            # change status of the current task to FINISHED
            self.tasks_executing[0]["status"] = WaraPSTaskStates.FINISHED.value

            self.tasks_executing.pop(0)
        else:
            # log
            self._node.get_logger().error("No tasks executing")
            return None
        
    def get_executing_tasks(self):
        """
        Returns the list of executing tasks.
        """
        return self.tasks_executing
    
    def get_current_task_params(self):
        """
        Returns the parameters of the current task.
        """
        if len(self.tasks_executing) > 0:
            return self.tasks_executing[0]["task"]["params"]
        else:
            # log
            self._node.get_logger().error("No tasks executing")
            return None
        
    def get_current_task_status(self):
        """
        Returns the status of the current task.
        """
        if len(self.tasks_executing) > 0:
            return self.tasks_executing[0]["status"]
        else:
            # log
            self._node.get_logger().error("No tasks executing")
            return None
        
    def set_current_task_status(self, status):
        """
        Sets the status of the current task.
        """
        if len(self.tasks_executing) > 0:
            # Accept both enum and string for status
            if isinstance(status, WaraPSTaskStates):
                self.tasks_executing[0]["status"] = status.value
            else:
                self.tasks_executing[0]["status"] = status
        else:
            self._node.get_logger().error("No tasks executing")
            return None
    
    def set_mission_status(self, status: str):
        """
        Sets the status of the current mission.
        """
        # Set the instance variable that A_Chilling checks
        self.mission_status = status
        
        # Also set it in mission_command dict if it exists
        if self.mission_command is not None:
            self.mission_command["status"] = status
        else:
            self._node.get_logger().warn("No mission_command to set status for, but mission_status set anyway")

    def move_task_to_past(self):
        """
        Moves the current task to the past tasks list.
        """
        if len(self.tasks_executing) > 0:
            self.past_tasks.append(self.tasks_executing[0])
            self.tasks_executing.pop(0)
        else:
            # log
            self._node.get_logger().error("No tasks executing")
            return None
        
    def __str__(self):
        """
        Returns the string representation of the WaraPSTaskHandler object. Should be a table of the tasks available, executing and past tasks.
        """

        # create a string representation of the tasks available
        tasks_available_str = "Tasks Available:\n"
        for task in self.tasks_available:
            tasks_available_str += f"\t{task['name']}\n"

        # create a string representation of the tasks executing
        tasks_executing_str = "Tasks Executing:\n"
        for task in self.tasks_executing:
            tasks_executing_str += f"\t{task['task']['name']}\n"

        # create a string representation of the past tasks
        past_tasks_str = "Past Tasks:\n"
        for task in self.past_tasks:
            past_tasks_str += f"\t{task['task']['name']}\n"

        return f"{tasks_available_str}{tasks_executing_str}" #{past_tasks_str}"
    
    def publish_feedback_to_current_task(self, feedback: str):
        """
        Publishes feedback to the current task.
        """
        if len(self.tasks_executing) > 0:
            # create a feedback message
            feedback_msg = {
                "agent-uuid": self._wara_ps_dict["agent-uuid"],
                "task-uuid": self.tasks_executing[0]["task-uuid"],
                "feedback": feedback,
                "status": self.tasks_executing[0]["status"]
            }
            msg = String()
            msg.data = json.dumps(feedback_msg)
            self._wara_ps_exec_feedback_pub.publish(msg)
            # self._node.get_logger().info('Published Feedback message')
        else:
            # log
            # self._node.get_logger().error("No tasks executing")
            return None 
        
    def publish_feedback_to_tst(self, feedback: str):
        """
        Publishes feedback to the TST.
        """
        if len(self.tasks_executing) > 0:
            # create a feedback message
            feedback_msg = {
                "agent-uuid": self._wara_ps_dict["agent-uuid"],
                "tst-uuid": self.mission_command["tst"]["tst-uuid"],
                "task-uuid": self.tasks_executing[0]["task-uuid"],
                "feedback": feedback,
                "status": self.tasks_executing[0]["status"]
            }
            msg = String()
            msg.data = json.dumps(feedback_msg)
            self._wara_ps_tst_feedback_pub.publish(msg)
            # self._node.get_logger().info('Published TST Feedback message')
        else:
            # log
            # self._node.get_logger().error("No tasks executing")
            return None

    # Every abort names WHERE IT CAME FROM. Vocabulary, deliberately closed and short, because
    # these are the only things that can abort this vehicle and each implies a different next
    # move for the operator:
    #
    #   operator_c2      an abort arrived on waraps/abort -- a human or another C2, over MQTT
    #   vehicle_stack    an abort arrived on smarc/abort -- this hull's own stack (obstacle
    #                    detector, health checker, anything relaying core/abort)
    #   bt_health        THIS process's behaviour tree failed its own health checks and parked
    #                    itself. No message arrived from anywhere; the tree decided.
    #
    # 2026-08-15 (#29). Before this, A_Abort -- the tree's OWN health fallback -- called
    # _bigredbutton_cb() in-process with the literal "Big Red Button pressed", so a
    # vehicle-internal abort logged as "from MQTT/C2" and published a WARA-PS response saying it
    # was replying to a big red button nobody had pressed. The operator's own record then said a
    # human aborted a mission that the vehicle aborted on itself, which is the worst possible
    # direction for that error to point: it sends the next session looking for an operator
    # action instead of a health fault. Same family as SETTLED §1's "never put a guess in a
    # status string" -- here the guess was baked into a protocol message.
    ABORT_ORIGIN_OPERATOR_C2 = "operator_c2"
    ABORT_ORIGIN_VEHICLE_STACK = "vehicle_stack"
    ABORT_ORIGIN_BT_HEALTH = "bt_health"
    # The fourth origin, added 2026-08-18 after it cost days. The mission-timeout path set
    # `emergency_flag = True` inline and never went through _apply_abort, so it recorded no
    # origin at all -- Mission Control offered "Clear emergency (cause not recorded)" and the
    # operator had nothing to act on. `~/.ros/log` held EIGHT of these, each read as a fresh
    # mystery. An abort that cannot name itself is the defect this constant list exists for.
    ABORT_ORIGIN_MISSION_TIMEOUT = "mission_timeout"

    def _mission_progress(self, now: float, elapsed: float) -> dict:
        """Waypoints done / total, and pace-based estimates for the rest.

        PROGRESS IS OBSERVED, NOT INSTRUMENTED. This edge-detects `past_tasks` growing rather
        than hooking the task lifecycle: the lifecycle is the part that flies the vehicle, and a
        display has no business adding branches to it. If the list is not what this expects the
        estimates simply go None, which the HUD renders as "--".

        THE ESTIMATES ARE PACE, AND THEY SAY SO. Average seconds per completed waypoint,
        extrapolated. That is honest for a plan of similar legs and wrong for a plan whose last
        leg is ten times the first -- which is exactly why the HUD prefixes them with "~" and
        why the mission TIMEOUT is shown alongside rather than replaced by them. A hard limit
        and a guess must never be printed as if they were the same kind of number: today's
        entire 300 s hunt was one number being mistaken for another.
        """
        total = self._mission_wp_total
        if not total:
            return {"wp_total": None, "wp_done": None, "wp_current": None,
                    "eta_finish_s": None, "eta_next_s": None}
        try:
            done = max(0, len(self.past_tasks) - self._mission_past_base)
        except (TypeError, AttributeError):
            return {"wp_total": total, "wp_done": None, "wp_current": None,
                    "eta_finish_s": None, "eta_next_s": None}
        done = min(done, total)
        if done != self._mission_wp_done_seen:
            self._mission_wp_done_seen = done
            self._mission_last_wp_at = now
        out = {"wp_total": total, "wp_done": done,
               "wp_current": min(done + 1, total),
               "eta_finish_s": None, "eta_next_s": None}
        if done > 0:
            per_wp = elapsed / done
            out["eta_finish_s"] = round(per_wp * (total - done), 1)
            in_current = now - (self._mission_last_wp_at or now)
            # Never negative: a leg already running longer than the average is "due", not
            # "overdue by a guess" -- clamping here keeps a soft estimate from reading like the
            # hard overrun that the timeout row shows.
            out["eta_next_s"] = round(max(0.0, per_wp - in_current), 1)
        return out

    def mission_timer_state(self, now: float = None) -> dict:
        """The mission clock as data. Pure apart from the clock read, so it can be tested.

        Four fields and a state word, chosen so a display never has to do arithmetic it might
        get wrong, and never has to guess what silence means:

          running   a mission is timing right now
          idle      the vehicle is fine, nothing is being timed  (NOT the same as running=0)
          untimed   a mission is running with no timeout set at all -- say so rather than
                    render a countdown from a limit that does not exist

        `fraction` is elapsed/limit, because "600 s left of 1188 with one waypoint to go" is the
        reading that would have caught the 300 s bug in five seconds, and a bare seconds count
        is not that reading.
        """
        if now is None:
            now = self.current_time()
        if self.mission_start_time is None:
            # IDLE STILL REPORTS (Ivan, 2026-08-18): "we could keep the row there even in
            # idling, perhaps just with previous mission data or empty". A row that disappears
            # between missions is indistinguishable from a feature that was never built, and
            # the operator loses the one summary of the run that just finished.
            return {"state": "idle", "elapsed_s": None, "limit_s": None,
                    "remaining_s": None, "fraction": None,
                    "wp_total": None, "wp_done": None, "wp_current": None,
                    "eta_finish_s": None, "eta_next_s": None,
                    "emergency": bool(self.emergency_flag),
                    **self._last_mission_summary}
        elapsed = max(0.0, float(now) - float(self.mission_start_time))
        prog = self._mission_progress(now, elapsed)
        if not self.mission_timeout:
            return {"state": "untimed", "elapsed_s": round(elapsed, 1), "limit_s": None,
                    "remaining_s": None, "fraction": None, **prog,
                    "emergency": bool(self.emergency_flag),
                    **self._last_mission_summary}
        limit = float(self.mission_timeout)
        remaining = limit - elapsed
        state = {
            "state": "running",
            "elapsed_s": round(elapsed, 1),
            "limit_s": round(limit, 1),
            # A NEW FIELD, DELIBERATELY NOT A NEW `state` WORD (2026-08-19). While the emergency
            # flag is up the limit above is NOT being enforced -- the guard in lvl_3_heartbeat
            # skips it -- so a display rendering this as a live countdown is telling the operator
            # a limit will act when it will not. Mission #37 read `fraction 2.668, state:
            # running` for nearly two hours on exactly that basis.
            # A field rather than a fourth state word because the Unity dashboard scrapes this
            # JSON with a deliberately minimal parser (see _last_mission_summary's own note):
            # an unknown KEY is ignored, an unknown `state` renders as nothing at all. Whether
            # the timeout SHOULD still be enforced under an emergency is Ivan's call; reporting
            # honestly that it currently is not, is not.
            "emergency": bool(self.emergency_flag),
            # Allowed to go NEGATIVE on purpose: clamping at zero would hide an overrun that
            # has not yet been acted on, and "-4 s" is exactly the thing worth seeing.
            "remaining_s": round(remaining, 1),
            "fraction": round(elapsed / limit, 3) if limit > 0 else None,
            **prog,
            **self._last_mission_summary,
        }
        # Keep a summary so the idle row has something true to show afterwards. Written every
        # tick rather than on a "mission ended" event, because there are several ways a mission
        # can end (complete, abort, timeout) and a summary that only survives the tidy one is
        # the summary you least need.
        # FLAT KEYS, not a nested object. The Unity dashboard scrapes this JSON with a
        # deliberately minimal parser (JsonUtility would need a [Serializable] mirror class and
        # silently yields 0 for anything missing -- which is how "no data" becomes "zero" on a
        # dashboard). A nested "last" would force that parser to grow nesting for one field.
        self._last_mission_summary = {
            "last_elapsed_s": state["elapsed_s"], "last_limit_s": state["limit_s"],
            "last_wp_done": prog["wp_done"], "last_wp_total": prog["wp_total"],
        }
        return state

    def _publish_mission_timer(self):
        try:
            msg = String()
            msg.data = json.dumps(self.mission_timer_state())
            self._mission_timer_pub.publish(msg)
        except Exception:   # pragma: no cover -- a diagnostic must never break the tree
            pass

    def _apply_abort(self, origin: str, detail: str, respond: bool, respond_to: str = None):
        """Do the aborting. One body, three callers, and the origin travels with it.

        `respond_to` is separate from `respond` on purpose. Only the WARA-PS topic carries an
        actual request that a tst/response is an answer to; the tree's own health abort publishes
        an announcement with NO `response-to`, because there is nothing it is answering. It still
        publishes -- no abort path is allowed to get quieter than it was, since an abort the
        operator cannot see is worse than one they see mislabelled.
        """
        self._node.get_logger().warn(
            f"ABORT [origin={origin}] {detail} -- raising emergency flag, aborting all tasks")
        self.emergency_flag = True
        self.last_abort_origin = origin
        self.last_abort_detail = detail

        # set all tasks executing to aborted
        for task in self.tasks_executing:
            task["status"] = WaraPSTaskStates.ABORTED.value
            self.past_tasks.append(task)

        # clear the executing tasks list
        self.tasks_executing = []

        if respond:
            response_msg = {
                "agent-uuid": self._wara_ps_dict["agent-uuid"],
                "response": "all tasks aborted",
                # New field, additive: an older C2 ignores it, and this one stops having to infer
                # the origin from the text of `response-to`.
                "abort-origin": origin,
                "abort-detail": detail,
            }
            if respond_to is not None:
                response_msg["response-to"] = respond_to
            msg = String()
            msg.data = json.dumps(response_msg)
            self._wara_ps_tst_response_pub.publish(msg)
            self._node.get_logger().info(
                f"Published abort response to WARA-PS (origin={origin})")
        return

    def _bigredbutton_cb(self, data: String):
        """
        This method is called when the big red button is pressed.
        It will abort all tasks and set the aborted flag to True.
        """
        # Names its own topic. Both abort callbacks logged the identical
        # "Big Red Button pressed" line, so a log that proved an abort had arrived could
        # not say WHICH path delivered it -- and the two mean completely different things:
        # waraps/abort is an operator or C2 abort over MQTT, smarc/abort is the vehicle's
        # own stack (relayed from core/abort by sam_smarc_publisher). Cost most of a night
        # on 2026-08-13/14: the flag was confirmed to latch 91 ms before every mission
        # cancel, with no way to tell who set it.
        self._apply_abort(
            self.ABORT_ORIGIN_OPERATOR_C2,
            f"via WARA-PS topic {Topics.WARA_PS_ABORT_TOPIC} (String, from MQTT/C2): "
            f"{data.data!r}",
            respond=True, respond_to=data.data)
        return

    def _emptybigredbutton_cb(self, data: Empty):
        """
        same as above, but no feedback to be sent.
        """
        # See _bigredbutton_cb. This is the VEHICLE-SIDE abort: smarc/abort, which
        # sam_smarc_publisher relays from core/abort, and which wasp_bt also publishes to
        # itself via SMARCVehicle.abort(). If this fires with nothing obvious upstream,
        # suspect that self-publish loop before suspecting an operator.
        self._apply_abort(
            self.ABORT_ORIGIN_VEHICLE_STACK,
            f"via vehicle topic {Topics.ABORT_TOPIC} (Empty). Sources: core/abort relayed by "
            f"sam_smarc_publisher, or SMARCVehicle.abort() in this process.",
            respond=False)
        return True

    def abort(self, origin: str = None, detail: str = None):
        """Abort every executing task and raise the emergency flag.

        `origin` is required in practice: it defaults to bt_health because the only in-process
        caller is the tree's own health fallback (A_Abort), and a caller that does not say who it
        is has, by construction, not arrived from a topic. It is a keyword rather than a
        positional so an existing `abort()` call site keeps working and gets the honest answer
        instead of the old "Big Red Button pressed" lie.
        """
        origin = origin or self.ABORT_ORIGIN_BT_HEALTH
        if detail is None:
            detail = ("the behaviour tree's own health checks failed and it parked the vehicle "
                      "(A_Abort). No abort arrived from an operator or from the vehicle stack.")
        # respond=True, respond_to=None: this path used to publish a tst/response (it went through
        # _bigredbutton_cb), and taking that away would make an internal abort quieter than it was
        # -- the wrong direction entirely. It publishes the same announcement, carrying the origin,
        # and simply stops claiming to be a reply to a message nobody sent.
        self._apply_abort(origin, detail, respond=True, respond_to=None)
        return True

    def _reset_emergency_cb(self, request, response):
        self.emergency_flag = False
        # Clear the mission deadline too. An operator clearing an emergency is asking for a clean
        # slate, and an expired deadline left armed re-raises the flag on the next tick -- which is
        # what made this service report success and change nothing (2026-08-06). The timeout branch
        # above now retires its own timers, so this is belt-and-braces rather than the only guard.
        self.mission_start_time = None
        self.mission_timeout = None
        response.success = True
        # Name the cause being cleared. Data Cube spec invariant 4b: one manual clear for every
        # cause, and the cause is NAMED -- so the operator confirming it can see what they are
        # dismissing, and five clears for the same fault leave five readable records.
        cleared = self.last_abort_origin
        response.message = ("Emergency flag set to False." if cleared is None
                            else f"Emergency flag set to False (was: {cleared} -- "
                                 f"{self.last_abort_detail}).")
        self._node.get_logger().info(
            f"Emergency flag reset to False by service call. Cleared cause: "
            f"{cleared if cleared is not None else 'not recorded'}")
        # The cause stays on the record after the clear -- it is history, not live state, and
        # deleting it is how "why did this abort last time?" became unanswerable.
        return response
    
    def get_available_tasks(self):
        """
        Returns the list of available tasks.
        """
        return self.tasks_available

    def remove_available_task(self, task_name: str = None, ros_name: str = None):
        """
        Removes matching tasks from the available task list.
        Matches on WaraPS task name and/or ROS action name.
        Returns True if any task was removed.
        """
        if task_name is None and ros_name is None:
            self._node.get_logger().warn("remove_available_task called without task_name or ros_name")
            return False

        initial_count = len(self.tasks_available)
        self.tasks_available = [
            task for task in self.tasks_available
            if not (
                (task_name is not None and task.get("name") == task_name)
                or (ros_name is not None and task.get("ros_name") == ros_name)
            )
        ]

        return len(self.tasks_available) < initial_count
    
    def current_time(self):
        """
        Returns the current time in seconds.
        """
        return self._node.get_clock().now().to_msg().sec + self._node.get_clock().now().to_msg().nanosec * 1e-9 - self.start_offset
    

    def publish_bt_tip(self, tip: str):
        """
        Publishes the BT head to the MQTT broker.
        This is used to inform the WaraPS that the BT is ready to receive commands.
        """

        tip_msg = {
            "agent-uuid": self._wara_ps_dict["agent-uuid"],
            "tip": tip
        }
        msg = String()
        msg.data = json.dumps(tip_msg)
        self._wasp_bt_tip_pub.publish(msg)
        return