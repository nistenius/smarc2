"""When is a mission OVER, and has this one already been surfaced?

Invariant 5b's last gap. `A_SurfaceAndReport` (farm_inspection_core.SurfaceAndReportCore) has
existed since 2026-08-17 and does the surfacing correctly -- but it is reachable only from the
`auv-farm-inspection` task. **An ordinary waypoint mission never surfaces.** The last waypoint
empties the queue, the tree falls through to `A_Chilling`, and the diving controller keeps
holding its last commanded depth, indefinitely.

That is not only a recovery problem. It is why the recorder never stops: the end-of-run rule
(`bridge_node._end_of_run_recording`) requires *mission ended AND SURFACED AND idle*, where
"surfaced" is the controller's own `ctrl/neutral_handoff` verdict rather than a depth reading.
A plain mission never produces that verdict, so the 300 s grace never starts and the bag is
never closed. Observed twice on 2026-08-29, and the workaround -- killing the recorder by hand
-- destroyed a flown mission, because `bridge_node` owns that recorder and restarts it.

This module answers ONE question, in pure logic with no ROS and no tree:

    given how many tasks are executing right now, should the tree surface the vehicle?

and it is an EDGE, not a level: the queue being empty is not enough, because a vehicle that has
never flown anything is also empty and must not blow its tank on power-up. The mission must
have been seen to RUN and then to STOP.

WHY THE STATE LIVES ON THE TASK HANDLER, NOT ON THE BEHAVIOUR. `ros_bt._update_task_handler_tree`
REBUILDS the task-handler subtree whenever the set of available tasks changes -- which happens
when an action server's heartbeat comes or goes, i.e. at arbitrary moments including mid-mission.
A flag held on the behaviour would be destroyed by that rebuild and the vehicle would either
surface twice or not at all, depending on when the rebuild landed. The handler outlives the
tree, so the memory goes there. (Same family as invariant 12's "a guard whose condition is set
in a different process is not a guard": a latch that a rebuild can erase is not a latch.)
"""
from typing import Optional


class EndOfMissionSurfaceCore:
    """Edge-detects 'the mission just ended' and remembers whether it has been acted on.

    States, in the order a flight passes through them:

        idle, never flown      -> should_surface() False   (do NOT surface on power-up)
        tasks executing        -> armed
        queue empties          -> should_surface() True    (ONCE)
        surface attempted      -> should_surface() False   (whatever the outcome)
        a new mission starts   -> armed again
    """

    def __init__(self):
        #: True once tasks have been seen executing and not yet acted on.
        self.armed = False
        #: True while the queue is empty AND we owe this mission a surfacing.
        self.pending = False
        #: The outcome of the last attempt, for the record: confirmed | timeout | held |
        #: no_position | None. NEVER used to decide anything -- only reported.
        self.last_outcome: Optional[str] = None
        self.missions_surfaced = 0

    def note_executing(self, n_executing: int) -> None:
        """Call every tick with len(task_handler.get_executing_tasks())."""
        if n_executing > 0:
            # A mission is running. Arm, and clear any pending surfacing from a previous
            # one -- if a new mission started, the old one's recovery is moot.
            self.armed = True
            self.pending = False
        elif self.armed:
            # The queue just emptied and we have seen it run: this mission owes a surfacing.
            self.armed = False
            self.pending = True

    def should_surface(self) -> bool:
        return self.pending

    def note_surface_finished(self, outcome: Optional[str] = None) -> None:
        """Called when A_SurfaceAndReport has reached a terminal state, ANY terminal state.

        Deliberately not conditional on success. A surfacing that was HELD by the protective
        stop, or that timed out without the controller confirming, has still been attempted
        and reported; retrying it on the next tick would loop the tree against a vehicle that
        is stopped against something. The operator has the report and the decision.
        """
        self.pending = False
        self.last_outcome = outcome
        self.missions_surfaced += 1

    def describe(self) -> str:
        if self.pending:
            return "mission ended — end-of-mission surfacing owed"
        if self.armed:
            return "mission running"
        if self.last_outcome:
            return f"idle — last end-of-mission surfacing: {self.last_outcome}"
        return "idle — no mission flown since start"


def get_or_create(task_handler) -> EndOfMissionSurfaceCore:
    """The one instance for this vehicle, hung off the handler so a tree rebuild cannot
    erase it. Created lazily so nothing else has to know about it."""
    core = getattr(task_handler, "_end_of_mission_core", None)
    if core is None:
        core = EndOfMissionSurfaceCore()
        setattr(task_handler, "_end_of_mission_core", core)
    return core
