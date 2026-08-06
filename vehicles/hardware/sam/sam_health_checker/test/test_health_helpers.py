"""TopicRateMonitor, tested without ROS.

WHY THIS EXISTS
---------------
This class decides whether wasp_bt will accept a mission at all: it drives vehicle_health, and
`_handle_tst_command` refuses every start-tst while health_status != READY. It had two defects
that only showed up as "the mission is silently rejected", which is the worst possible symptom
because it points at the BT rather than here.

The class needs a ROS node only for a clock, a logger and create_subscription/create_timer, so a
twenty-line fake supplies all four and the logic becomes ordinary Python. Fake time is a feature,
not a compromise: the DVL case that caused the real bug takes 4+ seconds of wall clock per
evaluation to reproduce honestly, and nobody runs a test suite that does that.

Run: python3 test_health_helpers.py
"""
import pathlib
import sys
import types

# Run standalone from anywhere: `python3 test/test_health_helpers.py` from the package root, or
# via colcon. The package is not installed when this runs, so point at it directly.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# rclpy is imported at module scope by health_helpers and is not pip-installable. Nothing in
# TopicRateMonitor actually calls it -- the node is injected -- so stub the import.
for name, attrs in (("rclpy", {}), ("rclpy.node", {"Node": object})):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules.setdefault(name, mod)

from sam_health_checker.helpers.health_helpers import TopicRateMonitor   # noqa: E402

results = []
def check(name, cond):
    results.append((name, bool(cond)))


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t
    def now(self):
        return types.SimpleNamespace(nanoseconds=self.t * 1e9)


class FakeLogger:
    def __init__(self):
        self.warns, self.infos = [], []
    def info(self, m):
        self.infos.append(m)
    def warn(self, m):
        self.warns.append(m)
    def error(self, m):
        self.warns.append(m)


class FakeNode:
    """Everything TopicRateMonitor touches, and nothing else."""
    def __init__(self):
        self.clock = FakeClock()
        self.logger = FakeLogger()
        self.subs, self.timers = [], []
    def get_clock(self):
        return self.clock
    def get_logger(self):
        return self.logger
    def create_subscription(self, msg_type, topic, cb, qos):
        self.subs.append((topic, cb))
        return object()
    def create_timer(self, timer_period_sec=None, callback=None):
        self.timers.append((timer_period_sec, callback))
        return object()


DVL = "/sam/core/dvl"
IMU = "/sam/core/imu"


def monitor(rate=1.0, topics=None, **kw):
    node = FakeNode()
    topics = topics or {DVL: [object, rate]}
    m = TopicRateMonitor(node, topics, timeout_time_sec=7.5, window_size=5,
                         report_interval=1.0, **kw)
    return node, m


def publish(node, m, topic, n, period):
    """n messages at `period` seconds apart, advancing the fake clock as a publisher would."""
    for _ in range(n):
        node.clock.t += period
        m.timestamps[topic].append(node.clock.t)


# --- the rate-dependent timeout, which is the bug that cost the most ---------------------------
# The staleness test used to measure from timestamps[0] -- the OLDEST sample in a window_size-deep
# deque -- so the effective timeout shrank as the topic's nominal rate fell. At 20 Hz a 5-sample
# window spans 0.2 s and nobody notices. At the DVL's 1 Hz nominal the oldest sample is ~4 s old
# during perfectly healthy operation, so a configured 7.5 s timeout behaved like 3.5 s.
node, m = monitor(rate=1.0)
publish(node, m, DVL, 5, 1.0)              # a full window of perfectly nominal 1 Hz DVL
check("a nominal 1 Hz topic is not faulted (the DVL regression)",
      m._evaluate_topic(DVL, 1.0) == "")
check("window really is 4 s deep, so the old code had <4 s of headroom left",
      round(m.timestamps[DVL][-1] - m.timestamps[DVL][0], 3) == 4.0)

node.clock.t += 3.6                        # 3.6 s of silence: past the OLD effective limit...
check("3.6 s of silence on a 1 Hz topic is still healthy against a 7.5 s timeout",
      m._evaluate_topic(DVL, 1.0) == "")
node.clock.t += 4.2                        # ...total 7.8 s, now genuinely past 7.5 s
reason = m._evaluate_topic(DVL, 1.0)
check("but 7.8 s of silence does fault", reason.startswith("timeout"))
check("and the reason quotes the measured gap and the limit",
      "7.80" in reason and "7.50" in reason)

# --- rate tolerance ----------------------------------------------------------------------------
# Zero tolerance meant any jitter below nominal was a fault. Unusable at 1 Hz.
node, m = monitor(rate=1.0, rate_tolerance=0.8)
publish(node, m, DVL, 5, 1.05)             # 0.95 Hz -- slightly under nominal
check("5% under nominal is not a fault at 0.8 tolerance", m._evaluate_topic(DVL, 1.0) == "")
node, m = monitor(rate=1.0, rate_tolerance=0.8)
publish(node, m, DVL, 5, 1.5)              # 0.67 Hz -- below 0.8 Hz
r = m._evaluate_topic(DVL, 1.0)
check("33% under nominal IS a fault", r.startswith("rate"))
check("the rate reason shows measured, limit and tolerance", "0.67" in r and "0.80" in r)

# --- too little data is not a verdict ----------------------------------------------------------
node, m = monitor()
check("no samples at all is not a fault", m._evaluate_topic(DVL, 1.0) == "")
publish(node, m, DVL, 1, 1.0)
check("one sample is still not a fault -- a rate needs two", m._evaluate_topic(DVL, 1.0) == "")

# --- latching: the vehicle default -------------------------------------------------------------
node, m = monitor(rate=1.0, latch_faults=True)
publish(node, m, DVL, 5, 1.0)
m.determine_fault()
check("healthy topic, no fault latched", m.fault is False)
node.clock.t += 20.0
check("a dropout faults", m.determine_fault() is True)
publish(node, m, DVL, 5, 1.0)              # topic comes back, perfectly healthy
for _ in range(10):
    m.determine_fault()
check("under latching the fault does NOT clear when the topic returns -- surface and "
      "investigate, not carry on", m.fault is True)
check("and the faulted topic is named", m.fault_reasons() and DVL in m.fault_reasons()[0])

# --- the reset service ------------------------------------------------------------------------
m.reset_faults()
check("reset_faults clears the latched fault", m.fault is False)
check("reset_faults clears the per-topic state too", not m.fault_reasons())
check("reset_faults also drops the stale window -- otherwise the timestamps from before the "
      "dropout re-trigger staleness on the very next evaluation",
      len(m.timestamps[DVL]) == 0)
check("and immediately after a reset the topic is not faulted again",
      m.determine_fault() is False)

# --- non-latching: the sim default ------------------------------------------------------------
# Every Unity editor stop/play drops all topics at once. Under latching that left the node stuck
# on VEHICLE_HEALTH_ERROR forever and wasp_bt rejected every start-tst, so one editor restart
# silently blocked all missions until someone restarted the node by hand.
node, m = monitor(rate=1.0, latch_faults=False, recover_cycles=3)
publish(node, m, DVL, 5, 1.0)
m.determine_fault()
node.clock.t += 20.0
check("sim: a dropout still faults", m.determine_fault() is True)

publish(node, m, DVL, 5, 1.0)
m.determine_fault()
check("sim: one good evaluation is not enough (debounce)", m.fault is True)
publish(node, m, DVL, 2, 1.0); m.determine_fault()
check("sim: two is not enough either", m.fault is True)
publish(node, m, DVL, 2, 1.0); m.determine_fault()
check("sim: three consecutive good evaluations clear it -- an editor restart is survivable",
      m.fault is False)

# A topic flapping on the limit must not flap the vehicle in and out of READY. The streak has
# to RESTART on a bad evaluation, not resume: two good, one bad, two good is not three good.
# Each "good" republishes a full window, so the evaluation is unambiguously healthy rather than
# a short window still spanning the dropout.
node, m = monitor(rate=1.0, latch_faults=False, recover_cycles=3)
publish(node, m, DVL, 5, 1.0); m.determine_fault()
node.clock.t += 20.0
check("sim: faulted to begin with", m.determine_fault() is True)

for _ in range(2):
    publish(node, m, DVL, 5, 1.0); m.determine_fault()
check("sim: two good evaluations, still faulted", m.fault is True)
check("sim: and the streak stands at two", m._healthy_streak[DVL] == 2)

node.clock.t += 20.0; m.determine_fault()          # bad again, before the streak completes
check("sim: a bad evaluation resets the streak to zero, it does not decrement",
      m._healthy_streak[DVL] == 0 and m.fault is True)

for _ in range(2):
    publish(node, m, DVL, 5, 1.0); m.determine_fault()
check("sim: two more good is still not three consecutive -- fault holds", m.fault is True)
publish(node, m, DVL, 5, 1.0); m.determine_fault()
check("sim: the third consecutive good evaluation clears it", m.fault is False)

# --- several topics are judged independently ---------------------------------------------------
node, m = monitor(topics={DVL: [object, 1.0], IMU: [object, 20.0]}, latch_faults=False)
publish(node, m, DVL, 5, 1.0)
publish(node, m, IMU, 5, 0.05)
m.determine_fault()
check("two healthy topics, no fault", m.fault is False)
# Starve only the DVL. Advancing the clock ages both, so re-feed the IMU at its own rate.
node.clock.t += 20.0
publish(node, m, IMU, 5, 0.05)
m.determine_fault()
check("one bad topic faults the vehicle", m.fault is True)
check("and only the bad topic is blamed",
      len(m.fault_reasons()) == 1 and DVL in m.fault_reasons()[0])

# --- ready is about ever having seen data, not about health ------------------------------------
node, m = monitor(topics={DVL: [object, 1.0], IMU: [object, 20.0]})
check("not ready before any message arrives", m.determine_ready() is False)
publish(node, m, DVL, 1, 1.0)
check("not ready while one topic is still silent", m.determine_ready() is False)
publish(node, m, IMU, 1, 0.05)
check("ready once every topic has produced at least one message",
      m.determine_ready() is True)
check("ready is sticky -- it describes startup, not current health",
      (node.clock.__setattr__("t", node.clock.t + 100.0), m.determine_ready())[1] is True)

# --- wiring ------------------------------------------------------------------------------------
node, m = monitor(topics={DVL: [object, 1.0], IMU: [object, 20.0]})
check("one subscription per topic", sorted(t for t, _ in node.subs) == sorted([DVL, IMU]))
check("exactly one report timer, at the report interval",
      len(node.timers) == 1 and node.timers[0][0] == 1.0)
cb = dict(node.subs)[DVL]
before = len(m.timestamps[DVL])
cb(object())
check("the subscription callback records a timestamp", len(m.timestamps[DVL]) == before + 1)
check("the window is bounded by window_size",
      [cb(object()) for _ in range(20)] and len(m.timestamps[DVL]) == 5)
check("recover_cycles is floored at 1 so 0 cannot mean 'never recover'",
      monitor(recover_cycles=0)[1].recover_cycles == 1)


# --- zero-length sample intervals under sim time (2026-08-05) ---------------------------------
# `intervals` being non-empty was taken as proof it was safe to divide by its mean. Under
# use_sim_time several messages share one /clock tick and every interval is 0.0, so the mean is
# 0.0 -- and sam_rate_health_node exited with ZeroDivisionError on both SAM VMs the moment Unity
# began publishing. Dying is the worst possible response: the topic loses its publisher entirely,
# so wasp_bt keeps its initial VEHICLE_HEALTH_ERROR and refuses every mission.
node, m = monitor(rate=20.0, topics={IMU: [object, 20.0]})
publish(node, m, IMU, 5, 0.0)              # five samples, all on the same clock tick
try:
    reason = m._evaluate_topic(IMU, 20.0)
    crashed = False
except ZeroDivisionError:
    reason, crashed = "ZeroDivisionError", True
check("identical timestamps do not kill the monitor", not crashed)
check("identical timestamps are not reported as a rate fault", reason == "")

# The same window, but with the node's own clock well past it, must still time out normally --
# the guard above must not turn into a way of never faulting.
node.clock.t += 30.0
check("a stale topic still times out even with zero-length intervals",
      "timeout" in m._evaluate_topic(IMU, 20.0))

# One real interval among zeros is still judged on its average, not skipped.
node, m = monitor(rate=20.0, topics={IMU: [object, 20.0]})
publish(node, m, IMU, 4, 0.0)
publish(node, m, IMU, 1, 1.0)              # mean interval 0.2 s -> 5 Hz, well under 20 x 0.8
check("a genuinely slow topic is still faulted",
      "rate" in m._evaluate_topic(IMU, 20.0))

# Tallied HERE, immediately before use. It used to be computed right after the last check in the
# file, which was correct only for as long as nobody appended another one -- and the first person
# who did (2026-08-05) got two visible FAIL lines, "44/44 passed", and exit code 0.
fail = [n for n, ok in results if not ok]

print()
for n, ok in results:
    print(("  PASS  " if ok else "  FAIL  ") + n)
print(f"\n{len(results)-len(fail)}/{len(results)} passed")
sys.exit(1 if fail else 0)
