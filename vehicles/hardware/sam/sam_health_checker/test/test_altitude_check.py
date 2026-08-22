"""MonitorNode.altitude_check, tested without ROS.

WHY THIS EXISTS
---------------
Measured on the rig 2026-08-18: a SAM floating at the surface with the DVL reporting a stable
4.6 m of bottom clearance had `sam_rate_health_node` raising

    Fault detected: low altitude! Current altitude: -0.00, Min altitude: 0.5

every few seconds, clearing after 3 healthy cycles, then raising again. vehicle_health flapped
READY <-> ERROR, and wasp_bt's own health condition aborted and re-parked the vehicle every time
the operator cleared the emergency in Mission Control. The mission could never start.

The cause was not here. `sam_smarc_publisher` was publishing the DR odometry's **z** (height
above sea level) onto ALTITUDE_TOPIC, while `min_altitude` means **height above the seabed**.
Two different physical quantities sharing one English word, and a surfaced vehicle therefore
looked aground.

This file guards the consumer half of that fix. In particular it guards a check that had been
DEAD CODE: the `-1` no-bottom-lock sentinel test existed and had never once matched, because no
publisher on this vehicle ever emitted -1. A guard that has never fired has never been proven,
so each case below is also mutation-tested (see MUTATIONS at the bottom).

The method needs a node only for a clock, a logger and the fault bookkeeping, so a small fake
supplies those and the logic becomes ordinary Python. Crucially this exercises the REAL
`MonitorNode.altitude_check`, not a re-implementation of it -- a test that re-implements the
code under test proves only the copy.

Run: python3 test/test_altitude_check.py
"""
import pathlib
import sys
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# The node module imports the whole ROS message universe at import time. None of it is
# pip-installable and none of it is touched by altitude_check, so stub every module it names.
_STUBS = {
    "rclpy": {},
    "rclpy.node": {"Node": object},
    "ament_index_python": {"get_package_share_directory": lambda *a, **k: ""},
    "sensor_msgs": {},
    "sensor_msgs.msg": {"Imu": object, "BatteryState": object},
    "std_msgs": {},
    "std_msgs.msg": {"Int8": object, "Float32": object, "String": object},
    "smarc_msgs": {},
    "smarc_msgs.msg": {"Topics": object, "Leak": object, "DVL": object},
    "sam_msgs": {},
    "sam_msgs.msg": {"Topics": object},
    "diagnostic_msgs": {},
    "diagnostic_msgs.msg": {"DiagnosticArray": object},
    "std_srvs": {},
    "std_srvs.srv": {"Trigger": object},
}
for name, attrs in _STUBS.items():
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules.setdefault(name, mod)

from sam_health_checker.sam_rate_health_node import MonitorNode, StatusReport  # noqa: E402

results = []


def check(name, cond):
    results.append((name, bool(cond)))


class FakeLogger:
    def __init__(self):
        self.warnings = []

    def warn(self, msg):
        self.warnings.append(msg)

    def info(self, msg):
        pass


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def now(self):
        return types.SimpleNamespace(nanoseconds=self.t * 1e9)


class FakeNode:
    """Just enough of MonitorNode for altitude_check, with the real methods bound on."""

    def __init__(self, altitude, limits=None, latch=False, age_s=0.0, recover_cycles=3):
        self.clock = FakeClock()
        self.logger = FakeLogger()
        self.current_altitude = types.SimpleNamespace(data=altitude)
        self.current_altitude_time = self.clock.t - age_s
        self.current_altitude_status = StatusReport()
        self.limits = {"min_altitude": 0.5} if limits is None else limits
        self.latch_faults = latch
        self.timeout_time_sec = 5.0
        self.recover_cycles = recover_cycles

    def get_clock(self):
        return self.clock

    def get_logger(self):
        return self.logger

    # the real implementations, unbound from the class
    raise_fault = MonitorNode.raise_fault
    clear_fault = MonitorNode.clear_fault
    altitude_check = MonitorNode.altitude_check


# --- the defect that grounded the rig -----------------------------------------------------
# A surfaced vehicle in deep water: the DVL sees no bottom and says so. This MUST NOT fault.
n = FakeNode(altitude=-1.0)
st = n.altitude_check()
check("no bottom lock (-1) does not fault", not st.fault)
check("no bottom lock logs no warning", n.logger.warnings == [])

# The exact-equality trap. -1 arriving as -1.0000001 across a serialisation boundary must still
# read as the sentinel; `== -1` would silently resume faulting.
n = FakeNode(altitude=-1.0000001)
check("sentinel just below -1 still reads as invalid", not n.altitude_check().fault)

# NaN and inf must read as NO INFORMATION. Note that merely asserting "does not fault" is NOT a
# discriminating test: `nan < 0.5` and `inf < 0.5` are both False, so a broken guard reaches the
# else branch and reports HEALTHY -- silently curing a real fault with a garbage reading. The
# property worth testing is therefore that they cannot clear an existing fault.
for label, bad in (("NaN", float("nan")), ("inf", float("inf"))):
    n = FakeNode(altitude=0.2)
    n.altitude_check()
    assert n.current_altitude_status.fault, "setup: expected a low-altitude fault first"
    n.current_altitude.data = bad
    for _ in range(10):
        n.altitude_check()
    check(f"{label} altitude does not fault", not n.logger.warnings[1:])
    check(f"{label} altitude cannot clear a real fault", n.current_altitude_status.fault)
    check(f"{label} altitude builds no healthy streak",
          n.current_altitude_status.healthy_streak == 0)

# --- the check still has to do its actual job -----------------------------------------------
# Genuinely close to the bottom. This is the reason the check exists at all.
n = FakeNode(altitude=0.2)
st = n.altitude_check()
check("0.2 m clearance faults", st.fault)
check("low-altitude fault names the value", "0.20" in st.reason and "low altitude" in st.reason)

n = FakeNode(altitude=4.6)   # what Unity's DVL actually reads at the Kristineberg quay
check("4.6 m clearance is healthy", not n.altitude_check().fault)

n = FakeNode(altitude=0.5)
check("exactly min_altitude is not a fault", not n.altitude_check().fault)

# An invalid reading must not COUNT AS HEALTHY either -- it must not debounce a real fault away.
n = FakeNode(altitude=0.2)
n.altitude_check()
check("faulted before invalid readings arrive", n.current_altitude_status.fault)
n.current_altitude.data = -1.0
for _ in range(10):
    n.altitude_check()
check("a stream of invalid readings never clears a real fault",
      n.current_altitude_status.fault)
check("invalid readings do not build a healthy streak",
      n.current_altitude_status.healthy_streak == 0)

# Real recovery still works, after the configured number of genuinely healthy cycles.
n.current_altitude.data = 4.6
for _ in range(n.recover_cycles):
    n.altitude_check()
check("real clearance clears the fault after recover_cycles",
      not n.current_altitude_status.fault)

# --- surrounding contract -------------------------------------------------------------------
n = FakeNode(altitude=0.2, limits={})
check("no min_altitude configured means no altitude fault", not n.altitude_check().fault)
check("ready is still asserted without a limit", n.current_altitude_status.ready)

n = FakeNode(altitude=4.6, age_s=30.0)
check("a stale altitude faults on timeout", n.altitude_check().fault)

n = FakeNode(altitude=0.2, latch=True)
n.altitude_check()
n.current_altitude.data = 4.6
for _ in range(10):
    n.altitude_check()
check("latched faults stay latched", n.current_altitude_status.fault)

n = FakeNode(altitude=0.2)
n.current_altitude = None
check("never having heard an altitude is not a fault", not n.altitude_check().fault)

# MUTATIONS this suite was RUN against and verified to catch (2026-08-18):
#   1. `altitude_m <= -1`             -> `altitude_m == -1`    : caught (sentinel just below -1)
#   2. drop the isfinite() clause                              : caught (NaN/inf cases)
#   3. `return status` on invalid     -> `clear_fault(...)`    : caught (invalid clears real fault)
#   4. `data < min_altitude`          -> `data <= min_altitude`: caught (exactly min_altitude)
#
# Mutation 2 SURVIVED the first version of this file, which is why the NaN/inf cases are written
# the way they are. The obvious assertion -- "a NaN altitude does not fault" -- passes with the
# guard deleted, because `nan < 0.5` and `inf < 0.5` are both False, so the code falls through to
# the else branch and reports the vehicle HEALTHY on a garbage reading. The test looked green and
# proved nothing. Any future check added here should be mutated before it is believed.

failed = [n for n, ok in results if not ok]
print(f"{len(results) - len(failed)}/{len(results)} checks passed")
for name in failed:
    print(f"  FAIL: {name}")
sys.exit(1 if failed else 0)
