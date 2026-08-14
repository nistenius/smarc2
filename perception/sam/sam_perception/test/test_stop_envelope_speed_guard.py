"""R_stop is quadratic in u, so the speed feeding it must be validated.

Rig, 2026-08-14: the estimator published u = 2.1e5 m/s (the HUD read "DR no data" at the
same moment) and the detector computed R_stop = 2.3e13 m. Every obstacle in the world is
inside that radius, so the protective stop latched and could never release, the behaviour
tree aborted, and the latched emergency flag then refused every following mission.

Invisible until that day only because the envelope had been running FLAT — t_react = 0 and
a_stop = 1e6, where u contributes nothing. The speed-dependent envelope did not introduce
this; it revealed it.

Pure arithmetic, no ROS: the guard is a clamp and a formula.
"""
import math
import unittest

STOP_U_MAX = 2.0


def guarded_speed(u, u_max=STOP_U_MAX):
    """The rule in obstacle_detector.odom_cb, isolated."""
    if not math.isfinite(u):
        return u_max
    u = abs(float(u))
    return min(u, u_max)


def r_stop(u, margin=0.5, t_react=1.0, a_stop=0.35):
    return margin + u * t_react + u * u / (2.0 * a_stop)


class TestSpeedGuard(unittest.TestCase):
    def test_the_measured_failure_no_longer_explodes_the_envelope(self):
        self.assertGreater(r_stop(214841.0), 1e10)          # what actually happened
        self.assertLess(r_stop(guarded_speed(214841.0)), 10.0)

    def test_nan_and_inf_do_not_propagate(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            self.assertTrue(math.isfinite(r_stop(guarded_speed(bad))))

    def test_an_unknown_speed_clamps_UP_not_to_zero(self):
        """Not knowing your speed is not a reason to assume you are stopped — that would
        shrink the stop envelope exactly when the estimator cannot be trusted."""
        self.assertEqual(guarded_speed(float("nan")), STOP_U_MAX)
        self.assertGreater(r_stop(guarded_speed(float("nan"))), r_stop(0.0))

    def test_reverse_is_speed_too(self):
        self.assertEqual(guarded_speed(-0.4), 0.4)

    def test_normal_dock_speeds_are_untouched(self):
        """The guard must not quietly reshape the envelope in ordinary operation."""
        for u in (0.0, 0.3, 0.5, 1.0, 1.9):
            self.assertAlmostEqual(guarded_speed(u), u)

    def test_the_dock_envelope_is_what_was_asked_for(self):
        """Ivan 2026-08-14: 0.5 m. That is the MARGIN; the trigger grows with speed."""
        self.assertAlmostEqual(r_stop(0.0), 0.5, places=2)
        self.assertLess(r_stop(0.3), 1.0)
        self.assertLess(r_stop(0.5), 1.5)

    def test_FLAT_mode_attenuates_a_bad_speed_but_does_not_survive_it(self):
        """Worth being exact, because I first wrote this test asserting flat mode was
        immune, and it is not.

        Flat sets t_react = 0 and a_stop = 1e6, which divides the quadratic term by about
        three million. At ordinary speeds that is indistinguishable from "u does not
        matter" — which is why the envelope's dependence on an unvalidated speed went
        unnoticed. At u = 2.1e5 m/s it still yields tens of kilometres. So the guard is
        needed in BOTH modes, and the earlier flat runs were fine because the estimator
        was fine, not because flat protected them.
        """
        flat = lambda u: r_stop(u, margin=0.5, t_react=0.0, a_stop=1e6)
        self.assertAlmostEqual(flat(0.5), 0.5, places=5)      # ordinary speeds: no effect
        self.assertGreater(flat(214841.0), 1e4)               # a bad one still ruins it
        self.assertLess(flat(guarded_speed(214841.0)), 1.0)   # guarded, in either mode


if __name__ == "__main__":
    unittest.main()


class TestAnAbsentSpeedSourceIsNotZero(unittest.TestCase):
    """The more dangerous half, found on the rig 2026-08-14.

        /sam_auv_v1/dr/odom   Publisher count: 0   Subscription count: 5

    The detector's configured speed source had no publisher at all -- five nodes,
    including this one, subscribed to a topic nobody wrote. `self.speed` initialises to
    0.0 and is only ever written by odom_cb, so with the source absent u stays 0.0
    forever, the reaction and braking terms vanish, and R_stop silently collapses to the
    bare margin. The protective stop degrades into exactly the flat envelope that put SAM
    into the dry-dock wall, and nothing anywhere says so.

    A safety layer whose input is missing must fail loud and conservative.
    """

    def envelope_speed(self, age_s, last_speed=0.0, timeout=3.0, u_max=STOP_U_MAX):
        """The rule in obstacle_detector._envelope_speed, isolated."""
        if age_s is None or age_s > timeout:
            return u_max
        return last_speed

    def test_a_source_that_never_published_reads_as_unknown(self):
        self.assertEqual(self.envelope_speed(None), STOP_U_MAX)

    def test_and_unknown_does_not_collapse_the_envelope_to_the_margin(self):
        collapsed = r_stop(0.0)
        honest = r_stop(self.envelope_speed(None))
        self.assertAlmostEqual(collapsed, 0.5, places=2)
        self.assertGreater(honest, collapsed,
                           "an absent speed made the stop layer LESS protective, silently")

    def test_a_stale_reading_is_also_unknown(self):
        self.assertEqual(self.envelope_speed(9.0, last_speed=0.4), STOP_U_MAX)

    def test_a_fresh_reading_is_believed(self):
        self.assertEqual(self.envelope_speed(0.2, last_speed=0.4), 0.4)

    def test_a_genuinely_stationary_vehicle_still_reads_zero(self):
        """Stationary and unknown must not be conflated in the other direction either."""
        self.assertEqual(self.envelope_speed(0.1, last_speed=0.0), 0.0)
