import math
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.toml"
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import load_app_config
from race_model.models import Athlete, mean_running_time, riegel_running_time, two_regime_running_time


def _make_test_athlete(
    *,
    name: str,
    sprint_ref_time: float,
    ref_time: float,
    endurance_exponent: float,
) -> Athlete:
    return Athlete(
        name=name,
        run_sprint_threshold_distance_m=400.0,
        run_sprint_reference_distance_m=100.0,
        run_sprint_reference_time_sec=sprint_ref_time,
        run_reference_distance_m=1500.0,
        run_reference_time_sec=ref_time,
        run_endurance_exponent=endurance_exponent,
        run_max_modeled_distance_m=5000.0,
        eat_base_sec=1.0,
        eat_slowdown_sec=0.0,
        post_eat_linear=0.0,
        post_eat_quadratic=0.0,
        eat_noise_sec=0.0,
        run_sprint_reference_time_rel_std=0.0,
        run_reference_time_rel_std=0.0,
        run_endurance_rel_std=0.0,
        post_eat_rel_std=0.0,
    )


class RunningModelTests(unittest.TestCase):
    def test_riegel_formula_matches_reference_distance(self) -> None:
        self.assertAlmostEqual(
            riegel_running_time(
                distance_m=1500.0,
                reference_distance_m=1500.0,
                reference_time_sec=240.0,
                endurance_exponent=1.06,
            ),
            240.0,
        )

    def test_riegel_formula_scales_distance(self) -> None:
        expected = 240.0 * math.pow(5000.0 / 1500.0, 1.06)
        self.assertAlmostEqual(
            riegel_running_time(
                distance_m=5000.0,
                reference_distance_m=1500.0,
                reference_time_sec=240.0,
                endurance_exponent=1.06,
            ),
            expected,
        )

    def test_two_regime_model_is_continuous_at_threshold(self) -> None:
        threshold_time = two_regime_running_time(
            distance_m=400.0,
            sprint_threshold_distance_m=400.0,
            sprint_reference_distance_m=100.0,
            sprint_reference_time_sec=9.8,
            reference_distance_m=1500.0,
            reference_time_sec=245.0,
            endurance_exponent=1.14,
            max_modeled_distance_m=5000.0,
        )
        expected = riegel_running_time(
            distance_m=400.0,
            reference_distance_m=1500.0,
            reference_time_sec=245.0,
            endurance_exponent=1.14,
        )
        self.assertAlmostEqual(threshold_time, expected)

    def test_sprint_specialist_can_win_short_and_lose_long(self) -> None:
        sprinter = _make_test_athlete(
            name="Sprinter",
            sprint_ref_time=9.8,
            ref_time=270.0,
            endurance_exponent=1.15,
        )
        balanced = _make_test_athlete(
            name="Balanced",
            sprint_ref_time=12.0,
            ref_time=230.0,
            endurance_exponent=1.06,
        )

        self.assertLess(mean_running_time(100.0, 0, sprinter), mean_running_time(100.0, 0, balanced))
        self.assertGreater(mean_running_time(5000.0, 0, sprinter), mean_running_time(5000.0, 0, balanced))

    def test_config_loads_sprint_and_endurance_parameters(self) -> None:
        config = load_app_config(CONFIG_PATH)

        chestnut = config.get_athlete("Chestnut")
        self.assertEqual(config.sprint_threshold_distance_m, 400.0)
        self.assertEqual(config.sprint_reference_distance_m, 100.0)
        self.assertEqual(config.reference_distance_m, 1500.0)
        self.assertEqual(config.max_modeled_distance_m, 5000.0)
        self.assertEqual(chestnut.run_sprint_reference_time_sec, 15.6)
        self.assertEqual(chestnut.run_reference_distance_m, 1500.0)
        self.assertEqual(chestnut.run_reference_time_sec, 390.0)
        self.assertAlmostEqual(chestnut.run_endurance_exponent, 1.07)

    def test_mean_running_time_applies_post_eating_multiplier(self) -> None:
        config = load_app_config(CONFIG_PATH)
        chestnut = config.get_athlete("Chestnut")

        expected = (
            two_regime_running_time(
                distance_m=1500.0,
                sprint_threshold_distance_m=chestnut.run_sprint_threshold_distance_m,
                sprint_reference_distance_m=chestnut.run_sprint_reference_distance_m,
                sprint_reference_time_sec=chestnut.run_sprint_reference_time_sec,
                reference_distance_m=chestnut.run_reference_distance_m,
                reference_time_sec=chestnut.run_reference_time_sec,
                endurance_exponent=chestnut.run_endurance_exponent,
                max_modeled_distance_m=chestnut.run_max_modeled_distance_m,
            )
            * (1.0 + chestnut.post_eat_linear * 2 + chestnut.post_eat_quadratic * 4)
        )
        self.assertAlmostEqual(mean_running_time(1500.0, 2, chestnut), expected)

    def test_invalid_distance_raises(self) -> None:
        with self.assertRaises(ValueError):
            two_regime_running_time(
                distance_m=0.0,
                sprint_threshold_distance_m=400.0,
                sprint_reference_distance_m=100.0,
                sprint_reference_time_sec=12.5,
                reference_distance_m=1500.0,
                reference_time_sec=240.0,
                endurance_exponent=1.06,
                max_modeled_distance_m=5000.0,
            )

    def test_distance_above_modeled_range_raises(self) -> None:
        with self.assertRaises(ValueError):
            two_regime_running_time(
                distance_m=5001.0,
                sprint_threshold_distance_m=400.0,
                sprint_reference_distance_m=100.0,
                sprint_reference_time_sec=12.5,
                reference_distance_m=1500.0,
                reference_time_sec=240.0,
                endurance_exponent=1.06,
                max_modeled_distance_m=5000.0,
            )

    def test_unknown_runner_raises(self) -> None:
        config = load_app_config(CONFIG_PATH)

        with self.assertRaises(ValueError):
            config.get_athlete("runner_A")


if __name__ == "__main__":
    unittest.main()
