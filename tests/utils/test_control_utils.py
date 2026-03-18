from types import SimpleNamespace

import pytest

from lerobot.utils.control_utils import _apply_keyboard_hotkey_token, sanity_check_bimanual_piper_pair
from lerobot.utils.recording_annotations import EPISODE_FAILURE, EPISODE_SUCCESS


@pytest.mark.parametrize(
    ("robot_type", "teleop_type"),
    [
        ("bi_piper_follower", "bi_piper_leader"),
        ("bi_piperx_follower", "bi_piperx_leader"),
        ("so101_follower", "so101_leader"),
    ],
)
def test_sanity_check_bimanual_piper_pair_accepts_valid_pairs(robot_type, teleop_type):
    sanity_check_bimanual_piper_pair(
        SimpleNamespace(type=robot_type),
        SimpleNamespace(type=teleop_type),
    )


def test_sanity_check_bimanual_piper_pair_accepts_missing_teleop():
    sanity_check_bimanual_piper_pair(SimpleNamespace(type="bi_piperx_follower"), None)


@pytest.mark.parametrize(
    ("robot_type", "teleop_type"),
    [
        ("bi_piper_follower", "bi_piperx_leader"),
        ("bi_piperx_follower", "bi_piper_leader"),
        ("so101_follower", "bi_piperx_leader"),
        ("so101_follower", "bi_piper_leader"),
    ],
)
def test_sanity_check_bimanual_piper_pair_rejects_mixed_pairs(robot_type, teleop_type):
    with pytest.raises(ValueError, match="must be paired"):
        sanity_check_bimanual_piper_pair(
            SimpleNamespace(type=robot_type),
            SimpleNamespace(type=teleop_type),
        )


def test_apply_keyboard_hotkey_token_updates_intervention_event():
    events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
        "toggle_intervention": False,
        "episode_outcome": None,
    }

    handled = _apply_keyboard_hotkey_token(events, "I", intervention_toggle_key="i")

    assert handled is True
    assert events["toggle_intervention"] is True
    assert events["exit_early"] is False


@pytest.mark.parametrize(
    ("token", "expected_field", "expected_value"),
    [
        ("s", "episode_outcome", EPISODE_SUCCESS),
        ("f", "episode_outcome", EPISODE_FAILURE),
        ("esc", "stop_recording", True),
        ("right", "exit_early", True),
        ("left", "rerecord_episode", True),
    ],
)
def test_apply_keyboard_hotkey_token_updates_terminal_controls(token, expected_field, expected_value):
    events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
        "toggle_intervention": False,
        "episode_outcome": None,
    }

    handled = _apply_keyboard_hotkey_token(
        events,
        token,
        intervention_toggle_key="i",
        episode_success_key="s",
        episode_failure_key="f",
    )

    assert handled is True
    assert events[expected_field] == expected_value


def test_apply_keyboard_hotkey_token_rejects_unknown_token():
    events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
        "toggle_intervention": False,
        "episode_outcome": None,
    }

    handled = _apply_keyboard_hotkey_token(events, "x", intervention_toggle_key="i")

    assert handled is False
    assert events == {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
        "toggle_intervention": False,
        "episode_outcome": None,
    }
