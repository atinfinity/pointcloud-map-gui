"""GLFW picks Wayland over X11 whenever it can, and under WSLg the Wayland
path never opens a window -- see app._prefer_x11_under_wsl."""
import pytest

from pointcloud_map_gui.app import _prefer_x11_under_wsl

WSL = {"WSL_DISTRO_NAME": "Ubuntu-24.04", "DISPLAY": ":0", "WAYLAND_DISPLAY": "wayland-0"}


@pytest.fixture
def env(monkeypatch):
    """A bare environment the test fills in, so the host's own does not leak."""
    for name in ("XDG_SESSION_TYPE", "DISPLAY", "WAYLAND_DISPLAY",
                 "WSL_DISTRO_NAME", "WSL_INTEROP"):
        monkeypatch.delenv(name, raising=False)
    import os
    return os.environ


def _set(env, monkeypatch, values):
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    return env


def test_wsl_session_is_switched_to_x11(env, monkeypatch):
    _set(env, monkeypatch, WSL)
    _prefer_x11_under_wsl()
    assert env["XDG_SESSION_TYPE"] == "x11"


def test_wsl_interop_alone_is_enough_to_detect_wsl(env, monkeypatch):
    _set(env, monkeypatch, {"WSL_INTEROP": "/run/WSL/1_interop", "DISPLAY": ":0"})
    _prefer_x11_under_wsl()
    assert env["XDG_SESSION_TYPE"] == "x11"


def test_an_explicit_session_type_is_respected(env, monkeypatch):
    _set(env, monkeypatch, {**WSL, "XDG_SESSION_TYPE": "wayland"})
    _prefer_x11_under_wsl()
    assert env["XDG_SESSION_TYPE"] == "wayland"


def test_a_non_wsl_session_is_left_alone(env, monkeypatch):
    _set(env, monkeypatch, {"DISPLAY": ":0", "WAYLAND_DISPLAY": "wayland-0"})
    _prefer_x11_under_wsl()
    assert "XDG_SESSION_TYPE" not in env


def test_no_x_server_means_nothing_to_switch_to(env, monkeypatch):
    _set(env, monkeypatch, {"WSL_DISTRO_NAME": "Ubuntu-24.04"})
    _prefer_x11_under_wsl()
    assert "XDG_SESSION_TYPE" not in env
