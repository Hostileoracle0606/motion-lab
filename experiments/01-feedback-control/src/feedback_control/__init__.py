"""Feedback controllers for Motion Lab Experiment 01.

Simulation components stay in their own module so importing the controller API
does not require importing MuJoCo.
"""

from feedback_control.controllers import (
    ControllerOutput,
    pd_control,
    proportional_control,
)

__all__ = [
    "ControllerOutput",
    "pd_control",
    "proportional_control",
]
