"""Verify the shared Python environment without opening a GUI window."""

from importlib.metadata import version

import gymnasium
import mujoco
import numpy as np
import stable_baselines3
import tensorboard
import torch


EXPECTED_VERSIONS = {
    "mujoco": "3.11.0",
    "numpy": "2.2.6",
    "gymnasium": "1.3.0",
    "stable-baselines3": "2.9.0",
    "torch": "2.13.0",
    "tensorboard": "2.21.0",
}


def print_versions():
    print("Installed packages:")
    for package, expected in EXPECTED_VERSIONS.items():
        actual = version(package)
        status = "OK" if actual == expected or actual.startswith(f"{expected}+") else "CHECK"
        print(f"  {package:20s} {actual:15s} expected={expected:8s} [{status}]")


def verify_mujoco_step():
    xml = """
    <mujoco model="setup_check">
      <option timestep="0.01"/>
      <worldbody>
        <geom type="plane" size="1 1 0.1"/>
        <body pos="0 0 1">
          <freejoint/>
          <geom type="sphere" size="0.05" mass="0.1"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    initial_height = float(data.qpos[2])
    for _ in range(10):
        mujoco.mj_step(model, data)
    if not data.time > 0.0 or not float(data.qpos[2]) < initial_height:
        raise RuntimeError("MuJoCo step check failed")
    print(f"MuJoCo step check: OK (time={data.time:.2f}, z={data.qpos[2]:.3f})")


def main():
    # These references make missing imports fail before the simulation check.
    _ = (gymnasium, stable_baselines3, tensorboard, torch, np)
    print_versions()
    verify_mujoco_step()
    print("Python environment verification: PASS")


if __name__ == "__main__":
    main()
