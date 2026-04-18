# Autonomous Spectroscopy with MCP

Prototype of a **fully autonomous experimental workflow** for **diamond color center spectral microscopy** using the **Model Context Protocol (MCP)**.

Experiments involving diamond color centers require precise coordination between multiple hardware components—lasers, optical elements, spectrometers, and camera-based detectors. These experiments are often run manually, which is time-consuming, error-prone, and limits experimental throughput.

This project explores how a unified software interface can orchestrate the entire experimental pipeline automatically. The system integrates hardware control, experiment sequencing, data acquisition, and result validation into a single MCP-based framework capable of executing a complete measurement run **without human intervention**.

---

## Goals

- Demonstrate **end-to-end experimental autonomy**
- Provide a **unified MCP interface** for multiple laboratory devices
- Build a **reliable orchestration layer** for experiment execution
- Enable **repeatable and scalable spectroscopy measurements**

---

## Core Features

- **MCP Hardware Interfaces**
  - Cobolt laser control
  - Optical component configuration
  - Spectrometer operation
  - Camera-based detection
  - Power meter monitoring

- **Experiment Orchestration**
  - Automated measurement sequencing
  - Parameter configuration
  - Device synchronization

- **Safety and Reliability**
  - Hardware initialization routines
  - Safety checks and interlocks
  - Error handling and recovery

- **Autonomous Data Acquisition**
  - Automated experiment execution
  - Data capture and storage
  - Basic result validation

---

## Power meter (Thorlabs PM) and VISA

The optical power meter is controlled through **PyVISA** (see `pyproject.toml`).  For real hardware access, the host also needs a **VISA runtime**—typically **NI-VISA** on Windows—installed separately from Python.  PyVISA uses that stack to list resources (USB, LAN, etc.) and open instrument sessions.  Without it, use `PM_SIMULATE=1` for an in-process stub.  Set `PM_VISA_ADDRESS` when auto-detection is not appropriate.

**NI-VISA vs Thorlabs Optical Power Monitor:** These stacks can **interfere** with each other—installing or reinstalling one may break the other.  If you install NI-VISA for PyVISA, avoid reinstalling Thorlabs Optical Power Monitor afterward to “fix” a broken setup

---

## System Architecture

The system is designed around MCP’s **client-server model**:

- **MCP Servers**
  - Hardware wrappers exposing device control as MCP tools

- **Orchestrator / Run Engine**
  - Coordinates experiment execution
  - Calls MCP tools to configure and control devices
  - Handles experiment logic and workflow automation

This architecture enables modular integration of new hardware and supports scalable automation of future experiments.

---

## Expected Outcome

The final prototype will demonstrate a **fully autonomous spectroscopy measurement run**, serving as a proof-of-concept for automation in experimental quantum optics workflows.
