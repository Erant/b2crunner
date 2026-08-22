"""Native (non-ComfyUI) execution engine for the Body2COLMAP pipeline.

This package reimplements the orchestration currently done by ComfyUI graphs
(see ../submit.py and ../workflows/) as a standalone Python pipeline: YAML
workflows made of named Steps, executed by a Dispatcher that hides whether a
given step runs in-process, in an isolated subprocess venv, or against a
long-lived model microservice.
"""
