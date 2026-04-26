"""Fallback setup.py for older pip / editable installs.

Modern installers should use pyproject.toml directly. This file exists for
compatibility with pip<21.3.
"""
from setuptools import setup

setup()
