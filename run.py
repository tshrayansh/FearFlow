"""
Top-level execution script for the Zebrafish Alarm Contagion ABM.
Runs the complete simulation campaign, hypothesis testing, and generates all publication figures.
Usage:
    python3 run.py
"""

from zebrafish_alarm_abm.run_experiments import main

if __name__ == "__main__":
    main()
