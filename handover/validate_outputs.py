"""Post-deployment output validation for the receiving team.

Run from the repository root after the daily pipeline::

    python handover/validate_outputs.py

It checks the input files against handover/input_schema.yaml and every output against the acceptance criteria in
handover/acceptance_criteria.md, writes outputs/validation_report.csv and exits with code 1 on any error.
"""

from gasrisk.validation import main

if __name__ == "__main__":
    main()
