"""Demo entrypoint: incident-report flow against the federated global model.

Thin wrapper around `python -m federated_ueba.agent` -- see that module
for the available options.
"""

from federated_ueba.agent.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
