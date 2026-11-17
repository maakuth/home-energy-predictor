#!/bin/bash

# The scheduled SARIMA benchmark is disabled. Historical artifacts and the
# standalone tools remain available for explicit offline analysis.
if [ "$#" -ne 0 ]; then
  echo "SARIMA slow-task execution is disabled; arguments are not accepted." >&2
  exit 2
fi
echo "SARIMA slow-task benchmark is disabled; nothing to run."
