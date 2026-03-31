# home-energy-predictor notes for agents
- This is a python app that tries to predict home energy usage by an ML model
- Implement changes using test-driven development: first add failing test, do changes, observe test passing
- Use python virtualenv to run tests: .venv/bin/python3 -m pytest
- DON'T do any changes to database or home assistant without explicit permission. The machine running agent probably doesn't even have access to these.
- Offer to save your work to git frequently
- Don't do heredoc hacks to modify files. If there's something preventing file modification, say so and the user will help.
- The development likely isn't running on the machine that has connectivity to the HA and psql, so don't bother trying to run it against them 
