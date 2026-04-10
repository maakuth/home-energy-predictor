#!/bin/sh

echo Pulling runtime files from murrikka

for file in future_predictions.json energy_model.json hepo.db processed_data.csv optimization_plan.json; do
	rsync -vt murrikka:src/home-energy-predictor/$file .
done

