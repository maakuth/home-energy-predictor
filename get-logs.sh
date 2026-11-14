#!/bin/sh

set +x

since="7 days ago"

journalctl --user -uhome-energy-predictor-15min.service -S"$since" > logs/recentlog.txt
journalctl --user -u home-energy-predictor-minutely.service -S"$since" > logs/recentlog-often.txt
journalctl --user -u home-energy-predictor-weekly.service -S"$since"> logs/recentlog-weekly.txt
journalctl --user -u hepo-measurement-collector.service -u hepo-measurement-collector.timer -S"$since" > logs/recentlog-collector.txt
