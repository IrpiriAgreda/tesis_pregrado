#!/bin/bash
python "$SUMO_HOME/tools/randomTrips.py" -n osm.net.xml --seed 42 -p 10 -o osm.passenger.trips.xml -e 86400 --vehicle-class passenger --vclass passenger --prefix veh --lanes --validate
python "$SUMO_HOME/tools/randomTrips.py" -n osm.net.xml --seed 42 -p 10 -o osm.truck.trips.xml -e 86400 --vehicle-class truck --vclass truck --prefix truck  --validate
