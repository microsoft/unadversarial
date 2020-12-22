if [ $1 = "bus" ]
then
    PYTHONPATH=src/better_corruptions:$PYTHONPATH python -m src.main \
        --custom-file /src/resources/bus.ply \
        --json-config $2 --single-class 654 --out-dir $3
elif [ $1 = "warplane" ]
then
    PYTHONPATH=src/better_corruptions:$PYTHONPATH python -m src.main \
        --custom-file /src/resources/warplane_highpoly.ply \
        --json-config $2 --single-class 895 --out-dir $3
elif [ $1 = "ship" ]
then
    PYTHONPATH=src/better_corruptions:$PYTHONPATH python -m src.main \
        --custom-file /src/resources/ship.ply \
        --single-class 510 --json-config $2 --out-dir $3
elif [ $1 = "truck" ]
then
    PYTHONPATH=src/better_corruptions:$PYTHONPATH python -m src.main \
        --custom-file /src/resources/truck.ply \
        --single-class 867 --json-config $2 --out-dir $3
elif [ $1 = "car" ]
then
    PYTHONPATH=src/better_corruptions:$PYTHONPATH python -m src.main \
        --custom-file /src/resources/car.ply \
        --single-class 817 --json-config $2 --out-dir $3
fi
