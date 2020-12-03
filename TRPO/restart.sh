while true
do
    xvfb-run -a -e /dev/stdout -s '-screen 0 1400x900x24' $MALMO_MINECRAFT_ROOT/launchClient.sh -port 10000 > outclient1 &
    CLI1_PID=$!
    sleep 120
    xvfb-run -a -e /dev/stdout -s '-screen 0 1400x900x24' $MALMO_MINECRAFT_ROOT/launchClient.sh -port 10001 > outclient2 &
    CLI2_PID=$!
    sleep 120
    python train.py --mynumber 1>> "log.out"
    sleep 5
    ps
    kill $CLI1_PID
    sleep 5
    kill $CLI2_PID
    sleep 5
    killall java
    sleep 5
    ps
done