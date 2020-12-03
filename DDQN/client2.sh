
conda activate tensorflow
xvfb-run -a -e /dev/stdout -s '-screen 0 1400x900x24' $MALMO_MINECRAFT_ROOT/launchClient.sh -port 10001 > outclient2;


