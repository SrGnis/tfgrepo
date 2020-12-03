echo "cliente1"
screen -S pantalla1 -dm bash client1.sh
echo "esperando"
sleep 120
echo "cliente2"
screen -S pantalla2 -dm bash client2.sh
echo "esperando"
sleep 120
echo "entrenado"
python marloDDQN.py >> "log.out"
echo "fin"
sleep 5
screen -XS pantalla1 quit
screen -XS pantalla2 quit
killall java
sleep 5

