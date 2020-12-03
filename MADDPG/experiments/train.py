import argparse
import numpy as np, numpy.random
import tensorflow as tf
import time
import pickle
import marlo
import gym
import copy
from random import random

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers

epi_per_iter = 10

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=10, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1000, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=32, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./saves/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=100, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="./saves/", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def get_trainers(num_adversaries, obs_shape_n, action_space, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(2):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, action_space, i, arglist,
            local_q_func=(arglist.adv_policy=='ddpg')))
    return trainers

def sumtuple(n):
    return sum(list(n))

@marlo.threaded
def reiniciar(env):
    obs = env.reset()
    return obs

def train(arglist):

############################################

    @marlo.threaded
    def funcion(env, action, agent_num):

        contador = 0
        while True: # Ejecutamos la accion evitando errores
            _ , r, done, info, new_obs = env.step(np.argmax(action)+1)
            new_obs = new_obs['observation']
            if new_obs == None:
                new_obs = last_obs[agent_num]
            else:
                new_obs = [new_obs.get('XPos'), new_obs.get('ZPos'), new_obs.get('Yaw')]
            contador += 1
            if r != 0:
                break
            elif info != None:
                if "caught_the_Chicken" in info:
                    r += 1
                    print("SE HA HARCODEADO LA PUNTUACION ", done, " ", info)
                    break

                if "Agent0_defaulted" in info:
                    r = -0.02
                    break

                if "Agent1_defaulted" in info:
                    r = -0.02
                    break
                    
            elif contador >= 100:
                print("SE HA TARDADO MUCHO EN REALIZAR LA ACCION")
                break
        return new_obs, r, done, info

#######################################################

    with U.single_threaded_session():

        # Create environment
        
        client_pool = [('127.0.0.1', 10000),('127.0.0.1', 10001)]
        join_tokens = marlo.make("MarLo-MobchaseTrain1-v0",
            params=dict(
                client_pool=client_pool,
                agent_names=["MarLo-Agent-0", "MarLo-Agent-1"],
                videoResolution=[64,64],
                kill_clients_after_num_rounds=500,
                forceWorldReset=False,
                max_retries=500,
                retry_sleep=0.1,
                step_sleep=0.1,
                prioritise_offscreen_rendering=False,
                suppress_info= False
            ))
        assert len(join_tokens) == 2
        
        # Create agent trainers
        #obs_shape_n = [(64,64,3,),(64,64,3,)]
        observation_space = [gym.spaces.Box(low=-np.inf, high=+np.inf, shape=(6,), dtype=np.float32), gym.spaces.Box(low=-np.inf, high=+np.inf, shape=(6,), dtype=np.float32)]
        obs_shape_n = [observation_space[i].shape for i in range(2)]
        action_space = [gym.spaces.Discrete(4), gym.spaces.Discrete(4)]
        num_adversaries = 0
        trainers = get_trainers(num_adversaries, obs_shape_n, action_space, arglist)
        
        # Initialize
        U.initialize()
        
        epis_trans = 0
        epsilon = 0.0

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.restore:
            print('Loading previous state...')
            resbuf = pickle.load( open( "./saves/losbuffers.p", "rb" ) )
            epis_trans = resbuf[2]
            epsilon = resbuf[3]
            U.load_state(arglist.load_dir+str(epis_trans))
            trainers[0].replay_buffer = resbuf[0]
            trainers[1].replay_buffer = resbuf[1]
            

        episode_rewards = []
        agent_rewards = [[] for _ in range(2)] # lista de sumas de las recompensas de cada episodio
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        saver = tf.train.Saver()
        t_start = time.time()

        #inicial0 = [1.5, 2.5, 270, 5.5, 6.5, 180]
        #inicial1 = [5.5, 6.5, 180, 1.5, 2.5, 270]
        inicial0 = [1.5, 2.5, 270, 3.5, 4.5, 180]
        inicial1 = [3.5, 4.5, 180, 1.5, 2.5, 270]


        while True:
            #NEW

            last_obs = []

            agent_rewards[0].append(0)
            agent_rewards[1].append(0)

            env0 = marlo.init(join_tokens[0])
            env1 = marlo.init(join_tokens[1])

            # Run agent-0
            agent_thread_0, res0 = reiniciar(env0)
            # Run agent-1
            agent_thread_1, res1 = reiniciar(env1)

            obs0 = res0.get()
            obs1 = res1.get()

            obs0 = inicial0
            obs1 = inicial1

            done0 = False
            done1 = False

            num_eps = 0

            #Ejecutar 10 episodios
            while True:

                if(random() > epsilon):
                    action0 = trainers[0].action(np.array(obs0))  # se obtine la accion que ejecuta la politica
                else:
                    action0 = np.random.dirichlet(np.ones(4),size=1)[0]
                    
                if(random() > epsilon):
                    action1 = trainers[1].action(np.array(obs0))  # se obtine la accion que ejecuta la politica
                else:
                    action1 = np.random.dirichlet(np.ones(4),size=1)[0]
                #print("Estan dentro")
                # Run agent-0
                agent_thread_0, resul0 = funcion(env0, action0, 0)
                # Run agent-1
                agent_thread_1, resul1 = funcion(env1, action1, 1)

                # Wait for both the threads to complete execution
                agent_thread_0.join()
                #print("Esta fuera 1")
                agent_thread_1.join()
                #print("Estan fuera")

                nob0, r0, done0, i0 = resul0.get()
                nob1, r1, done1, i1 = resul1.get()

                last_obs = [copy.deepcopy(nob0),copy.deepcopy(nob1)]

                # Las nuevas observciones
                varhelp = copy.deepcopy(nob0)
                nob0.extend(nob1)
                nob1.extend(varhelp)

                #print("ESTAS SON LAS OBSERVACIONES")
                #print(nob0)
                #print(nob1)

                trainers[0].experience(np.array(obs0), action0, r0, np.array(nob0), done0, False)
                trainers[1].experience(np.array(obs1), action1, r1, np.array(nob1), done1, False)

                agent_rewards[0][-1] += r0
                agent_rewards[1][-1] += r1

                obs0 = nob0
                obs1 = nob1

                if done0 or done1:
                    print("EPISODIO NUMERO:", num_eps)
                    # Run agent-0
                    agent_thread_0, res0 = reiniciar(env0)
                    # Run agent-1
                    agent_thread_1, res1 = reiniciar(env1)

                    obs0 = res0.get()
                    obs1 = res1.get()
                    obs0 = inicial0
                    obs1 = inicial1
                    done0 = False
                    done1 = False
                    num_eps += 1
                    
                    loss = None
                    for agent in trainers:
                        agent.preupdate()
                    for agent in trainers:
                        loss = agent.update(trainers)
                        print("LA LOSS", loss)
                    
                    if num_eps % epi_per_iter == 0:
                        break
                    agent_rewards[0].append(0)
                    agent_rewards[1].append(0)
                    
            
            #Fin de ejecutar 10 episodios
            print("FIN DEL SAMPLE")

            # Se obtiene una lista de tuplas que contienen las rewards de los agentes emparejadas por episodios utilizadno los ultimos episodios generados en la iteracion
            # A estas tuplas se transforman a listas y se aplica sum()
            # El resultado de esto se coloca al final de episode_rewards
            #  
            # En resumen: se suman las ultimas rewards de los agentes por episodios y se añaden a la lista
            episode_rewards.extend( list(map(sumtuple, list(zip(agent_rewards[0][epis_trans:], agent_rewards[1][epis_trans:])))))

            epis_trans += 10
            if epsilon > 0.1:
                epsilon -= 0.002
            
            print("TOTAL DE EPISODIOS TRANSCURRIDOS: ", epis_trans, " Epsilon: ", epsilon)

            # update all trainers, if not in display or benchmark mode


            # save model, display training output
            if (epis_trans % arglist.save_rate == 0):
                U.save_state(arglist.save_dir + str(epis_trans), saver=saver)
                losbuffers = [trainers[0].replay_buffer, trainers[1].replay_buffer, epis_trans, epsilon]
                pickle.dump( losbuffers, open( "./saves/losbuffers" + str(epis_trans) + ".p", "wb" ) )
                pickle.dump( losbuffers, open( "./saves/losbuffers.p", "wb" ) )
            if (epis_trans % 1000 == 0):
                break

if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
    epi_per_iter = 10
    