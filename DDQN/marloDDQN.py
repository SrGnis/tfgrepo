import gym
import time
import marlo
import numpy as np
import tensorflow as tf
import math
import random
import time
import datetime
import pickle
from enum import Enum
import matplotlib.pyplot as plt
from lxml import etree
import logging

frame_size = [60,60]
num_input = frame_size[0] * frame_size[1]

client_pool = [('127.0.0.1', 10000),('127.0.0.1', 10001)]
join_tokens = marlo.make("MarLo-MobchaseTrain1-v0",
        params=dict(
            client_pool=client_pool,
            agent_names=["MarLo-Agent-0", "MarLo-Agent-1"],
            videoResolution=frame_size,
            kill_clients_after_num_rounds=500,
            forceWorldReset=False,
            max_retries=500,
            retry_sleep=0.1,
            step_sleep=0.1,
            prioritise_offscreen_rendering=False,
            suppress_info= False
        ))
assert len(join_tokens) == 2

class Convolutional():
    def __init__(self,num_classes,learningRate):
        self.x = tf.placeholder("float", [None, num_input])

        #Reshape the flatten data
        self.input_layer = tf.reshape(self.x, [-1, frame_size[1], frame_size[0], 1])
    
        #Convolutional Layer 1
        self.conv1 = tf.layers.conv2d(
            inputs = self.input_layer,
            filters = 32,
            kernel_size = [6, 6],
            strides=[2, 2],
            padding = "valid",
            activation = tf.nn.relu)
        #Output size = 28

        #Convolutional Layer 2
        self.conv2 = tf.layers.conv2d(
            inputs = self.conv1,
            filters = 64,           
            kernel_size = [6, 6],
            strides = [2, 2],
            padding = "valid",
            activation = tf.nn.relu)
        #Output size = 12

        #Convolutional Layer 3
        self.conv3 = tf.layers.conv2d(
            inputs = self.conv2,
            filters = 64,
            kernel_size = [4, 4],
            strides = [2, 2],
            padding = "valid",
            activation = tf.nn.relu)
        #Output size = 5

        #Flatten the data to pass it through the feed forward
        self.dims = self.conv3.get_shape().as_list()
        self.final_dimension = self.dims[1] * self.dims[2] * self.dims[3]
        self.conv3_flat = tf.reshape(self.conv3, [-1, self.final_dimension])
        
        #Feed Forward
        self.dense = tf.layers.dense(inputs = self.conv3_flat, units = 512, activation = tf.nn.relu)
        
        self.Qout = tf.layers.dense(inputs = self.dense, units = num_classes)
        
        #Indexes of the actions the network shall take
        self.prediction = tf.argmax(self.Qout, 1)
        
        self.actions = tf.placeholder(shape = [None], dtype = tf.int32)
        #Multiply our Q-values by a OneHotEncoding to only take the chosen ones.
        self.actions_onehot = tf.one_hot(self.actions, num_classes, dtype = tf.float32)
        #So that Q's going to be the Q-values calculated by the Target network
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis = 1)

        #NextQ corresponds to the Q estimated by the Bellman Equation
        self.nextQ = tf.placeholder(shape = [None], dtype = tf.float32)
                
        #The loss value coresponds to the difference between the two different Q-values estimated
        self.loss = tf.reduce_mean(tf.square(self.nextQ - self.Q))
    
        #Let's print the important informations
        self.merged = tf.summary.merge([tf.summary.histogram("nextQ", self.nextQ),
                                        tf.summary.histogram("Q", self.Q),
                                        tf.summary.scalar("Loss", self.loss)])
        
        self.learningRate = learningRate
        #We would prefer the Adam Optimizer
        self.trainer = tf.train.AdamOptimizer(learning_rate = self.learningRate)
        self.updateModel = self.trainer.minimize(self.loss)

class ChooseNetwork(Enum):
    Convolutional = Convolutional

class TensorBoardInfosLogger():
    def __init__(self):
        self.percent_win = tf.placeholder(dtype=tf.float32)
        self.mean_j_by_win = tf.placeholder(dtype=tf.float32)
        self.mean_rewards_sum = tf.placeholder(dtype=tf.float32)
        self.merged = tf.summary.merge([tf.summary.scalar("Percent_of_win_on_last_50_episodes", self.percent_win), 
                                        tf.summary.scalar("Number_of_steps_by_win_on_last_50_episodes", self.mean_j_by_win),
                                        tf.summary.scalar("Mean_of_sum_of_rewards_on_last_50_episodes", self.mean_rewards_sum), ])

def processState(state):
    gray_state = np.dot(state[...,:3], [0.299, 0.587, 0.114]) #Downscale input to greyscale
    return np.reshape(gray_state, num_input)/255.0 #Normalize pixels

def reverse_processState(state):
    return np.reshape(state, (frame_size[0], frame_size[1]))*255.0

def get_stacked_states(episode_frames, trace_length): #Fills the stacked frames with images full of zero if the sequence is too short
    if len(episode_frames) < trace_length:
        nb_missing_states = trace_length - len(episode_frames)
        zeros_padded_states = [np.zeros(num_input) for _ in range(nb_missing_states)]
        zeros_padded_states.extend(episode_frames)
        return np.reshape(np.array(zeros_padded_states), num_input*trace_length)
    else:
        return np.reshape(np.array(episode_frames[-trace_length:]), num_input*trace_length)

def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]): #Get the weights of the original network
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value()))) #Update the Target Network weights
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)

class experience_buffer():
    def __init__(self, buffer_size = 200000): #Stores steps
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])
    
    def get(self):
        return np.reshape(np.array(self.buffer), [len(self.buffer), 5])

def print_debug_states(tf_session, QNet, raw_input, trace_length):
    tmp = tf_session.run(QNet.input_layer, feed_dict={QNet.x:[raw_input]})
    for depth in range(tmp.shape[3]):
        print("## Input image n°" + str(depth) + " ##")
        plt.imshow(tmp[0, :, :, depth], cmap=plt.get_cmap('gray'))
        plt.show()

@marlo.threaded
def run(join_token, agentn):
    
    env = marlo.init(join_token)
    graph1 = tf.Graph()
    NetType = ChooseNetwork.Convolutional
    learningRate = 0.001
    num_nodes = 256
    num_classes = len(env.action_names[0])-1

    batch_size =  32 #How many steps to use for each training step.
    trace_length = 4 #How long each experience trace will be when training

    myBuffer = experience_buffer()

    update_freq = 4 #How often to perform a training step.
    num_episodes = 20001 #How many episodes of game environment to train network with
    total_steps = 0
    rList = [] #List of our rewards gained by game
    jList = [] #Number of moves realised by game
    j_by_loss = [] #Number of moves before resulting with a death of the agent
    j_by_win = [] #Number of moves before resulting with a win of the agent
    j_by_nothing = [] #This list's going to be used to count how many times the agent moves until the limit of moves is reached
    y = .95 #Discount factor on the target Q-values

    pre_train_steps = 5000 #How many episodes of random actions before training begins.
    startE = 1 #Starting chance of random action
    endE = 0.1 #Final chance of random action
    annealing_steps = 10000 #How many epsiodes of training to reduce startE towards endE.
    e = startE
    stepDrop = (startE - endE) / annealing_steps
    nb_win = 0
    nb_win_tb = 0
    nb_nothing = 0
    nb_loss = 0
    tau = 0.001
    load_model = True

    # Creating an explicit and unique title for Tensorboard
    date = str(time.time()).replace(".","")
    net = str(NetType).split(".")[1]
    bs = "BatchSize-" + str(batch_size)
    strlr = "lr-" + str(learningRate)
    rand_step = "RandStep-" + str(pre_train_steps)
    nb_to_reduce_e = "ReducE-" + str(annealing_steps)
    write_path = "train/" + net + "_" + bs + "_" + strlr + "_" + rand_step + "_" + nb_to_reduce_e + "_" + date[-5:] + str(agentn)

    # DEBUGGING
    is_debug = False
    if is_debug:
        write_path = 'train/test'


    #TRAIN
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.reset_default_graph()
    with tf.Session(config=config) as sess:
        
        #Use a Double Q-Network
        mainQN = NetType.value(num_classes,learningRate)
        targetQN = NetType.value(num_classes,learningRate)
        
        trainables = tf.trainable_variables()
        targetOps = updateTargetGraph(trainables,tau)
        
        #Save the network
        saver = tf.train.Saver()
        path_to_save = "saves/" + "final" + str(agentn) + "/"
        
        init = tf.global_variables_initializer()
        sess.run(init)
        
        tb_infos_logger = TensorBoardInfosLogger()
        writer = tf.summary.FileWriter(write_path)
        
        i = 0
        
        if load_model == True:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(path_to_save)
            saver.restore(sess,ckpt.model_checkpoint_path)
            resbuf = pickle.load( open( path_to_save + str(7000) + ".pickle", "rb" ) )
            e = resbuf["epsilon"]
            total_steps = resbuf["Total_steps"]
            myBuffer = resbuf["Buffer"]
            rList = resbuf["rList"]
            i = resbuf["Num Episodes"]
            i += 1
            jList = resbuf["jList"]
            
        while i <= num_episodes:
            
            print(agentn," EPISODIO: ", i)
            if(agentn == 1):
                time.sleep(3)
				
            episodeBuffer = experience_buffer()
            #print(agentn," 1")
            s = env.reset()
            #print(agentn," 2")
            s = processState(s)
            d = False
            j = 0
            episode_frames = []
            episode_frames.append(s)
            episode_qvalues = []
            episode_rewards = []
            
            moves = []
            #print(agentn," EMPEZANDO EPISODIO")
            
            ### Epsilon Greedy ###
            if i > pre_train_steps:
                if e > endE:
                    e -= stepDrop
                # Make full exploration before the number of pre-train episodes then play with an e chance of random action during the training (e-greedy)
            
            while not d:
                j += 1
                #print(agentn," 2")

                #print(agentn," 3")
                if(np.random.rand(1) < e or total_steps < pre_train_steps):
                    #index_action_predicted = env.action_space.sample()
                    index_action_predicted = random.randint(0,3)
                    episode_qvalues.append([1 if i == index_action_predicted else 0 for i in range(len(env.action_names[0])-1)])
                else:
                    if is_debug:
                        print_debug_states(sess, mainQN, s, trace_length)
                        
                    prediction, qvalues = sess.run([mainQN.prediction, mainQN.Qout], \
                                                        feed_dict = {mainQN.x:[s]})
                    index_action_predicted = prediction[0]
                    episode_qvalues.append(qvalues[0])

                #Get new state and reward from environment
                # poner contador por si se ralla todo
                #print(agentn," 4")
                contador = 0
                while True:
                    s1_raw, r, d, info = env.step(index_action_predicted+1)
                    contador += 1
                    #print(agentn, " " ,info)
                    if r != 0:
                        break
                    elif info != None:
                        if info == "caught_the_Chicken":
                            r += 1
                            print("SE HA HARCODEADO LA PUNTUACION ", d, " ", info)
                            break
                            
                        if info == "Agent0_defaulted":
                            break
                        
                        if info == "Agent1_defaulted":
                            break
                    elif contador >= 100:
                        print("SE HA TARDADO MUCHO EN REALIZAR LA ACCION")
                        break
                #print(agentn," 5")
                s1 = processState(s1_raw)
                moves.append(index_action_predicted)
                print(agentn, "accion: ", index_action_predicted, "done: ", d)
                episodeBuffer.add(np.reshape(np.array([s, index_action_predicted, r, s1, d]), [1, 5]))
                episode_frames.append(s1)
                

                if total_steps > pre_train_steps:
                    if total_steps % (update_freq) == 0:
                        
                        updateTarget(targetOps,sess) #Update Target Network
                        
                        trainBatch = myBuffer.sample(batch_size) #Get a random batch of experiences.
                        
                        if is_debug:
                            print_debug_states(sess, mainQN, trainBatch[0, 0], trace_length)
                    
                        #Estimate the action to choose by our first network
                        actionChosen = sess.run(mainQN.prediction, \
                                                feed_dict = {mainQN.x:np.vstack(trainBatch[:, 3])})
                        #Estimate all the Q-values by our second network --> Double
                        allQValues = sess.run(targetQN.Qout, \
                                                feed_dict = {targetQN.x:np.vstack(trainBatch[:, 3])})

                        #Train our network using target and predicted Q-values
                        end_multiplier = -(trainBatch[:, 4] -1)
                        maxQ = allQValues[range(batch_size), actionChosen]
                        #Bellman Equation
                        targetQ = trainBatch[:, 2] + (y * maxQ * end_multiplier)

                        _, summaryPlot = sess.run([mainQN.updateModel, mainQN.merged], \
                                                    feed_dict = {mainQN.x:np.vstack(trainBatch[:, 0]), \
                                                                mainQN.nextQ:targetQ, \
                                                                mainQN.actions:trainBatch[:, 1]})
                            
                        writer.add_summary(summaryPlot, total_steps)  
                #print(agentn," 6")        
                episode_rewards.append(r)
                #if (s == s1).all():
                    #print("State error : State did not changed though the action was " + env.action_names[0][index_action_predicted])
                
                s = s1
                total_steps += 1
                    
                if d == True:
                    print("AGENTE ", agentn, " REWARD ", r)
                    print("AGENTE ", agentn, " PASOS ", j)
                    if r == 0 or r == 10 or r == -10:
                        print("Unrecognized reward Error : " + str(r))
                        j_by_loss.append(j)
                    elif r > 0:
                        j_by_win.append(j)
                    elif r < 0:
                        j_by_nothing.append(j)                    
                    break
                      
            myBuffer.add(episodeBuffer.buffer)
            jList.append(j)
            rList.append(sum(episode_rewards))
            rewards = np.array(rList)
            
            if i % (50) == 0:
                nb_of_win_on_last_50 = (len(j_by_win) - nb_win_tb)
                win_perc = nb_of_win_on_last_50/50*100
                mean_j_by_win = np.mean(j_by_win[-nb_of_win_on_last_50:])
                mean_rewards_sum = np.mean(rList[-50:])
                summaryPlot = sess.run(tb_infos_logger.merged, 
                                       feed_dict = {tb_infos_logger.percent_win: win_perc, \
                                                    tb_infos_logger.mean_j_by_win: mean_j_by_win, \
                                                    tb_infos_logger.mean_rewards_sum: mean_rewards_sum})                       
                writer.add_summary(summaryPlot, i)  
                nb_win_tb = len(j_by_win)
            
            if i % (500) == 0:
                print("#######################################")
                print("% Win " + str(agentn) + " : " + str((len(j_by_win) - nb_win)/5) + "%")
                print("% Nothing " + str(agentn) + " : " + str((len(j_by_nothing) - nb_nothing)/5) + "%")
                print("% Loss " + str(agentn) + " : " + str((len(j_by_loss) - nb_loss)/5) + "%")
                
                print("Nb J before win " + str(agentn) + " : " + str(np.mean(j_by_win[-(len(j_by_win) - nb_win):])))
                print("Nb J before die " + str(agentn) + " : " + str(np.mean(j_by_loss[-(len(j_by_loss) - nb_loss):])))
                      
                print("Total Steps " + str(agentn) + " : " + str(total_steps))
                print("I " + str(agentn) + " : " + str(i))
                print("Epsilon " + str(agentn) + " : ", str(e))
                      
                nb_win = len(j_by_win)
                nb_nothing = len(j_by_nothing)
                nb_loss = len(j_by_loss)
                    
            if i % (1000)== 0 and i != 0:
                print("-------------------Guardando----------------------")
                #Save all the other important values
                saver.save(sess, path_to_save + str(i) + '.ckpt')
                with open(path_to_save + str(i) + ".pickle", 'wb') as file:
                    dictionnary = {
                        "epsilon": e,
                        "Total_steps": total_steps,
                        "Buffer": myBuffer,
                        "rList": rList,
                        "Num Episodes": i,
                        "jList": jList}
                    
                    pickle.dump(dictionnary, file, protocol = pickle.HIGHEST_PROTOCOL)
                        
            i += 1
        
        saver.save(sess, path_to_save + str(i) + '.ckpt')

# Run agent-0
thread_handler_0, _ = run(join_tokens[0],0)
# Run agent-1
thread_handler_1, _ = run(join_tokens[1],1)

# Wait for both the threads to complete execution
thread_handler_0.join()
thread_handler_1.join()

