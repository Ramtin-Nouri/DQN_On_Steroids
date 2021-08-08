from nets import DynamicsNetwork,QNetwork, StateNetwork
import TF2_Keras_Template as template
import gym,random,numpy as np, os
from collections import deque
from tqdm import tqdm
import tensorflow as tf
random.seed(459)


totalSteps = 2000
totalEpisodes = 10000
learningRate = 0.01
lrdecay = 0.01
gamma = 0.95
batchSize = 64

epsilon = 0.9
min_epsilon = 0.01
max_epsilon = 0.9
decay = 0.01

env=gym.make("Breakout-v4")
env.seed(459)


encoder = StateNetwork.NeuralNetwork()
encoderModel,_ = encoder.getModel((208,160,3),(208,160,3)) #original size is 210 but that's not divisible by 8


net = QNetwork.NeuralNetwork()
model,epoch = net.getModel((208,160,3),[env.action_space.n,encoderModel,learningRate,lrdecay])

#Get Loggers
logger = template.Logger("savedata/agent/",model)
callbacks = logger.getCallbacks(period=20,predict=False)

file_writer = tf.summary.create_file_writer(logger.folderName)
file_writer.set_as_default()


def getAction(state):
    exploit= random.uniform(0,1)
    if exploit > epsilon:
        prediction = model.predict(np.array([state]))[0]
        return prediction.argmax()
    else:
        return env.action_space.sample()

def replay():
    minibatch = random.sample(memory,batchSize)
    states = []
    targets = []
    for state, action, reward, nextState, done in minibatch:
        states.append(state)
        if done:
            target = reward
        else:
            target = reward + gamma*np.amax(model.predict(np.array([nextState])))
        target_f = model.predict(np.array([state]))[0]
        target_f[action]=target
        targets.append(target_f)
    model.train_on_batch(np.array(states),np.array(targets))


def runOnce():
    score = 0
    state = env.reset()[2:]
    for _ in tqdm(range(totalSteps)):
        action = getAction(state)
        newState, reward, done, _ = env.step(action)
        newState = newState[2:]
        
        score += reward
            
        memory.append((state,action,reward,newState,done))
        if done:
            break
        if len(memory)>=batchSize:
            replay()
        
        state = newState
    return score
    
   
    
memory = deque(maxlen=1000)
highScore = 0
scoreQue = deque(maxlen=100)

for episode in range(totalEpisodes):
    score = runOnce()
    if score > highScore:
        highScore = score
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay*episode)
    scoreQue.append(score)
    rm = np.mean(scoreQue)
    print("Episode: ",episode, "| Score: " ,score,"| HighScore: ",highScore, "| Epsilon: ", epsilon, "Running Mean: ", np.mean(scoreQue))
    tf.summary.scalar("Score",score,step=episode)
    tf.summary.scalar("Highscore",highScore,step=episode)
    tf.summary.scalar("Epsilon",epsilon,step=episode)
    tf.summary.scalar("Running Mean",np.mean(scoreQue),step=episode)