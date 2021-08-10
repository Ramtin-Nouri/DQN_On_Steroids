from nets import QNetwork
import TF2_Keras_Template as template
import gym,random,numpy as np, sys
from collections import deque
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import load_model,Model
random.seed(459)

maxQueueSize=10000
totalSteps = 2000
totalEpisodes = 10000
learningRate = 0.01
lrdecay = 0.01
gamma = 0.95
batchSize = 2048

epsilon = 1.0
min_epsilon = 0.01
max_epsilon = 1.0
decay = 0.01

env=gym.make("Breakout-v4")
env.seed(459)

#Get Encoder Model
try:
    autoencoderModel = load_model(sys.argv[1],compile=False)
except:
    print("ERROR: Please specify Path to encoder model as first parameter")
    quit()
layer_name = 'encoding'
encoder = Model(inputs=autoencoderModel.layers[0].input,
                                outputs=autoencoderModel.get_layer(layer_name).output)

            


net = QNetwork.NeuralNetwork()
model,epoch = net.getModel(encoder.output_shape[1:],[env.action_space.n,learningRate,lrdecay])

#Get Loggers
logger = template.Logger("savedata/agent/",model)
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


def runEpisode():
    score = 0
    obs = env.reset()[2:]
    state = encoder.predict(np.array([obs]))[0]
    for _ in tqdm(range(totalSteps)):
        action = getAction(state)
        obs, reward, done, _ = env.step(action)
        obs = np.array([obs[2:]])
        newState = encoder.predict(obs)[0]
        
        score += reward
            
        memory.append((state,action,reward,newState,done))
        if done:
            break
        if len(memory)>=batchSize:
            replay()
        
        state = newState
    return score
    
   
    
memory = deque(maxlen=maxQueueSize)
highScore = 0
scoreQue = deque(maxlen=100)

for episode in range(totalEpisodes):
    score = runEpisode()
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