import DynamicsNetwork,QNetwork,datamanager
import TF2_Keras_Template as template
import gym,random,numpy as np
from collections import deque
random.seed(459)


totalSteps = 2000
totalEpisodes = 10000
learningRate = 0.01
lrdecay = 0.01
gamma = 0.95
batchSize = 64

epsilon = 1.0
min_epsilon = 0.01
max_epsilon = 1.0
decay = 0.005

env=gym.make("Breakout-v0")
env.seed(459)


encoder = DynamicsNetwork.NeuralNetwork()
encoderModel,_ = encoder.getModel((208,160,3),(208,160,3)) #original size is 210 but that's not divisible by 8


net = QNetwork.NeuralNetwork()
model,epoch = net.getModel((208,160,3),env.action_space.n,[encoderModel,learningRate,lrdecay])

#Get Loggers
logger = template.Logger("savedata/",model)
callbacks = logger.getCallbacks(period=20,predict=False)


def getAction(state):
    exploit= random.uniform(0,1)
    if exploit > epsilon:
        prediction = model.predict(np.array([state]))
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
    state = env.reset()
    for step in range(totalSteps):
        action = getAction(state)
        newState, reward, done, _ = env.step(action)
        
        score += reward
            
        memory.append((state,action,reward,newState,done))
        if done:
            break
        if len(memory)>=batchSize:
            replay()
        
        state = newState
    return score
    
def summaryAndQuit():
    global summary
    for t in summary:
        print(t)
    os._exit(1)
    
    
memory = deque(maxlen=1000)

highScore = -100

scoreQue = deque(maxlen=100)
summary = []
for episode in range(totalEpisodes):
    score = runOnce()
    if score > highScore:
        highScore = score
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay*episode)
    scoreQue.append(score)
    rm = np.mean(scoreQue)
    print("Episode: ",episode, "| Score: " ,score,"| HighScore: ",highScore, "| Epsilon: ", epsilon, "Running Mean: ", np.mean(scoreQue))
    #TODO: use tensorboard instead
    if episode%10==0 and episode !=0:
        summary.append((episode, score, rm))

summaryAndQuit()