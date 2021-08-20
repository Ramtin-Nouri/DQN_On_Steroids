from threading import Thread
from nets import QNetwork
import TF2_Keras_Template as template
import gym,random,numpy as np, sys,time,datetime,os
from collections import deque
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import load_model,Model
random.seed(459)

maxQueueSize=1000
totalSteps = 2000
totalEpisodes = 10000
learningRate = 0.01
lrdecay = 0.01
gamma = 0.95
batchSize = 4

epsilon = 1.0
min_epsilon = 0.01
max_epsilon = 1.0
decay = 0.01

class Environment():
    """
        Custom Wrapper around OpenAi Gym Environment running observations through an encoder model first
        And other useful functions
    """
    def __init__(self):
        self.gym_env=gym.make("Breakout-v4")
        self.gym_env.seed(459)

        #Get Autoencoder Model:
        try:
            autoencoder = load_model(sys.argv[1],compile=False)
        except:
            print("ERROR: Please specify Path to encoder model as first parameter")
            quit()
        #Extract Encoder part from autoencoder:
        layer_name = 'encoding'
        self.model = Model(inputs=autoencoder.layers[0].input,outputs=autoencoder.get_layer(layer_name).output)

    def getOutputShape(self):
        return self.model.output_shape[1:]
    
    def getActionSpace(self):
        return self.gym_env.action_space.n

    def getSample(self):
        return self.gym_env.action_space.sample()

    def reset(self):
        obs = self.gym_env.reset()
        obs = np.array([obs[2:]])
        return self.model.predict(obs)[0]


    def step(self,action):
        obs, reward, done, _ = self.gym_env.step(action)
        obs = np.array([obs[2:]])
        return self.model.predict(obs)[0], reward, done

            
class Agent(Thread):
    """
        Agent sampling actions and filling the memory.
        It uses a copy of the model that is only updated once in a while
    """

    def __init__(self):
        Thread.__init__(self)
        self.env = Environment()
        self.model = net.getModel(self.env.getOutputShape(),[self.env.getActionSpace(),learningRate,lrdecay])
        self.memory = deque(maxlen=maxQueueSize)

    def getAction(self,state):
        exploit= random.uniform(0,1)
        if exploit > epsilon:
            prediction = self.model.predict(np.array([state]))[0]
            return prediction.argmax()
        else:
            return self.env.getSample()

    def getMemorySample(self):
        return np.array(random.sample(self.memory,batchSize))

    def runEpisode(self):
        score = 0
        state = self.env.reset()
        for _ in tqdm(range(totalSteps)):
            action = self.getAction(state)
            newState, reward, done = self.env.step(action)
            
            score += reward
                
            self.memory.append((state,action,reward,newState,done))
            if done:
                break            
            state = newState
        return score
    
    def run(self):
        highScore = 0
        scoreQue = deque(maxlen=100)
        for episode in range(totalEpisodes):
            score = self.runEpisode()
            if score > highScore:
                highScore = score
            epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay*episode)
            scoreQue.append(score)
            print("Episode: ",episode, "| Score: " ,score,"| HighScore: ",highScore, "| Epsilon: ", epsilon, "Running Mean: ", np.mean(scoreQue))
            with file_writer.as_default():
                tf.summary.scalar("Score",score,step=episode)
                tf.summary.scalar("Highscore",highScore,step=episode)
                tf.summary.scalar("Epsilon",epsilon,step=episode)
                tf.summary.scalar("Running Mean",np.mean(scoreQue),step=episode)
                

class Trainer(Thread):
    """
        Samples from Agents memory and trains network
    """
    def __init__(self,agent):
        Thread.__init__(self)
        self.model,self.epoch = net.getModel(agent.env.getOutputShape(),[agent.env.getActionSpace(),learningRate,lrdecay])
        self.agent = agent

    def run(self):
        while True:
            minibatch = self.agent.getMemorySample()
            targets = []
            w,h,c = agent.env.getOutputShape()
            states = np.concatenate(minibatch[:,0]).reshape(batchSize,w,h,c)
            newStates = np.concatenate(minibatch[:,3]).reshape(batchSize,w,h,c)
            qValues = self.model.predict(newStates,batch_size=batchSize )
            allTargets = self.model.predict(states,batch_size=batchSize )
            for i, (_, action, reward, _, done) in enumerate(minibatch):
                if done:
                    target = reward
                else:
                    target = reward + gamma * np.amax(qValues[i])
                allTargets[i][action]=target
                targets.append(allTargets[i])

            loss = self.model.train_on_batch(np.array(states),np.array(targets))
            with file_writer.as_default():
                tf.summary.scalar("Loss",loss,step=self.epoch)
            #print(F"Epoch {self.epoch} ; Loss: {loss}")
            self.epoch +=1


#Get Tensorboard filewriter
today = datetime.datetime.today()
folderName = "%s%04d-%02d-%02d-%02d-%02d-%02d/" % ("savedata/agent/",today.year,today.month,today.day,today.hour,today.minute,today.second)
os.makedirs("%s"%(folderName))

file_writer = tf.summary.create_file_writer(folderName)

net = QNetwork.NeuralNetwork()
agent = Agent()
trainer = Trainer(agent)

with open('%s/architecture.txt'%(folderName),'w') as fh:
    trainer.model.summary(print_fn=lambda x: fh.write(x + '\n'))

agent.start()
while len(agent.memory) < batchSize:time.sleep(1)
trainer.start()