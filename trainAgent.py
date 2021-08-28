"""Implementation of multithreaded double DQN.

It expects an (auto-)encoder model as first argument.
This model is expected to encode the environments observation
to a more dense and meaningful representation.
The dense representation must be the output
of the layer that is named 'encoding'.

If the argument '--dynamic' is given as second argument
the encoder model will be fed with the last four
observations from the environment instead of only the last.
"""
from threading import Thread
from nets import QNetwork
import gym
import random
import numpy as np
import sys
import time
import datetime
import os
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
random.seed(459)

maxQueueSize = 25000
maxSteps = 2000
totalEpisodes = 100000
gamma = 0.99
batchSize = 256

learningRate = 0.0001

min_epsilon = 0.01
max_epsilon = 1.0
decay = 0.0001

targetUpdateRate = 1000


class StaticEnvironment():
    """
    Custom Wrapper around OpenAi Gym Environment.

    It runs observations through an encoder model first.
    And other useful functions.
    """

    def __init__(self):
        """Create Environment and load encoder."""
        self.gym_env = gym.make("BreakoutNoFrameskip-v4")
        self.gym_env.seed(459)

        # Get Autoencoder Model:
        try:
            autoencoder = load_model(sys.argv[1], compile=False)
        except IndexError:
            print("ERROR: \
                Please specify Path to encoder model as first parameter.")
            quit()
        # Extract Encoder part from autoencoder:
        layer_name = 'encoding'
        self.model = Model(
            inputs=autoencoder.layers[0].input,
            outputs=autoencoder.get_layer(layer_name).output)

    def getOutputShape(self):
        """Return shape of encoders output (without batchsize)."""
        return self.model.output_shape[1:]

    def getActionSpace(self):
        """Return number of actions."""
        return self.gym_env.action_space.n

    def getSample(self):
        """Return a randomly sampled action."""
        return self.gym_env.action_space.sample()

    def reset(self):
        """Reset environment and return (encoded) observation."""
        obs = self.gym_env.reset()
        obs = np.array([obs[2:]])
        return self.model.predict(obs)[0]

    def step(self, action):
        """Take a step in envirnoment given an action.

        Arguments:
        ----------
        action: int
            action to take in the environment

        Returns:
        --------
        obs: np.array
            encoded observation after taking action
        reward: float
            reward returned by environment
        done: bool
            whether game has finished
        """
        obs, reward, done, _ = self.gym_env.step(action)
        obs = np.array([obs[2:]])
        return self.model.predict(obs)[0], reward, done


class DynamicEnvironment():
    """
    Custom Wrapper around OpenAi Gym Environment.

    It runs observations through an encoder model first.
    Each output is based on 4 observations.
    Other than that identical to StaticEnvironment
    And other useful functions.
    """

    def __init__(self):
        """Create Environment and load encoder."""
        self.gym_env = gym.make("BreakoutNoFrameskip-v4")
        self.gym_env.seed(459)
        self.previousObservations = deque(maxlen=4)

        # Get Autoencoder Model:
        try:
            autoencoder = load_model(sys.argv[1], compile=False)
        except IndexError:
            print("ERROR:\
                 Please specify Path to encoder model as first parameter")
            quit()
        # Extract Encoder part from autoencoder:
        layer_name = 'encoding'
        self.model = Model(
            inputs=autoencoder.layers[0].input,
            outputs=autoencoder.get_layer(layer_name).output)

    def getOutputShape(self):
        """Return number of actions."""
        return self.model.output_shape[1:]

    def getActionSpace(self):
        """Return number of actions."""
        return self.gym_env.action_space.n

    def getSample(self):
        """Return a randomly sampled action."""
        return self.gym_env.action_space.sample()

    def reset(self):
        """Reset environment and return (encoded) observation.

        Because the encoder expects 4 observations as input
        I fill the observation deque with the first observation.
        """
        obs = self.gym_env.reset()
        obs = np.array([obs[2:]])
        for _ in range(4):
            self.previousObservations.append(obs)
        stacked = np.stack(self.previousObservations,
                           axis=-1).reshape(1, 208, 160, 12)
        return self.model.predict(stacked)[0]

    def step(self, action):
        """Take a step in envirnoment given an action.

        The environements observation is added to the deque,
        which holds the last four observations. This is fed to the encoder.

        Arguments:
        ----------
        action: int
            action to take in the environment

        Returns:
        --------
        obs: np.array
            encoded observation after taking action
        reward: float
            reward returned by environment
        done: bool
            whether game has finished
        """
        obs, reward, done, _ = self.gym_env.step(action)
        obs = np.array([obs[2:]])
        self.previousObservations.append(obs)
        stacked = np.stack(self.previousObservations,
                           axis=-1).reshape(1, 208, 160, 12)
        return self.model.predict(stacked)[0], reward, done


class Agent(Thread):
    """
    Agent sampling actions and filling the memory.

    Is an implementation of a Thread.
    It uses a copy of the model that is only updated once in a while

    Attributes:
    ----------
    env: StaticEnvironment/DynamicEnvironment
    model: TF/Keras model
    memory: deque
    predictedActions: int
        Counts the number of actions that were
        not random but from the max Q-Value.
        only for debugging actually.
    epsilon: float
        decaying epsilon from epsilon greedy strategy
    """

    def __init__(self):
        """Initialize all fields.

        Create environment depending on whether
        '--dynamic' is passed as argument to application.
        """
        Thread.__init__(self)
        if len(sys.argv) > 2 and sys.argv[2] == "--dynamic":
            print("Use Dynamic Environment")
            self.env = DynamicEnvironment()
        else:
            print("Use Static Environment")
            self.env = StaticEnvironment()
        self.model, _ = net.getModel(self.env.getOutputShape(), [
                                     self.env.getActionSpace(), learningRate])
        self.memory = deque(maxlen=maxQueueSize)
        self.predictedActions = 0
        self.epsilon = max_epsilon

    def getAction(self, state):
        """Return an action according ti epsilon greedy policy.

        Draw a random number and if it's smaller than the current epsilon
        return a random action rom the environment and else
        return the action with the highest Q-Value on the 'state'

        Arguments:
        ---------
        state: np.array()
            last observation

        Returns:
        --------
        action: int
        """
        exploit = random.uniform(0, 1)
        if exploit > self.epsilon:
            prediction = self.model.predict(np.array([state]))[0]
            self.predictedActions += 1
            return prediction.argmax()
        else:
            return self.env.getSample()

    def getMemorySample(self):
        """Return (batchSize) random samples from memory."""
        return np.array(random.sample(self.memory, batchSize))

    def runEpisode(self):
        """Run an episode in the environment.

        Take actions according to policy and fill the memory with tuples of
        (State, Action, Reward, New State, Done)

        Returns:
        score: int
            Total reward over whole episode
        steps: int
            Total steps took
        """
        score = 0
        steps = 0
        state = self.env.reset()
        for _ in range(maxSteps):  # Limit length of episode to maxSteps
            action = self.getAction(state)
            newState, reward, done = self.env.step(action)

            score += reward

            self.memory.append((state, action, reward, newState, done))
            if done:
                break
            state = newState
            steps += 1
        return score, steps

    def run(self):
        """Call 'runEpisode' in a loop and record scores and metrics.

        run runEpisode' totalEpisodes times and each time
        print to console and tensorboard:
        - score of last run
        - highscore until now
        - steps of last run
        - current epsilon
        - running mean of score
        - number of actions sampled from q-value predictions
        """
        global shouldRun
        print("Starting Agent")
        highScore = 0
        scoreQue = deque(maxlen=100)
        totalSteps = 0
        for episode in range(totalEpisodes):
            if not shouldRun:
                return
            self.predictedActions = 0
            score, steps = self.runEpisode()
            totalSteps += steps
            if score > highScore:
                highScore = score
            self.epsilon = min_epsilon + \
                (max_epsilon - min_epsilon)*np.exp(-decay*episode)
            scoreQue.append(score)
            print("Episode: ", episode, "| Score: ", score,
                  "| HighScore: ", highScore,
                  "| Epsilon: ", self.epsilon,
                  "Running Mean: ", np.mean(scoreQue))
            with file_writer.as_default():
                tf.summary.scalar("Score", score, step=episode)
                tf.summary.scalar("Predicted Actions",
                                  self.predictedActions, step=episode)
                tf.summary.scalar("Episode Length", steps, step=episode)
                tf.summary.scalar("Steps", totalSteps, step=episode)
                tf.summary.scalar("Highscore", highScore, step=episode)
                tf.summary.scalar("Epsilon", self.epsilon, step=episode)
                tf.summary.scalar(
                    "Running Mean", np.mean(scoreQue), step=episode)
        shouldRun = False

    def copyModel(self, model):
        """Copy weights of Trainers model to this models weights."""
        self.model.set_weights(model.get_weights())


class Trainer(Thread):
    """
    Samples from Agents memory and trains network.

    Is an implementation of a Thread.
    Attributes:
    model: TF/Keras model
    agent: reference to Agent
    """

    def __init__(self, agent):
        """Load model and copy current weights to agents model."""
        Thread.__init__(self)
        self.model, self.epoch = net.getModel(agent.env.getOutputShape(),
                                              [agent.env.getActionSpace(),
                                              learningRate])
        self.agent = agent
        self.agent.copyModel(self.model)

    def run(self):
        """In a loop: Sample from memory and train model.

        Draw a minibatch from memory.
        Predict Q-Values on states and newStates.
        Calculate target Q-values from discounted next Q-Values.
        Train model on target Q-values.
        Save loss to tensorboard.
        """
        print("Starting Trainer")
        while shouldRun:
            minibatch = self.agent.getMemorySample()
            w, h, c = agent.env.getOutputShape()
            states = np.concatenate(
                minibatch[:, 0]).reshape(batchSize, w, h, c)
            newStates = np.concatenate(
                minibatch[:, 3]).reshape(batchSize, w, h, c)

            targetQValues = self.model.predict(states, batch_size=batchSize)
            nextQValues = self.model.predict(newStates, batch_size=batchSize)
            maxNextQ = np.amax(nextQValues, axis=1)
            for i, (_, action, reward, _, done) in enumerate(minibatch):
                targetQValues[i][action] = reward if done else reward + \
                    gamma * maxNextQ[i]

            loss = self.model.train_on_batch(np.array(states), targetQValues)
            with file_writer.as_default():
                tf.summary.scalar("Loss", loss, step=self.epoch)
            self.epoch += 1
            if self.epoch % targetUpdateRate == 0:
                print("Copy Model")
                agent.copyModel(self.model)


# Get Tensorboard filewriter
today = datetime.datetime.today()
folderName = "%s%04d-%02d-%02d-%02d-%02d-%02d/" % (
    "savedata/agent/", today.year, today.month, today.day, today.hour,
    today.minute, today.second)
os.makedirs("%s" % (folderName))
file_writer = tf.summary.create_file_writer(folderName)


net = QNetwork.NeuralNetwork()
agent = Agent()
trainer = Trainer(agent)

with open('%s/architecture.txt' % (folderName), 'w') as fh:
    trainer.model.summary(print_fn=lambda x: fh.write(x + '\n'))

# Start Threads
shouldRun = True
agent.start()
while len(agent.memory) < batchSize:
    time.sleep(1)
trainer.start()

# Wait until they're finished or set them to stop on Ctrl-C
try:
    agent.join()
    trainer.join()
except KeyboardInterrupt:
    shouldRun = False
