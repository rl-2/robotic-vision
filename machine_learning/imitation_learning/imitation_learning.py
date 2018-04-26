#==========================================================================  
# Imitation Learning Working Code 
# 
# The code was initially written for UCSB Deep Reinforcement Learning Seminar 2018
#
# Authors: Jieliang (Rodger) Luo, Sam Green
#
# April 9th, 2018
#==========================================================================

import gym
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import getch
import ipdb
import numpy as np
import argparse

# From the command line, run with
# ipython imitation_learning.py -- --mode learn
# or
# ipython imitation_learning.py -- --mode eval 


parser = argparse.ArgumentParser(description='Imitation learning for MountainCar-v0.')
parser.add_argument('--mode', type=str, required=True, choices=["learn", "eval"], default="learn")
parser.add_argument('--model', type=str, required=False, default="model_mt_car_imitation_steps_733.pkl")
args = parser.parse_args()

num_episodes = 100

# build a neural network
class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(2, 10)
		self.fc2 = nn.Linear(10, 3)

	def forward(self, x):
		x = F.relu(self.fc1(x)) # activation function 
		x = self.fc2(x)
		return F.softmax(x, dim=1)

# keyboard controls
# a -> left, s -> don't move, d -> right
def humanInput():

	char = getch.getch()

	# Convert from bytes to string
	# char = char.decode("utf-8")
	
	if char == 'a':
		a = 0
	elif char == 's':
		a = 1
	elif char == 'd':
		a = 2
	return a

def main():

	# create environment
	env = gym.make('MountainCar-v0')
	env = env.unwrapped # do we need this function? 

	print(env.action_space) # 3 actions: push left, no push, push right
	print(env.observation_space) # 2 observations: position, velocity 
	print(env.observation_space.high) # max position & velocity: 0.6, 0.07
	print(env.observation_space.low) # min position & velocity: -1.2, -0.07

	#initialize the network
	net = Net()

	# build the loss function
	criterion = nn.CrossEntropyLoss()#nn.L1Loss()
	optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9) # stochastic gradient descent

	if args.mode == "learn":
		print("Learning mode")
		# learning from expert (user inputs)
		for episode in range(num_episodes):
			
			observation = env.reset()
			done = False
			step = 0
			actions = []
			observations = [] 

			while not done:
				env.render()
				#print(observation)
				
				action = humanInput()
				observation, reward, done, info = env.step(action)
		
				# store all the observations and actions from one episode
				observations.append(observation)
				actions.append(action)
				
				step += 1
				
			print("Episode {} finished after {} steps".format(episode+1, step))

			# print(observations)
			# print(actions)

			# train the network

			# wrap observations and actions in Variables
			observations = torch.FloatTensor(observations)
			actions = torch.LongTensor(actions)

			inputs, labels = Variable(observations), Variable(actions)

			# zero the parameter gradients
			optimizer.zero_grad()
			#ipdb.set_trace()
			
			# forward + backward + optimize
			outputs = net(inputs)
			#print(outputs)
			loss = criterion(outputs, labels)
			print("Loss value: {}".format(loss))
			
			loss.backward()
			optimizer.step()

			print("Network updated!")

			# See how well the current policy performs
			observation = env.reset()
			done = False
			step_count = 0
			max_steps = 1500
			while not done and step_count < max_steps:
				step_count += 1
				env.render()
				#print(observation)
				action_probabilities = net(Variable(torch.FloatTensor([observation])))
				print(action_probabilities.data.numpy()[0])
				#ipdb.set_trace()
				action = np.random.choice([0,1,2], p=action_probabilities.data.numpy()[0])
				observation, reward, done, info = env.step(action)

			# We trained a model that solved the Mountain Car env before max_steps!
			if done and step_count < max_steps:
			#net.save_state_dict("model_mt_car_imitation_steps_{}.pt".format(step_count))
				torch.save(net, "model_mt_car_imitation_steps_{}.pkl".format(step_count))

	elif args.mode == "eval":
		print("Eval mode")
		net = torch.load(args.model)

		observation = env.reset()
		done = False
		step_count = 0
		max_steps = 2000
		while not done and step_count < max_steps:
			step_count += 1
			env.render()
			#print(observation)
			action_probabilities = net(Variable(torch.FloatTensor([observation])))
			print(action_probabilities.data.numpy()[0])
			#ipdb.set_trace()
			action = np.random.choice([0,1,2], p=action_probabilities.data.numpy()[0])
			observation, reward, done, info = env.step(action)

if __name__ == "__main__":
	main()