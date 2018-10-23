import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# environment for Easy21
def step(state, action):

  BLACK = 1 # symbolic constant for card colour

  dealer_card, sum_player = state 
  sum_dealer = dealer_card

  if (action == 1): # if player chooses to hit
    card_num, card_col = draw()
    sum_player += card_num if card_col == BLACK else -card_num

  else: # if player sticks, dealer starts taking turns
    while (sum_dealer < 17 and sum_dealer >= 1):
      card_num, card_col = draw()
      sum_dealer += card_num if card_col == BLACK else -card_num
      

  if sum_player > 21 or sum_player < 1: # if player goes bust
    reward = -1
  elif sum_dealer > 21 or sum_dealer < 1: # if dealer goes bust
    reward = 1
  elif sum_player > sum_dealer and action == 0:
    reward = 1
  elif sum_player < sum_dealer and action == 0:
    reward = -1
  else:
    reward = 0

  if action == 1 and reward != -1:
    done = False
  else:
    done = True

  return dealer_card, sum_player, reward, done
    
# function for sampling a card from the (infinite) deck
def draw():

  RED = 0
  BLACK = 1

  cards = np.arange(1,11)
  colour_prob = [1./3, 2./3] # probabilities of drawing a red or black card

  card_col = np.random.choice([RED,BLACK], p=colour_prob)
  card_num = np.random.choice(cards)

  return card_num, card_col

# initialise a game of Easy21 by drawing a (black) card for the dealer and player
def init_game():

  dealer_card, _ = draw()
  player_card, _ = draw()

  return dealer_card, player_card


"""
Monte-Carlo Control

"""
print("Starting Q-learning...")
dealer_card, sum_player = init_game()

N_sa = np.zeros((10,21,2)) # counter for number of state and action visitations
Q_star_sa = np.zeros((10,21,2)) # action-value function
V_star_s = np.zeros((10,21)) # state-value function
N_0 = 1000

num_episodes = 0
player_states,actions,total_rewards = [],[],[]

while(num_episodes < 100000):

  # epsilon-greedy exploration
  epsilon = N_0/(N_0 + np.sum(N_sa[dealer_card-1, sum_player-1]))
  action = np.random.choice(2) if np.random.uniform() < epsilon else np.argmax(Q_star_sa[dealer_card-1, sum_player-1])

  # save states and actions for each episode
  player_states.append(sum_player)
  actions.append(action)
  N_sa[dealer_card-1, sum_player-1, action] += 1  # Every-visit MC

  # make a turn and record reward
  dealer_card, sum_player, reward, done = step((dealer_card, sum_player), action)
  total_rewards.append(reward)

  if done:
    ep_traj = np.vstack((player_states,actions)) # stack all state and actions in this episode

    for (x,y) in ep_traj.T: # Monte-Carlo update
      Q_star_sa[dealer_card-1, x-1, y] += (1./N_sa[dealer_card-1, x-1, y])*(np.sum(total_rewards) - Q_star_sa[dealer_card-1, x-1, y])

    # reset episode
    dealer_card, sum_player = init_game()
    player_states,actions,total_rewards = [],[],[]
    num_episodes += 1

print("Done")


"""
SARSA 

"""
print("Starting SARSA...")
MSE_Q = np.zeros(11)
lambda_ = np.zeros(11)

MSE_lambda_0 = np.zeros(10000)
MSE_lambda_1 = np.zeros(10000)

for i in range(11):

	num_episodes = 0
	lambda_[i] = 0.1*i
	gamma = 1

	N_sa = np.zeros((10,21,2)) # counter for number of state and action visitations
	Q_sa = np.zeros((10,21,2)) # action-value function
	V_s = np.zeros((10,21)) # state-value function
	E_sa = np.zeros((10,21,2)) # eligibility traces
	N_0 = 1000

	dealer_card, sum_player = init_game()
	action = np.random.choice(2)

	# make a turn and record reward
	dealer_card_next, sum_player_next, reward, done = step((dealer_card, sum_player), action)

	while(num_episodes < 10000):

		# epsilon-greedy exploration
		epsilon = N_0/(N_0 + np.sum(N_sa[dealer_card_next-1, sum_player_next-1]))
		action_next = np.random.choice(2) if np.random.uniform() < epsilon else np.argmax(Q_sa[dealer_card_next-1, sum_player_next-1])

		N_sa[dealer_card-1, sum_player-1, action] += 1 
		E_sa[dealer_card-1, sum_player-1, action] += 1 
		
		# TD error
  		delta = reward + gamma*Q_sa[dealer_card_next-1, sum_player_next-1, action_next] - Q_sa[dealer_card-1, sum_player-1, action]

  		for s1 in range(10):
  			for s2 in range(21):
  				for a in range(2):
  					if N_sa[s1,s2,a] > 0:
				  		Q_sa[s1,s2,a] += (1./N_sa[s1,s2,a])*delta*E_sa[s1,s2,a]
				  		E_sa[s1,s2,a] = gamma*lambda_[i]*E_sa[s1,s2,a]

		dealer_card = dealer_card_next
		sum_player = sum_player_next
		action = action_next

		# make a turn and record reward
		dealer_card_next, sum_player_next, reward, done = step((dealer_card, sum_player), action)

  		if done:
			E_sa = np.zeros((10,21,2)) # eligibility traces
			dealer_card, sum_player = init_game()
			action = np.random.choice(2)
			# make a turn and record reward
  			dealer_card_next, sum_player_next, reward, done = step((dealer_card, sum_player), action)
  			if lambda_[i] == 0:
				MSE_lambda_0[num_episodes] = np.sum((Q_sa - Q_star_sa) ** 2)
			elif lambda_[i] == 1:
				MSE_lambda_1[num_episodes] = np.sum((Q_sa - Q_star_sa) ** 2)
			num_episodes += 1


	MSE_Q[i] = np.sum((Q_sa - Q_star_sa) ** 2)

print("Done")

print("Plotting results...")
# V*(s) = max_a Q*(s,a)
V_star_s = np.max(Q_star_sa, axis=2)

# plot optimal value function
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')

X_axis = np.arange(1,11)
Y_axis = np.arange(1,22)

X, Y = np.meshgrid(X_axis, Y_axis, indexing='ij')

ax.plot_wireframe(X,Y,V_star_s)


fig2 = plt.figure(2)
plt.plot(lambda_, MSE_Q)

fig3 = plt.figure(3)
plt.plot(np.arange(1,10001), MSE_lambda_0)

fig4 = plt.figure(4)
plt.plot(np.arange(1,10001), MSE_lambda_1)

plt.show()