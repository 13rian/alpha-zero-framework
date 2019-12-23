import matplotlib.pyplot as plt
import numpy as np


# define the styles for the plot
plt.style.use('seaborn-dark-palette')


# plot the policy loss
policy_loss = np.load("policy_loss.npy")

spacing = 1
generation = np.arange(0, len(policy_loss), spacing)

fig1 = plt.figure(1)
plt.plot(generation, policy_loss[0::spacing])
axes = plt.gca()
axes.grid(True, color=(0.9, 0.9, 0.9))
plt.title("Policy Loss")
plt.xlabel("Generation")
plt.ylabel("Cross-Entropy Loss")
fig1.show()


# plot the value loss
value_loss = np.load("value_loss.npy")

fig2 = plt.figure(2)
plt.plot(generation, value_loss[0::spacing])
axes = plt.gca()
axes.grid(True, color=(0.9, 0.9, 0.9))
plt.title("Value Loss")
plt.xlabel("Generation")
plt.ylabel("MSE Loss")
fig2.show()


# plot the average moves played
avg_moves = np.load("avg_moves.npy")

spacing = 1
generation = np.arange(0, len(policy_loss), spacing)

fig3 = plt.figure(3)
plt.plot(generation, avg_moves[0::spacing])
axes = plt.gca()
axes.grid(True, color=(0.9, 0.9, 0.9))
plt.title("Average Moves Played")
plt.xlabel("Generation")
plt.ylabel("Move Count")
fig3.show()



# plot the score against the minimax player
net_generation = np.load("net_generation.npy")
white_scores = np.load("white_scores.npy")
black_scores = np.load("black_scores.npy")

fig4 = plt.figure(4)
endIdx = 40
plt.plot(net_generation[0:endIdx], white_scores[0:endIdx], label="white")
plt.plot(net_generation[0:endIdx], black_scores[0:endIdx], label="black")
axes = plt.gca()
# axes.set_ylim([0, 1.5])
axes.grid(True, color=(0.9, 0.9, 0.9))
plt.legend()
plt.title("Network Value Error")
plt.xlabel("Generation")
plt.ylabel("MSE Value")
fig4.show()


plt.show()
