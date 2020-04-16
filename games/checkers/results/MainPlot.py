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


# plot nets vs best network
net_vs_net_gen = np.load("net_vs_net_gen.npy")
net_vs_net_pred = np.load("net_vs_net_pred.npy")
# net_vs_net_mcts = np.load("net_vs_net_mcts.npy")

fig5 = plt.figure(4)
plt.plot(net_vs_net_gen, net_vs_net_pred, label="net only")
# plt.plot(net_vs_net_gen, net_vs_net_mcts, label="mcts")
axes = plt.gca()
axes.set_ylim([0, 0.55])
axes.grid(True, color=(0.9, 0.9, 0.9))
plt.legend()
plt.title("Score vs Best Network")
plt.xlabel("Generation")
plt.ylabel("Prediction Score")
fig5.show()


plt.show()
