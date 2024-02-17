## Import packages

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from resilience import resilience
from model_env import init_params, call_ode
from RL_utils import list_flattener
from agent import DQNAgent
from random import uniform
from sklearn.preprocessing import MinMaxScaler


## Define Agent

n_episodes = 2000  # number of episodes
best_score = -np.inf
load_checkpoint = False
n_neur = 128  # number of neurons in hidden layer
ln = 1  # number of hidden layers

iteration = 15  # for saving out the agent

agent = DQNAgent(gamma=0.99, epsilon=1.0, lr=0.001, input_dims=2, n_actions=2, mem_size=1000, eps_min=0,
                 batch_size=32, ln=ln, n_neuron=n_neur, replace=100, eps_dec=1 / (n_episodes * 0.95), chkpt_dir='',
                 algo='DQNAgent_' + str(ln) + '_' + str(n_neur) + '_' + str(iteration), target_tau=1e-4)

scores = []
win_pct_list = []


## Define variables

T_m = []
dT1_m = []
M_m = []
T0 = 330
M0, I0, D0, V0, Tj0 = init_params()
x0 = [V0, M0, I0, D0, T0, Tj0]
t, V, M, I, D, T, Tj = call_ode((0, 60000), 0, x0)
dT1 = np.diff(T) / np.diff(t)
T_m.append(T)
dT1_m.append(dT1)

T0 = 324
M0, I0, D0, V0, Tj0 = init_params()
x0 = [V0, M0, I0, D0, T0, Tj0]
t, V, M, I, D, T, Tj = call_ode((0, 60000), 0, x0)
dT1 = np.diff(T) / np.diff(t)
T_m.append(T)
dT1_m.append(dT1)

T_m = list_flattener(T_m)
T_m = np.array(T_m)
dT1_m = list_flattener(dT1_m)
dT1_m = np.array(dT1_m)

T_sc = MinMaxScaler(feature_range=(0, 1))
dT1_sc = MinMaxScaler(feature_range=(-1, 1))

T_sc.fit(T_m.reshape(-1, 1))
dT1_sc.fit(dT1_m.reshape(-1, 1))


## DQN algorithm

for episode in range(0, n_episodes):

    if load_checkpoint:
        agent.load_models()
        agent.epsilon = 0

    T0 = uniform(324, 330)  # initial reactor temperature
    # T0 = 326.7, 327.7  # for plotting the decision boundary

    M0, I0, D0, V0, Tj0 = init_params()
    x0 = [V0, M0, I0, D0, T0, Tj0]
    t_l, V_l, M_l, I_l, D_l, T_l, Tj_l, Fh_l, tinj_l = [[0]], [[V0]], [[M0]], [[I0]], [[D0]], [[T0]], [[Tj0]], [], []

    t, V, M, I, D, T, Tj = call_ode((0, 1000 * 60), 0, x0)  # determination of Rb (baseline resilience)
    args = t, T, M, V, 0, 0, T0
    Q_res, Rb = resilience(args)

    Fh = 0  # l/s, diluent flowrate
    inj = 60  # sec, constant injection time
    ts, te = 0, 1

    a1 = 0  # 0 if there has not yet been an injection, auxiliary variable
    s_a_m = []  # states and actions memory

    while True:
        tspan = (ts * 60, (te * 60) + 1)
        t, V, M, I, D, T, Tj = call_ode(tspan, Fh, x0)

        t_l.append(t[1:])
        M_l.append(M[1:])
        T_l.append(T[1:])
        V_l.append(V[1:])

        x0 = [V[-1], M[-1], I[-1], D[-1], T[-1], Tj[-1]]  # initial values
        dT1 = (np.diff(T_l[-1]) / np.diff(t_l[-1]))  # first derivative of reactor temperature

        if a1 == 0:  # at every 1 minute

            T_act_sc = T_sc.transform(T_l[-1].reshape(-1, 1))[-1]  # reactor temperature
            dT1_act_sc = dT1_sc.transform(dT1.reshape(-1, 1))[-1]  # first derivative of reactor temperature
            state = (T_act_sc[0], dT1_act_sc[0])  # state

            action = agent.choose_action(state)  # agent decides

            if action == 0:  # if the agent does not intervene (action=0), Fh is not added
                Fh = 0
            else:  # if the agent intervenes, Fh is added, and increase the auxiliary variable to 1
                Fh = 7  # l/s
                a1 += 1

                tinj_l.append(te)  # for plotting
                Fh_l.append(Fh)

            ts += 1
            te += 1

            s_a_m.append([state, action])

        else:  # if the agent has already intervened
            Fh = 0
            ts += 1
            te = 1000

        if t_l[-1][-1] >= 999.9:
            break

    t_l = list_flattener(t_l)
    t_l = np.array(t_l)
    M_l = list_flattener(M_l)
    M_l = np.array(M_l)
    T_l = list_flattener(T_l)
    T_l = np.array(T_l)
    V_l = list_flattener(V_l)
    V_l = np.array(V_l)

    args = t_l, T_l, M_l, V_l, tinj_l, inj, T0  # determination of resilience
    Q_res, R = resilience(args)

    for i in range(0, len(s_a_m)):  # append state to s_a_m memory
        if i == len(s_a_m) - 1:
            if len(s_a_m) >= 1000:
                s_a_m[i].append((T_sc.transform(T_l.reshape(-1, 1))[-1][0],
                                 dT1_sc.transform(dT1.reshape(-1, 1))[-1][0]))
                s_a_m[i].append(R)
            else:
                s_a_m[i].append((T_sc.transform(T_l.reshape(-1, 1))[(len(s_a_m) + 1) * 60][0],
                                 dT1_sc.transform(dT1.reshape(-1, 1))[0][0]))
                s_a_m[i].append(R)
        else:
            s_a_m[i].append(s_a_m[i + 1][0])
            s_a_m[i].append(R)

    for i in range(0, len(s_a_m)):
        state = s_a_m[i][0]
        action = s_a_m[i][1]
        reward = s_a_m[i][3]
        if Rb > 0.99 and action == 1:  # punishment, if the resilience is good, but the agent still intervenes
            reward = -5 * (1 / tinj_l[0])
        if Rb > 0.99 and action == 0:
            reward = 1
        new_state = s_a_m[i][2]

    if not load_checkpoint:
        agent.store_transition(state, action, reward, new_state)
        agent.learn()

    agent.decrement_epsilon()
    scores.append(reward)

    mean_scores = np.mean(scores[-50:])
    win_pct_list.append(mean_scores)
    if episode % 50 == 0:
        if episode % 50 == 0:
            print('episode', episode, 'reward-mean %.4f' % mean_scores, 'epsilon %.2f' % agent.epsilon)

    if mean_scores > best_score:
        if not load_checkpoint:
            agent.save_models()
        best_score = mean_scores


## Plot average reward

scoresd = pd.Series(scores)
p = scoresd.rolling(50)
moving_averages = p.mean()
plt.figure(1)
plt.plot(moving_averages)
plt.xlabel('Episode')
plt.ylabel('Average reward')


## Plot decision boundary

# plt.figure(85)
# plt.plot(t_l,T_l-273.15, label='$T_{}$ = {:.1f} °C'.format('{0}', T0-273.15))
# plt.plot(t_l,np.ones(len(t_l))*54.1, 'g--', label='Decision boundary')
# plt.xlabel('Time (min)')
# plt.ylabel('Temperature (°C)')
# plt.legend()
# plt.show()
