# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import itertools
import matplotlib.pylab as plt
import seaborn as sns

n_h = 16  # hidden states
n_o = n_h  # observations
T = 7  # timesteps
n_a = 5  # actions
n_pi = (n_a - 1) ** (T - 1)  # strategies

# start state p(h1)
p_h0 = np.zeros(n_h)
p_h0[0] = 1

# prior belief p(o_t) Wo will ich hin?
prior = np.zeros(n_h)
prior[15] = p = 0.9
for i in range(n_h - 1):
    prior[i] = (1 - p) / (n_h - 1)

# probability observation p(o|h)
obs = np.zeros((n_o, n_h))
for i in range(n_h):
    obs[i, i] = b = 1
    obs[i - 1, i] = (1 - b) / 4
    obs[i, i - 1] = (1 - b) / 4
    obs[i - 4, i] = (1 - b) / 4
    obs[i, i - 4] = (1 - b) / 4

# p(h_t+1|h_t, r) step right
p_r = np.zeros((n_h, n_h))
for i in range(n_h):
    if i in [0, 4, 8, 12]:
        p_r[i, i - 1] = 0
        p_r[i - 1, i - 1] = 1
    else:
        p_r[i, i - 1] = c = 0.9
        p_r[i - 1, i - 1] = 1 - c

# p(h_t+1|h_t, l) step left
p_l = np.zeros((n_h, n_h))
for i in range(n_h):
    if i in [0, 4, 8, 12]:
        p_l[i - 1, i] = 0
        p_l[i, i] = 1
    else:
        p_l[i - 1, i] = d = 0.9
        p_l[i, i] = 1 - d

# p(h_t+1|h_t, u) step up
p_u = np.zeros((n_h, n_h))
for i in range(n_h):
    if i in [0, 1, 2, 3]:
        p_u[i, i - 4] = 0
        p_u[i - 4, i - 4] = 1
    else:
        p_u[i, i - 4] = e = 0.9
        p_u[i - 4, i - 4] = 1 - e

# p(h_t+1|h_t, d) step down
p_d = np.zeros((n_h, n_h))
for i in range(n_h):
    if i in [0, 1, 2, 3]:
        p_d[i - 4, i] = 0
        p_d[i, i] = 1
    else:
        p_d[i - 4, i] = f = 0.9
        p_d[i, i] = 1 - f

# p(h_t+1|h_t, s) stay
p_s = np.zeros((n_h, n_h))
for i in range(n_h):
    p_s[i, i] = 1

# 16x16x4 Matrix mit allen Aktionen
B = np.zeros((n_h, n_h, n_a))
B[:, :, 0] = p_r
B[:, :, 1] = p_l
B[:, :, 2] = p_u
B[:, :, 3] = p_d
B[:, :, 4] = p_s

# policies
steps = np.array(list(itertools.product(list(range(n_a - 1)), repeat=T - 1)))
last_step = np.zeros((n_pi, 1))
for i in range(n_pi):
    last_step[i] = 4
pol = np.append(steps, last_step, 1).astype(int)

# berechne forward messages
m_fwd = np.zeros((T, n_h, T))

print(np.dot(obs, prior) == prior)
# print(prior.shape)

def fwd_messages(i):
    m_fwd[0, :, i] = 1. / n_h  # obs[o[0]]
    for k in range(1, T):
        # starting at timestep 1 to make sure in first step this works
        action = pol[a, k - 1]
        if 0 < k <= i + 1:
            # for current and past states use observations
            x = m_fwd[k - 1, :, i] * obs[o[k - 1]]
            m_fwd[k, :, i] = np.dot(B[:, :, action], x)
            m_fwd_norm = m_fwd[k, :, i].sum()
            m_fwd[k, :, i] /= m_fwd_norm
        elif k > i + 1:
            # for not yet seen states use prior
            x = m_fwd[k - 1, :, i] * np.dot(obs.T, prior)
            m_fwd[k, :, i] = np.dot(B[:, :, action], x)
            m_fwd_norm = m_fwd[k, :, i].sum()
            m_fwd[k, :, i] /= m_fwd_norm


# berechne backward messages
m_bwd = np.zeros((T, n_h, T))


def bwd_messages(i):
    m_bwd[6, :, i] = np.dot(obs.T, prior)
    for k in reversed(range(0, T - 1)):
        action = pol[a, k]
        if k > i:
            m_bwd[k, :, i] = np.dot(B[:, :, action].T, m_bwd[k + 1, :, i]) * np.dot(obs.T, prior)
            m_bwd_norm = m_bwd[k, :, i].sum()
            m_bwd[k, :, i] /= m_bwd_norm
        elif 0 <= k <= i:
            m_bwd[k, :, i] = np.dot(B[:, :, action].T, m_bwd[k + 1, :, i]) * obs.T[:,o[k]]
            m_bwd_norm = m_bwd[k, :, i].sum()
            m_bwd[k, :, i] /= m_bwd_norm


# a: index policy, p_h: aktueller Zustand, h: Vektor mit Zustand pro Schritt, p_h1: next state
p_h = p_h0
a = 42
q_h = np.zeros((T, n_h, T))
h = np.zeros(T + 1)
o = np.zeros(T).astype(int)
for j in range(n_a):
    for i in range(T):
        if pol[a, i] == j:
            p_oi = np.dot(obs, p_h)
            o[i] = np.random.choice(n_h, p=p_oi)
            fwd_messages(i)
            bwd_messages(i)
            # q_h nicht mal m^k?
            q_h[:, :, i] = m_bwd[:, :, i] * m_fwd[:, :, i]
            for k in range(T):
                q_h[k, :, i] /= q_h[k, :, i].sum()
            p_h1 = np.dot(B[:, :, j], p_h)
            h[i + 1] = g = np.random.choice(n_h, p=p_h1)
            p_h = np.zeros(n_h)
            p_h[g] = 1

        # plot agent's way
x = np.zeros(T)
y = np.zeros(T)
grid = q_h[6, :, 6].reshape((4, 4))
fig = plt.figure(figsize=[12, 10])
ax = fig.gca()
sns.heatmap(grid, vmax=1, ax=ax, linewidths=2, xticklabels=False,
            yticklabels=False).invert_yaxis()

for i in range(T):
    x[i] = h[i] % 4 + 0.5
    y[i] = h[i] // 4 + 0.5
plt.plot(x, y, linewidth=3, alpha=0.7)
plt.show()
