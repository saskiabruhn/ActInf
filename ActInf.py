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

goal = 15
# prior belief p(o_t) Wo will ich hin?
prior = np.zeros(n_h)
prior[goal] = p = 0.9
for i in range(n_h - 1):
    prior[i] = (1 - p) / (n_h - 1)

# probability observation p(o|h)
obs = np.zeros((n_o, n_h))
for i in range(n_h):
    obs[i, i] = b = 1
    # obs[i - 1, i] = (1 - b) / 4
    # obs[i, i - 1] = (1 - b) / 4
    # obs[i - 4, i] = (1 - b) / 4
    # obs[i, i - 4] = (1 - b) / 4

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


def fwd_messages(timestep, policy):
    m_fwd = np.zeros((T, n_h))
    m_fwd_norm = np.zeros((T))
    # state 0 does not get a fwd message from before
    m_fwd[0, :] = 1. / n_h  # obs[o[0]]

    for k in range(1, T):
        # action from previous state to get to current state
        action = pol[policy, k - 1]
        # messages up to one step further than current state can be calculated with seen observations
        if 0 < k <= timestep + 1:
            # for current and past states use observations
            m_fwd[k, :] = B[:, :, action] @ (m_fwd[k - 1, :] * obs[o[k - 1]])
            m_fwd_norm[k] = m_fwd[k, :].sum()
            if m_fwd_norm[k] != 0:
                m_fwd[k, :] /= m_fwd_norm[k]
        elif k > timestep + 1:
            # for not yet seen states use prior
            m_fwd[k, :] = B[:, :, action] @ (m_fwd[k - 1, :] * (obs @ prior))
            m_fwd_norm[k] = m_fwd[k, :].sum()
            if m_fwd_norm[k] != 0:
                m_fwd[k, :] /= m_fwd_norm[k]

    return m_fwd, m_fwd_norm


# berechne backward messages
def bwd_messages(timestep, policy):
    m_bwd = np.zeros((T, n_h))
    m_bwd_norm = np.zeros((T))
    # last step does not get a bwd message from futre state
    m_bwd[6, :] = obs @ prior
    for k in reversed(range(0, T - 1)):
        action = pol[policy, k]

        # B is transposed here because we want to do somae inverse bayes to infer the probability of having been in some state given now we are in another state
        if k >= timestep:
            m_bwd[k, :] = B[:, :, action].T @ (m_bwd[k + 1, :] * (obs @ prior))
            m_bwd_norm[k] = m_bwd[k, :].sum()
            if m_bwd_norm[k] != 0:
                m_bwd[k, :] /= m_bwd_norm[k]
        elif 0 <= k < timestep:
            m_bwd[k, :] = B[:, :, action].T @ (m_bwd[k + 1, :] * obs[o[k+1]])
            m_bwd_norm[k] = m_bwd[k, :].sum()
            if m_bwd_norm[k] != 0:
                m_bwd[k, :] /= m_bwd_norm[k]
    return m_bwd, m_bwd_norm


# a: index policy, p_h: aktueller Zustand, h: Vektor mit Zustand pro Schritt, p_h1: next state
p_h = p_h0
q_h = np.zeros((T, n_h, T, n_pi))
q_h_norm = np.zeros((T, T, n_pi))
h = np.zeros(T + 1)
o = np.zeros(T).astype(int)
q_pi = np.zeros((n_pi, T))
obs_m = np.zeros((T, n_h))
obs_m[:, goal] = [1,1,1,1,1,1,1]
print(obs_m)
for i in range(T):
    p_oi = np.dot(obs, p_h)
    obs_m[i, :] = p_oi
    o[i] = np.random.choice(n_h, p=p_oi)
    for a in range(n_pi):
        m_fwd, m_fwd_norm = fwd_messages(timestep=i, policy=a)
        m_bwd, m_bwd_norm = bwd_messages(timestep=i, policy=a)

        # to avoid 0 for the log in q_pi
        for l in range(T):
            if m_fwd_norm[l] == 0:
                m_fwd_norm[l] = 0.0000001
        q_h[:, :, i, a] = (m_bwd * m_fwd) * obs_m
        for k in range(T):
            q_h_norm[k, i, a] = q_h[k, :, i, a].sum()
            if q_h_norm[k, i, a] != 0:
                q_h[k, :, i, a] /= q_h_norm[k, i, a]
        q_pi[a, i] = q_h_norm[T-1, i, a] * np.exp(np.log(m_fwd_norm).sum())

    q_pi[:, i] /= q_pi[:, i].sum()

    # action selection with maximum selection
    max_pol = np.argmax(q_pi[:,i])
    p_h1 = np.dot(B[:, :, pol[max_pol, i]], p_h)
    h[i + 1] = g = np.random.choice(n_h, p=p_h1)
    p_h = np.zeros(n_h)
    p_h[g] = 1

# plot agent's way
x = np.zeros(T)
y = np.zeros(T)
grid = q_h[6, :, 6, max_pol].reshape((4, 4))
fig = plt.figure(figsize=[12, 10])
ax = fig.gca()
sns.heatmap(grid, vmax=1, ax=ax, linewidths=2, xticklabels=False,
            yticklabels=False).invert_yaxis()

for i in range(T):
    x[i] = h[i] % 4 + 0.5
    y[i] = h[i] // 4 + 0.5
plt.plot(x, y, linewidth=3, alpha=0.7)
plt.show()