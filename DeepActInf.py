import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

n_h = 16  # hidden states
n_o = n_h  # observations
# T = 7  # timesteps
n_a = 5  # actions

goal =15
reward = np.zeros(n_h)
reward[goal] = p = 0.9
for i in range(n_h - 1):
    reward[i] = (1 - p) / (n_h - 1)

# probability observation p(o|h)
obs = np.zeros((n_o, n_h))
for i in range(n_h):
    obs[i, i] = b = 1

# p(h_t+1|h_t, r) step right
p_r = np.zeros((n_h, n_h))
for i in range(n_h):
    if i in [0, 4, 8, 12]:
        p_r[i, i - 1] = 0
        p_r[i - 1, i - 1] = 1

    else:
        p_r[i, i - 1] = c = 1
        p_r[i - 1, i - 1] = 1 - c

# p(h_t+1|h_t, l) step left
p_l = np.zeros((n_h, n_h))
for i in range(n_h):
    if i in [0, 4, 8, 12]:
        p_l[i - 1, i] = 0
        p_l[i, i] = 1
    else:
        p_l[i - 1, i] = d = 1
        p_l[i, i] = 1 - d

# p(h_t+1|h_t, u) step up
p_u = np.zeros((n_h, n_h))
for i in range(n_h):
    if i in [0, 1, 2, 3]:
        p_u[i, i - 4] = 0
        p_u[i - 4, i - 4] = 1
    else:
        p_u[i, i - 4] = e = 1
        p_u[i - 4, i - 4] = 1 - e

# p(h_t+1|h_t, d) step down
p_d = np.zeros((n_h, n_h))
for i in range(n_h):
    if i in [0, 1, 2, 3]:
        p_d[i - 4, i] = 0
        p_d[i, i] = 1
    else:
        p_d[i - 4, i] = f = 1
        p_d[i, i] = 1 - f

# p(h_t+1|h_t, s) stay
p_s = np.zeros((n_h, n_h))
for i in range(n_h):
    p_s[i, i] = 1

# 16x16x4 Matrix mit allen Aktionen
trans = np.zeros((n_h, n_h, n_a))
trans[:, :, 0] = p_r
trans[:, :, 1] = p_l
trans[:, :, 2] = p_u
trans[:, :, 3] = p_d
trans[:, :, 4] = p_s

class Model(Layer):

    def __init__(self):
        super(Model, self).__init__()

        self.hidden_layer_1 = tf.keras.layers.Dense(units=100,
                                                    activation=tf.keras.activations.relu,
                                                    input_shape=(1, 16)
                                                    )
        self.hidden_layer_2 = tf.keras.layers.Dense(units=100,
                                                    activation=tf.keras.activations.relu
                                                    )
        self.output_layer = tf.keras.layers.Dense(units=n_a,
                                                  activation=tf.keras.activations.softmax
                                                  )

    def call(self, x):
        x = self.hidden_layer_1(x)
        x = self.hidden_layer_2(x)
        x = self.output_layer(x)
        return x

def EFE_loss(value_state, value_next_state, reward):
    return tf.keras.losses.MSE(value_state, value_next_state - reward)

def p_loss_func(policy_output, value_output):
   return tf.math.reduce_mean(policy_output * tf.math.log(value_output))


policy_net = Model()
value_net = Model()

opt_p = tf.optimizers.Adam(0.001)
opt_v = tf.optimizers.Adam(0.001)

state = np.zeros(n_h)
state[0] = 1
state = state.reshape(1,16)
actions = ['right', 'left', 'up', 'down', 'stay']
print('init state: ', np.argmax(state))

with tf.GradientTape(persistent=True) as tape:
    for i in range(1000):
        print('i', i)
        action = policy_net(state)
        action_idx = tf.keras.backend.get_value(tf.math.argmax(action, axis=1))[0]
        print('action: ', actions[action_idx])

        next_state = np.dot(trans[:,:,action_idx], state.reshape(16,1)).reshape(1,16)
        current_reward = reward[np.argmax(next_state)]

        value_state = value_net(state)
        print('val: ', value_state)
        value_next_state = value_net(next_state)

        v_loss = EFE_loss(value_state, value_next_state, current_reward)
        print('v_loss', v_loss)
        gradients_v = tape.gradient(v_loss, value_net.trainable_variables)
        opt_v.apply_gradients(zip(gradients_v, value_net.trainable_variables))

        p_loss = p_loss_func(action, value_state)
        print('p_loss', p_loss)
        gradients_p = tape.gradient(p_loss, value_net.trainable_variables)
        opt_p.apply_gradients(zip(gradients_p, policy_net.trainable_variables))

        state = next_state.reshape(1,16)

        print('next state: ', np.argmax(next_state))





