"""
display policy for Soft-Actor-Critic, in a deterministic gridworld, with discrete actions.
policy found using dynamic programming.
(well, this is not actually SAC, because SAC has continuous actions, and is not optimizing using only dynamic
programming... however, the objective function and the state-action and policy updating are all taken from the SAC
paper)
"""
from itertools import count, product

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d

# set parameters here
width = 21  # width of gridworld, number of squares
height = 30  # height of gridworld, number of squares
reward_limits = (-1.0, -1.0)  # range of rewards.
discount = 0.99  # discount factor
alpha = 1e-3  # entropy temperature coefficient in SAC objective
convergence_th = 1e-5  # stop optimizing policy when the maximal difference in Q values is below this threshold
display = False  # should display policy after every dynamic programming iteration?
display_only_final = True  # should display policy after convergence?
display_trajectory = True  # should display trajectory?
display_policy = True # should display policy
argmax_trajectory = True  # when displaying trajectory, should the agent sample from the policy or argmax it?
save_image = False  # should save final image
save_image_sequence = False  # should save sequence of image

# the gridworld will have a "block" in its middle, which cannot be passed through by the agent
block_start_y = height // 3  # the lower y corrdintate of block
block_height = height // 3  # height of block
block_width = width // 3  # width of block

# useful enumerations
NUM_ACTIONS = 4  # the actions are: up, down, left, right
DOWN = 0
UP = 1
LEFT = 2
RIGHT = 3
X = 0
Y = 1

# calculate important locations
initial_x = width // 2  # x-coordinate of initial agent position (middle of x-axis)
initial_y = 0  # y-coordinate of initial agent position (bottom of gridworld)
target_x = width // 2  # x-coordinate of target position (middle of x-axis)
target_y = height - 1  # y-coordinate of target position (top of gridworld)
block_left = width // 2 - block_width // 2  # x-coordinate of left side of block
block_right = width // 2 + block_width // 2  # x-coordinate of right side of block
block_bottom = block_start_y  # y-coordinate of bottom side of block
block_top = block_start_y + block_height - 1  # y-coordinate of top side of block

# useful index arrays
x = np.arange(width)
y = np.arange(height)
actions = np.arange(NUM_ACTIONS)

# we now define the deterministic state transition function.
# transitions[i,j,k,:] is the new location, after applying action k at location i,j.
transitions = np.empty((width, height, NUM_ACTIONS, 2), np.int64)  # locx, locy, action, new_locx+nex_locy
# define the trivial transition function
transitions[x, :, UP, X] = x[:, None]
transitions[:, y, UP, Y] = y[None, :] + 1
transitions[x, :, DOWN, X] = x[:, None]
transitions[:, y, DOWN, Y] = y[None, :] - 1
transitions[:, y, LEFT, Y] = y[None, :]
transitions[x, :, LEFT, X] = x[:, None] - 1
transitions[:, y, RIGHT, Y] = y[None, :]
transitions[x, :, RIGHT, X] = x[:, None] + 1
# walls
transitions[x == x[0], :, LEFT, X] = x[0]
transitions[x == x[-1], :, RIGHT, X] = x[-1]
transitions[:, y == y[0], DOWN, Y] = y[0]
transitions[:, y == y[-1], UP, Y] = y[-1]
# block
inds_x = x == block_left - 1
transitions[inds_x, block_bottom:(block_top + 1), RIGHT, X] = x[inds_x, None]
inds_x = x == block_right + 1
transitions[inds_x, block_bottom:(block_top + 1), LEFT, X] = x[inds_x, None]
inds_y = y == block_bottom - 1
transitions[block_left:(block_right + 1), inds_y, UP, Y] = y[None, inds_y]
inds_y = y == block_top + 1
transitions[block_left:(block_right + 1), inds_y, DOWN, Y] = y[None, inds_y]

# the reward for a transition (s,a,s') will depend only on position s, here we create the (deterministic) reward
# for position s.
if reward_limits[0] == reward_limits[1]:  # this is the case of constant reward
    rewards = np.full((width, height), reward_limits[0])
else:
    rewards = np.random.RandomState(1111111).normal(size=(width, height))  # random rewards
    rewards = convolve2d(rewards, np.ones((2,2)), 'same', boundary='symm')  # make rewards smoother
    rewards = (rewards - rewards.max()) / (rewards.max() - rewards.min()) * (reward_limits[1] - reward_limits[0]) + \
              reward_limits[1]  # normalize to required range
rewards[target_x, target_y] = 0.0  # reward of target location is 0
rewards[block_left:(block_right + 1), block_bottom:(block_top + 1)] = np.nan  # block

# state-value function, and (stochastic) policy, initialized randomly
q = np.random.normal(size=(width, height, NUM_ACTIONS))
policy = np.empty((width, height, NUM_ACTIONS))  # policy[i,j,k] is the probability for action k in position i,j

# the value of the final position is always 0.0 (this is what seeds the dynamic programming to happen)
q[target_x, target_y] = 0.0


# softmax. we could use scipy.special.softmax. however, it has multiple array allocations which could be inefficient.
# therefore, we implement our own variant, which basically copies the code from scipy but without memory allocations.
# this is a class, because we want to define some persistent auxiliary arrays.
class Softmax:
    def __init__(self):
        # auxiliary arrays to hold intermediate calculations
        self.amax_aux = np.empty((width, height, 1))
        self.aux2 = np.empty((width, height, NUM_ACTIONS))
        self.aux3 = np.empty((width, height, 1))

    def __call__(self, x, axis, out):
        # calculations copied from scipy's softmax
        np.amax(x, axis=axis, keepdims=True, out=self.amax_aux)
        np.subtract(x, self.amax_aux, out=self.aux2)
        np.exp(self.aux2, out=self.aux2)
        with np.errstate(divide='ignore'):
            np.sum(self.aux2, axis=axis, keepdims=True, out=self.aux3)
            np.log(self.aux3, out=self.aux3)
        self.aux3 += self.amax_aux
        np.subtract(x, self.aux3, out=out)
        np.exp(out, out=out)
        return out


softmax = Softmax()

# for efficiency, we prefer to do as little memory allocations as possible. therefore, we define these auxiliary arrays
# to hold intermediate results of calculations
aux1 = np.empty_like(q)
entropy = np.empty((width, height))
value = np.empty((width, height))

# if required, prepare display
if display or display_only_final:
    plt.ion()  # probably not needed
    fig, ax = plt.subplots(1, 1, tight_layout=True)

    # draw grid lines
    for xx in range(width + 1):
        lw = 0.5 if (0 < xx < width) else 1.1
        ax.plot((xx,) * 2, (0, height), linewidth=lw, color='black', zorder=5)
    for yy in range(height + 1):
        lw = 0.5 if (0 < yy < height) else 1.1
        ax.plot((0, width), (yy,) * 2, linewidth=lw, color='black', zorder=5)
    ax.plot((initial_x, initial_x + 1), (0,) * 2, linewidth=3, color='deeppink', zorder=6, )
    ax.plot((target_x, target_x + 1), (height,) * 2, linewidth=3, color='blue', zorder=6)

    im = rewards.copy()
    if reward_limits[0] == reward_limits[1]:
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('rewards', [(0.0, 'white'), (1.0, 'white')])
    else:
        cmap = plt.get_cmap('afmhot')
        cmap = matplotlib.colors.ListedColormap(cmap(np.concatenate((np.linspace(0.3, 1.0, 30) ** 0.8, np.ones(10)))))
    # im[target_x, target_y] = 100.0
    cmap.set_bad('black')
    cmap.set_over('white')
    cmap.set_under('green')
    im_handle = ax.imshow(np.flipud(im.T), interpolation='none', cmap=cmap, extent=(0, width, 0, height), zorder=0)
    ax.set_xlim(-0.5, width + 0.5)
    ax.set_ylim(-0.5, height + 0.5)
    ax.set_title(r'$\alpha = {:.1f}\times 10^{{{:.0f}}}$'.format(*map(float,f'{alpha:.1E}'.split('E'))))
    ax.axis('off')

    if display_policy:
        # we now create arrows to represent the policy. each square will have 4 arrows, representing the 4 actions.
        # the color intensity of the arrows represents the probability of the action.
        xx, yy = np.meshgrid(x, y)
        zz = xx * 0.0  # this is simply a zeros array that will be useful for us for drawing and updating the arrows
        zz[(xx >= block_left) & (xx <= block_right) & (yy >= block_bottom) & (
                yy <= block_top)] = np.nan  # now arrow in block
        zz[(xx == target_x) & (yy == target_y)] = np.nan  # now arrow in target locations
        eps = 1e-3
        kwargs = dict(units='xy', angles='xy', scale_units='xy', scale=1.0, width=eps, headwidth=3 / eps,
                      headlength=4.0 / eps, headaxislength=4.0 / eps, minshaft=1.0, minlength=0.0,
                      zorder=6,
                      cmap=matplotlib.colors.LinearSegmentedColormap.from_list('quiver', [(0.0, (0.569, 0.11 , 0.616, 0.0)), (1.0,(0.569, 0.11 , 0.616, 1.0))]))
        quiver_up = ax.quiver(xx + 0.5, yy + 0.5, zz, zz + 0.5, zz, **kwargs)  # up action arrows
        quiver_down = ax.quiver(xx + 0.5, yy + 0.5, zz, zz - 0.5, zz, **kwargs)  # down action arrows
        quiver_left = ax.quiver(xx + 0.5, yy + 0.5, zz - 0.5, zz, zz, **kwargs)  # left action arrows
        quiver_right = ax.quiver(xx + 0.5, yy + 0.5, zz + 0.5, zz, zz, **kwargs)  # right action arrows

# find optimal policy by repeatedly applying Bellman backup on all states  .
# notice that we try to do all calculations without array memory allocations
for i in count():

    # this flag will become `True` once the Q values have converged
    converged = False

    if (
            i > 0):  # this condition is here because on the first iteration, we only want to calculate the policy,
        # without a Bellman backup
        with np.errstate(divide='ignore', invalid='ignore'):
            # calculate (negative) entropy of policy for all positions
            np.nansum(np.multiply(np.log(policy, out=aux1), policy, out=aux1), axis=2,
                      out=entropy)
        entropy *= (-1 * alpha)  # entropy times `alpha`
        np.sum(np.multiply(q, policy, out=aux1), axis=2, out=value)  # mean Q value for each position
        value += entropy  # value for Bellman update

        # update state-action array Q. we loop over all positions because I did not find a way to do this as a
        # vectorized operation, without needing to allocate an intermediate array
        max_diff = 0.0  # this will contain the maximal difference between new and current Q values. used for
        # convergence  test.pos_x
        for pos_x, pos_y, action in product(x, y, actions):

            # don't update target position, because its value is already set.
            if (pos_x == target_x) and (pos_y == target_y):
                continue
            # new Q value
            new_pos_x, new_pos_y = transitions[pos_x, pos_y, action]  # next location
            new_val = rewards[pos_x, pos_y] + discount * value[new_pos_x, new_pos_y]
            # update maximal difference
            max_diff = max(max_diff, abs(new_val - q[pos_x, pos_y, action]))
            # assign new Q value
            q[pos_x, pos_y, action] = new_val
        print(f'iteration {i}, max_diff={max_diff:.3E}')

        if max_diff < convergence_th:  # stop if converged
            converged = True

    # set the policy to be the softmax of the Q values
    softmax(np.divide(q, alpha, out=aux1), axis=2, out=policy)

    # display policy if required
    if display or (display_only_final and converged):

        # whether to sample a trajectory
        if display_trajectory:

            im[initial_x, initial_y] = -100.0

            # set current agent's position to initial position
            pos_x = initial_x
            pos_y = initial_y

            # step counter (we will terminate the episode if the agent gets stuck for too ling in the episode)
            cnt = 0

            # these will have the total discounted reward of the episode, with and without the entropy
            tot_reward = 0.0
            tot_reward_with_entropy = 0.0

            # do episode until agent gets to target
            while (pos_x != target_x) or (pos_y != target_y):

                # get distribution of actions from policy
                action_distribution = policy[pos_x, pos_y]

                # get action from distribution
                if argmax_trajectory:
                    # most probably action
                    action = np.argmax(action_distribution)
                else:
                    # sample action
                    action = np.random.choice(actions, p=action_distribution)

                # update total rewards
                with np.errstate(divide='ignore', invalid='ignore'):
                    tot_reward += discount * rewards[pos_x, pos_y]
                    tot_reward_with_entropy += discount * (
                            rewards[pos_x, pos_y] - alpha * np.nansum(
                        np.log(action_distribution) * action_distribution))

                # move to next position
                pos_x, pos_y = transitions[pos_x, pos_y, action]

                # update image to show current position of trajectory
                im[pos_x, pos_y] = -100.0

                # quit episode if taking too long
                cnt += 1
                if ((not converged) and (cnt >= 500)) or (converged and (cnt >= 3000)):
                    print('reached max episode length.')
                    break

            im_handle.set_data(np.flipud(im.T))
            print(
                f'iteration {i}.  total discounted reward: {tot_reward:.5f}, total discounted reward with entropy: '
                f'{tot_reward_with_entropy:.5f}')

        # update policy arrow colors. the colors represent the probabilities for each action.
        if display_policy:
            quiver_up.set_UVC(zz, zz + 0.5, policy[:, :, UP].T)
            quiver_down.set_UVC(zz, zz - 0.5, policy[:, :, DOWN].T)
            quiver_left.set_UVC(zz - 0.5, zz, policy[:, :, LEFT].T)
            quiver_right.set_UVC(zz + 0.5, zz, policy[:, :, RIGHT].T)

        # refresh figure
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(1e-5)
        if save_image_sequence:
            plt.savefig(f'images/alpha{alpha:013.10f}_step{i:05d}.png', dpi=100)

        # erase trajectory from image (for next time)
        im[:] = rewards[:]
        # im[target_x, target_y] = 100.0

    if converged:
        break
plt.ioff()
if save_image:
    plt.savefig(f'images/zoom_alpha_{alpha:013.10f}.png', dpi=400)
plt.show()
