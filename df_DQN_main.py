import numpy as np
import torch
from df_DQN_agent.env import Env
from df_DQN_agent.agent import Agent
from pandas import DataFrame, read_csv
from tqdm import tqdm
import matplotlib.pyplot as plt

EPOSIDE = 10
MEMORY_CAPACITY = 1000

base_p = 35
delta_p = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
state_list = np.load('./model_data/state_array.npy')
env = Env(read_csv('./model_data/df_label_env_df30min.csv'))
model_pump = torch.load('./DF_model_pump.pkl')
model_tower = torch.load('./DF_model_tower.pkl')
action_list = []
for i in range(len(delta_p)**2):
    action_list.append(i)
agent = Agent(state_list, action_list)


def train():
    reward_list = []
    for episode in tqdm(range(EPOSIDE)):
        state = env.reset()
        state = np.array([state])
        flag_p = model_pump.predict(state)
        flag_t = model_tower.predict(state)
        reward_sum = 0
        while True:
            action_pt = agent.choose_action(state)
            action = [int(action_pt) // 16, int(action_pt) % 16]
            if flag_p == 0:
                action[0] = base_p - action[0]
            else:
                action[0] = base_p + action[0]
            if flag_t == 0:
                action[1] = base_p - action[1]
            else:
                action[1] = base_p + action[1]

            state_, flag_p_, flag_t_, r, done = env.step(action)

            agent.store_transition(state, action_pt, r, state_)
            reward_sum += r
            if agent.memory_counter > MEMORY_CAPACITY:
                agent.learn()
            state = state_
            state = np.array([state])
            flag_p = model_pump.predict(state)
            flag_t = model_tower.predict(state)
            if done:
                print("Epi:", episode, "Reward:", reward_sum)
                break
        reward_list.append(reward_sum)
    plt.plot(reward_list)
    plt.savefig("./agent_convergence.jpg")
    plt.show()

    result_df = DataFrame(columns=['COP'])

    for j in range(len(read_csv('model_data/df_label_env_df30min.csv'))):
        Twb, CL= read_csv('model_data/df_label_env_df30min.csv').loc[j, ['Twb', 'CL']]
        state = [Twb, CL]
        action_pt = agent.choose_action(state)
        state_temp = np.array([state])
        flag_p = model_pump.predict(state_temp)
        flag_t = model_tower.predict(state_temp)
        action = [int(action_pt) // 16, int(action_pt) % 16]
        if flag_p == 0:
            action[0] = base_p - action[0]
        else:
            action[0] = base_p + action[0]
        if flag_t == 0:
            action[1] = base_p - action[1]
        else:
            action[1] = base_p + action[1]

        COP = env.check_cop(state, action)
        P = CL /COP
        length = len(result_df)
        result_df.loc[length, ['COP', 'pump', 'tower', 'power']] = COP, action[0], action[1], P
    result_df.to_csv('./COP_DQN_result/df_COP_agent_result_30min-'+str(EPOSIDE)+'year.csv', index=False)


if __name__ == '__main__':
    train()