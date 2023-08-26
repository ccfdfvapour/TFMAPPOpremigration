from migrate_env import MigrateEnv
import numpy as np

env_args = {"scenario": "pre-migrate",
            "n_agent": 12,
            "reward": "scoring"}

env = MigrateEnv(env_args=env_args)
env.reset()
r_total = np.zeros((10,12))
for step in range(10):
    actions = np.random.randint(0, 2, 12)
    actions = np.ones(12)
    actions = np.zeros(12)
    print(actions)
    obs, state, rewards, dones, infos, ava = env.step(actions)
    print(infos)
    for i in range(12):
        r_total[step][i]=rewards[i]
    #print(obs, rewards, dones)
    #print(dones)
#average_step reward
#print(np.mean(r_total,axis=0).sum())
#average_spisode reward
print(np.mean(r_total,axis=1).sum())