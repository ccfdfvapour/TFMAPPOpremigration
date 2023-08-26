from functools import partial
import gym
from gym.spaces import Box
from gym.wrappers import TimeLimit
import numpy as np
from multiagentenv import MultiAgentEnv


class MigrateEnv(MultiAgentEnv):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 获得从.sh传入参数
        self.scenario = kwargs["env_args"]["scenario"]
        self.n_agents = kwargs["env_args"]["n_agent"]
        self.reward_type = kwargs["env_args"]["reward"]
        #self.n_agents = 12
        self.Tmax = 10
        self.deltaT = 10
        # 服务器位置e
        self.e_locx = np.linspace(600, 3000, 3)
        self.e_locy = np.linspace(50, 50, 3)
        # 用户轨迹Lu（每300m，即15s采样一次位置）
        self.u_locx = np.linspace(600, 2600, 11)
        self.u_locy = np.linspace(10, 10,11)
        # 用户与服务器距离,基站覆盖半径为600m
        ued = np.zeros(shape=(self.Tmax + 1, len(self.e_locx)))
        for t in range(self.Tmax + 1):
            x1 = self.u_locx[t]
            y1 = self.u_locy[t]
            for e in range(len(self.e_locx)):
                x0 = self.e_locx[e]
                y0 = self.e_locy[e]
                ued[t][e] = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)  # 每个服务器与这一刻用户1距离

        # dmin = ued - 300
        # self.e = np.argmin(dmin, axis=1)
        self.e = np.array([0 ,0, 0, 0, 0, 0,0,0, 0, 1 ,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        # 与卸载服务器的连接时间
        self.dt = np.array((50, 40,30,20, 10,0, 60,50, 40,30,20))
        # 服务器负载
        np.random.seed(5)
        # self.a = np.random.randint(0,5,5)
        self.load = np.array((0.0, 0.0, 0.0))

        # 计算参数
        self.C = np.array((64.0, 44.0, 54.0, 38.0, 60.0, 48.0)) *3*1.0  # 计算能力mb/s *3
        self.bw = 128.0  # 服务器间带宽
        # 任务大小
        self.Ptask = 300.0  # Mb #default=300
        self.Din = 16  # Mb
        self.Dmig = self.Ptask / 2
        self.Dinfo = 1  # MB

        # 传输速度
        self.P = 0.25  # w用户天线发射功率
        self.Bwc = 2 * 10 ** 6  # hz
        self.f = 10015 * 10 ** 6  # hz 载波频率
        self.d = ued  # 用户与服务器距离
        self.h = 4.11 * (3 * 10 ** 8 / (4 * np.pi * self.f * self.d)) ** 2  # 衰落信道
        zao = np.array(([4000], [3641], [3745], [3409], [4673], [3720], [4536], [4827], [3134], [3024], [3634]),
                       dtype=float)
        zao = np.random.randint(500, 1200, (11, 3))
        self.Trans_speed = 1 / (1024 ** 2) * self.Bwc * np.log2(
            1 + self.P * self.h / ((zao * 10 ** 6) * (10 ** (-20))))*1.2*1.0  # v/2
        # print(256/self.Trans_speed)
        # 时延
        self.Ttrans = self.Din / self.Trans_speed
        # print(self.Ttrans)
        # self.load_wait_list = np.zeros(self.n_pistons)

        """
            x,y:用户位置    T:平均时延  load:服务器负载
        """
        min_x = 600
        max_x = 2600
        min_y = 10
        max_y = 10
        min_T = 0
        max_T = 1000
        min_load = 0.00
        max_load = 100.00
        min_trans_speed = 0.00
        max_trans_speed = 50.0
        self.low = np.array([min_x, min_y, min_load, min_load, min_trans_speed, min_trans_speed, min_T])
        self.high = np.array([max_x, max_y, max_load, max_load, max_trans_speed, max_trans_speed, max_T])

        self.action_space = [gym.spaces.Discrete(2) for n in range(self.n_agents)]
        self.observation_space = [Box(low=self.low, high=self.high, dtype=np.float64)
                                  for n in range(self.n_agents)]
        self.share_observation_space = self.observation_space.copy()
        self.pre_obs = None
        self.done = False
        self.t = 0
        self.current_agent = 0

        self.obs = np.array(
            (600, 10, self.load[0], self.load[1], self.Trans_speed[1][0], self.Trans_speed[1][1], 0.0))
        self.rewards = []
        self.dones = [False * self.n_agents]
        self.infos = [0 * self.n_agents]
        self.state = [np.array(
            (600, 10, self.load[0], self.load[1], self.Trans_speed[1][0], self.Trans_speed[1][1], 0.0))for _ in range(self.n_agents)]
        self.ava=[]
        for _ in range(self.n_agents):
            self.ava.append((np.array(([1,1]))))

    def reset(self, **kwargs):
        """ Returns initial observations and states"""
        # 重置相关信息
        self.done = False
        self.rewards = []
        self.dones = [False * self.n_agents]
        self.infos = [0 * self.n_agents]

        self.t = 0
        self.current_agent = 0
        self.load = np.array((0.0, 0.0, 0.0))
        self.obs = np.array(
            (600, 10, self.load[0], self.load[1], self.Trans_speed[1][0], self.Trans_speed[1][1], 0.0))
        self.state = [np.array(
            (600, 10, self.load[0], self.load[1], self.Trans_speed[1][0], self.Trans_speed[1][1], 0.0))for _ in range(self.n_agents)]


        return self.state, self.state, self.ava

    def step(self, actions):
        obs = []
        rewards = []
        dones = []
        infos = []
        for i in range(self.n_agents):
            action = actions[i]
            o, r, d, info = self.agent_step(action, i)
            obs.append(o)
            rewards.append(r)
            dones.append(d)
            infos.append(info)

        state = obs.copy()

        for i in range(3):
            if (self.load[i] - self.C[i] * self.deltaT / self.Ptask > 0):

                self.load[i] -= self.C[i] * self.deltaT / self.Ptask
            else:
                self.load[i] = 0.0

        return obs, state, rewards, dones, infos, self.ava

    def agent_step(self, action, i):
        # 计数器，从0-9
        t = self.t
        e = self.e[t]
        # 判断是否结束
        action = np.asarray(action)  # 一个数字：0/1
        Tt = self.Ttrans[t, e]
        """ 256 320 378? 512
            0:迁移且预迁移   -264 -126
            1:迁移但不预迁移 -275 -134
            rand: -240 -119
            train： -223 -112./
        """
        #action = np.random.randint(0,1)
        #action= 1
        if (action == 0):
            adl = 0.51
            adt = 0.5
        if (action == 1):
            adl = 1.0
            adt = 1.0

        if (((self.load[e] + adt) * self.Ptask - self.C[e] * self.dt[t]) > 0):
            if ((self.load[e] * self.Ptask - self.C[e] * self.dt[t]) < 0):
                dmig = (self.load[e] + adt) * self.Ptask - self.C[e] * self.dt[t]
                ndmig = self.Ptask * adt - dmig
                # print(dmig)
                if (action == 0):
                    tc1 = 0
                    tc2 = (self.Ptask * adt + dmig) / self.C[e + 1]

                    e1t = self.dt[t] + tc1
                    tmig = (self.Ptask * adt + dmig) / self.bw
                    e2t = tmig + self.load[e + 1] * self.Ptask / self.C[e + 1] + tc2
                    if (e2t <= e1t):
                        Tmwc = e1t
                    else:
                        Tmwc = e1t + (e2t - self.dt[t])

                    Td = ndmig / self.Trans_speed[t + 1][e] + (self.Ptask * adl + dmig) / self.Trans_speed[t + 1][e + 1]

                    self.load[e] += adl
                    self.load[e + 1] += adl + dmig / self.Ptask

                if (action == 1):
                    tc1 = 0
                    tc2 = dmig / self.C[e + 1]

                    e1t = self.dt[t] + tc1
                    tmig = dmig / self.bw
                    e2t = tmig + self.load[e + 1] * self.Ptask / self.C[e + 1] + tc2
                    if (e2t <= e1t):
                        Tmwc = e1t
                    else:
                        Tmwc = e1t + e2t

                    Td = ndmig / self.Trans_speed[t + 1][e] + dmig / self.Trans_speed[t + 1][e + 1]

                    self.load[e] += adl
                    self.load[e + 1] += 0.1*dmig / self.Ptask +(0.1-adl)
            else:
                tc1 = 0
                tc2 = self.Ptask * adt / self.C[e + 1]
                dmig = self.Ptask * adt
                tmig = dmig / self.bw
                if (action == 0):

                    if (((self.load[e + 1] + adl) * self.Ptask - self.C[e] * self.dt[t]) <= 0):
                        Tmwc = self.dt[t] + tmig + self.load[e + 1] * self.Ptask / self.C[e + 1] + tc2
                    else:
                        Tmwc = tmig + self.load[e + 1] * self.Ptask / self.C[e + 1] + tc2 + tc2
                if (action == 1):
                    Tmwc = self.dt[t] + tmig + self.load[e + 1] * self.Ptask / self.C[e + 1] + tc2

                Td = self.Ptask * adt / self.Trans_speed[t+1][e + 1]  # ??*or/

                self.load[e + 1] += adl

        else:
            if (action == 0):
                tc1 = self.Ptask * adt / self.C[e]
                tc2 = self.Ptask * adt / self.C[e + 1]

                e1t = self.load[e] * self.Ptask / self.C[e] + tc1
                tmig = (self.Ptask * adt) / self.bw
                # print(tmig)
                e2t = tmig + self.load[e + 1] * self.Ptask / self.C[e + 1] + tc2
                if (e2t <= self.dt[t]):
                    Tmwc = e1t
                    # print(Tmwc, t, 0)
                else:
                    Tmwc = e1t + (e2t - self.dt[t])
                    # print(Tmwc, t, 1)
                Td = self.Ptask * adt / self.Trans_speed[t][e] + self.Ptask * adt / self.Trans_speed[t+1][e + 1]
                self.load[e] += adl
                self.load[e + 1] += adl

            if (action == 1):  # 不预迁移

                tc1 = self.Ptask * adt / self.C[e]
                Tmwc = self.load[e] * self.Ptask / self.C[e] + tc1
                Td = self.Ptask * adt / self.Trans_speed[t][e]
                # print(Tmwc, t)
                self.load[e] += adl

        # sys.stdout.write("\r{}".format(self.load))
        # sys.stdout.flush()

        # 总时延
        T = Tt + Tmwc + Td

        x = int(self.u_locx[t])
        y = int(self.u_locy[t])
        load = self.load[e]
        load_ = self.load[e + 1]
        self.obs = np.array((x, y, load, load_, self.Trans_speed[t + 1][e], self.Trans_speed[t + 1][e + 1], T))
        # 奖励
        reward = np.array([-T])
        # 计数
        if i == self.n_agents - 1:
            self.t += 1

        if self.t >= 9:  # self.Tmax:
            self.done = True
        info = [Tmwc,Td]

        return self.obs, reward, self.done, info

    def render(self, **kwargs):
        # self.env.render(**kwargs)
        pass

    def close(self):
        pass

    def seed(self, args):
        pass

    def get_env_info(self):
        env_info = {"state_shape": self.observation_space[0].shape,
                    "obs_shape": self.observation_space[0].shape,
                    "n_actions": self.action_space[0].n,
                    "n_agents": self.n_agents,
                    "action_spaces": self.action_space
                    }
        return env_info
