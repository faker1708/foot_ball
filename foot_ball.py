import math
import random


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

import matplotlib.pyplot as plt

import pickle
import time


import copy

import numpy as np
import matplotlib.pyplot as plt
import torch
from RL_brain import PPO


#　前端
import pygame



class foot_ball():


    def action_decode(self,action):
        # 一维到多维的变换

        aa = action % (self.na)
        bb = action // (self.na)

        # 归一化
        aa = aa/self.na
        bb = bb/(self.nb-1)
        # bb = bb/self.nb

        return [aa,bb]


    def fb(self,mode,num_episodes):
        
        long = self.long
        wide = self.wide

        gate_wide = self.gate_wide

        piece = self.piece # 时间片，模拟器精度
        aa = self.aa  # 球在草地上的加速度

        if(mode == 'show'):

            kv = 0.3

            self.wold_x = 1920*kv
            self.wold_y = 1080*kv

            #使用pygame之前必须初始化
            pygame.init()
            pygame.display.set_caption('foot_ball')
        
            self.screen = pygame.display.set_mode((self.wold_x,self.wold_y))

        sum = 0
        
        for i in range(num_episodes):
            
            # print(mode,i)
            
            bx = random.randint(91,long)
            by = random.randint(wide*1/4,wide*3/4)
            vx = 0
            vy = 0

            px = bx
            py = by
            pvx = 0
            pvy = 0

            # state = env.reset()[0]  # 环境重置
            state =   np.array([bx,by,vx,vy,px,py,pvx,pvy]) 
            # print(state)

            done = 0  # 任务完成的标记

            if(mode == 'train'):

                # 构造数据集，保存每个回合的状态数据
                transition_dict = {
                    'states': [],
                    'actions': [],
                    'next_states': [],
                    'rewards': [],
                    'dones': [],
                }


            t = 0   # 模拟器 tick

            out_bottom = 0  # 球出底线

            while (1):

                action = self.agent.take_action(state)

                la = self.action_decode(action)
                angle = la[0]
                velocity = la[1]

                
                # angle = angle*360


                if(t==0):
                    # 角色一直在决策，但只有最开始时，能踢到球。
                    v = velocity* 30
                    theta = angle* math.pi*2
                    sin = math.sin(theta)
                    cos = math.cos(theta)
                else:
                    v = v-aa

                
                if(bx>long):
                    out_bottom =1
                    done = 1
                else:
                

                    if(v<2**-10):
                        done = 1    # 球停了就结束 ，不许人移动。



                
                # 位移的微分
                vy = v* sin * piece
                vx = v*cos * piece

                bx += vx 
                by += vy 

                

                next_state =   np.array([bx,by,vx,vy,px,py,pvx,pvy]) 


                if(mode == 'train'):

                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(0)
                    transition_dict['dones'].append(done)



                if(done):
                    break
                else:
                    state = next_state

                    if(mode == 'show'):
                        time.sleep(0.01)

                        table_color = 'green'
                        self.screen.fill(table_color)


                        sbx = bx*(self.wold_x/long)
                        sby = by*(self.wold_y/wide)

                        pygame.draw.circle(self.screen, 'white', (sbx,sby), 8, width=0)

                        spx = px*(self.wold_x/long)
                        spy = py*(self.wold_x/long)
                        pygame.draw.circle(self.screen, 'red', (spx,spy), 16, width=0)

                        # print(bx,by)
                        
                        pygame.display.flip() #更新屏幕内容

                    t+=1


            goal =0
            if(out_bottom):
                if(by>wide/2-gate_wide/2):
                    if(by<wide/2+gate_wide/2):
                        # print(mode,'goal \a')
                        goal = 1

                

            if(mode == 'train'):
                ll = transition_dict['rewards']
                for ii ,ele in enumerate(ll):
                    ll[ii] = goal    # 这样踢能进球
                
                
                # print('learn')
                self.agent.learn(transition_dict)
            sum+=goal
        out = sum/num_episodes


        return out


    def main(self):
        
        # 多维离散动作空间
        angle_n = 2**4
        velocity_n = 7

        # 状态空间
        n_states = 8
        
        device = torch.device('cuda') if torch.cuda.is_available() \
                                    else torch.device('cpu')

        gamma = 0.9  # 折扣因子
        actor_lr = 1e-3  # 策略网络的学习率
        critic_lr = 1e-2  # 价值网络的学习率
        

        self.na = angle_n
        self.nb = velocity_n

        n_actions = angle_n*velocity_n  # 动作数 2
        # self.n_states = n_states
        self.n_actions = n_actions
        
        self.agent = PPO(n_states=n_states,  # 状态数
                    n_actions=n_actions,  # 动作数
                    actor_lr=actor_lr,  # 策略网络学习率
                    critic_lr=critic_lr,  # 价值网络学习率
                    lmbda = 0.95,  # 优势函数的缩放因子
                    epochs = 10,  # 一组序列训练的轮次
                    eps = 0.2,  # PPO中截断范围的参数
                    gamma=gamma,  # 折扣因子
                    device = device
                    )
        
        
        self.return_list = []  # 保存每个回合的return

        
        self.long = 113
        self.wide = 76

        self.gate_wide = 7.32

        self.piece = 0.1 # 时间片，模拟器精度
        self.aa = 1   # 球在草地上的加速度
        
    
        # self.show(4)
        self.fb('show',4)
        while(1):
            test = self.fb('train',2**8)
            # test= self.fb('test',2**4)
            print('test',test)
            self.fb('show',4)

    def __init__(self):

        # for i in range(16):
            # result = self.fa(0,i/16)
            # print(result)
        self.main()

    
if __name__ == "__main__":
    foot_ball()