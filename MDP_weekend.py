import os, sys
import time
import datetime
import pandas as pd
import numpy as np
import math
from math import radians, cos, sin, asin, sqrt 
import random
import copy


import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.patches import Ellipse, Circle

ROOTDIR = os.path.abspath(os.path.realpath('./')) + '/Py'

sys.path.append(os.path.join(ROOTDIR, ''))

import dgckernel

class Stamp_transition(object):
    
    def __init__(self, **kwargs):
        """ Load your trained model and initialize the parameters """
        pass
    
    def Get_date(self,stamp):
        dateArray = datetime.datetime.fromtimestamp(stamp)
        otherStyleTime = dateArray.strftime("%Y-%m-%d %H:%M:%S")
        return otherStyleTime[:10]
    
    '''Time stamp'''
    def Get_stamp(self,time_str):
        timeArray = time.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        timeStamp = int(time.mktime(timeArray))
        return timeStamp
    
    '''Time step'''
    def Get_step(self,stamp,date_str,step):
        baseline = date_str+" 00:00:00";
        baseline = int(self.Get_stamp(baseline))
        current_step=int((stamp-baseline)/step)
        return current_step
    
    def Get_datelist(self, beginDate, endDate):
        date_list=[datetime.datetime.strftime(x,'%Y-%m-%d') for x in list(pd.date_range(start=beginDate, end=endDate))]
        return date_list
    
    def Get_weekday(self,date_str):
        date_str = date_str+" 00:00:00";
        date_str = time.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        return date_str.tm_wday+1

    def Get_normalization(self,end_step,t_step):
        if t_step>=end_step:
            return t_step-end_step
        else:
            return t_step


class Spatial_calculation(object):
    
    def __init__(self, Zoom):
        """ Load your trained model and initialize the parameters """
        self.Zoom=Zoom
        self.CALCULATOR = dgckernel.Calculator()
        self.CALCULATOR.SetLayer(Zoom)
        
    '''GRID ID'''

    def get_grid(self,lng,lat):

        return self.CALCULATOR.HexCellKey(dgckernel.GeoCoord(lat, lng))

    '''GRID SHAPE'''

    def get_grid_shape(self,grid):

        return self.CALCULATOR.HexCellVertexesAndCenter(grid)
        
    '''Neighbor Grid'''

    def grid_neighbor(self, grid, low_layer, up_layer):

        neighbors = self.CALCULATOR.HexCellNeighbor(grid, up_layer)
        _neighbors = self.CALCULATOR.HexCellNeighbor(grid, low_layer)
        neighbors = [e for e in neighbors if e not in _neighbors]
        return neighbors 
    
    def grid_eliminate(self,grid_list,sw,ne):
        grid_result=list()
        for grid in grid_list:
            v_f,c_f=self.get_grid_shape(grid)
            c_lng,c_lat=c_f.lng,c_f.lat;
            if c_lng>=sw[1] and c_lng<=ne[1] and c_lat>=sw[0] and c_lat<=ne[0]:
                grid_result.append(grid)
        return grid_result
    
    def Geo_distance(self,lng1,lat1,lng2,lat2):
        lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)]) 
        dlon=lng2-lng1
        dlat=lat2-lat1
        a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2 
        distance=2*asin(sqrt(a))*6371 
        distance=round(distance,2)
        return distance
    
    '''Get Distance'''
    def get_grid_dis(self,f_grid,t_grid):

        f_shape,f_center=self.get_grid_shape(f_grid);
        t_shape,t_center=self.get_grid_shape(t_grid);

        Topology_dis=1.3*self.Geo_distance(f_center.lng,f_center.lat,t_center.lng,t_center.lat)

        return Topology_dis

class MDP(object):
    
    def __init__(self,Grid,end_step,Reward_dic,Capacity_dic,Dispatch_PROB,Request_PROB):
        
        '''Input'''
        
        self.Grid=Grid
        
        self.end_step=end_step
        
        '''Param'''
        
        self.param_limit=5
        
        self.layer_limit=50
        
        self.iteration=100
        
        self.gumma=0.8
        
        '''Feature'''
        
        self.Reward_dic=Reward_dic
        
        self.Capacity_dic=Capacity_dic
        
        '''State'''
        
        self.State=list()
        for grid in self.Grid:
            for t_step in range(self.end_step):
                self.State.append(str(grid)+'-'+str(t_step))
                
        '''Action'''
        
        self.Action={}
        
        self.Get_Action()
        
        np.save(os.path.join(MODEL_PATH,'Action_weekend.npy'),self.Action)
        
        self.Action=np.load(os.path.join(MODEL_PATH,'Action_weekend.npy')).item()
        
        '''Probability'''
        
        self.Match_PROB=self.Get_Match_Prob()
        
        self.Dispatch_PROB=Dispatch_PROB
        
        self.Request_PROB=Request_PROB
        
        '''Gain'''
        
        self.Gain=self.Get_Gain()
        
        
        '''Reward'''
        
        self.Reward=self.Get_reward()
        
        '''State Value'''
        
        self.State_value={state:0 for state in self.State}
        
        
    def Get_Action(self):

        count=0

        for grid in self.Grid:

            for t_step in range(self.end_step):

                layer=4

                state=grid+'-'+str(t_step)

                Action_list=list()

                travel_time=1+int(layer/5)

                Action_candidate=[str(travel_time)+'-'+grid]+\
                [str(travel_time)+'-'+g for g in spatial_calculation.grid_neighbor(grid, 0, 1) if g in self.Grid]+\
                [str(travel_time)+'-'+g for g in spatial_calculation.grid_neighbor(grid, 1, 2) if g in self.Grid]+\
                [str(travel_time)+'-'+g for g in spatial_calculation.grid_neighbor(grid, 2, 3) if g in self.Grid]

                Action_state=list()

                for a in Action_candidate:

                    dest_step=t_step+travel_time

                    dest_state=a.split('-')[1]+'-'+str(dest_step) 

                    Action_state.append(dest_state)

                '''Calculate the Action's weight'''

                Action_weight={}

                for i in range(len(Action_state)):

                    if Action_state[i] in self.Reward_dic.keys():

                        if self.Reward_dic[Action_state[i]]>0:

                            Action_weight[i]=self.Reward_dic[Action_state[i]]*self.Capacity_dic[Action_state[i]]

                shortlist=sorted(Action_weight.items(),key=lambda item:item[1],reverse=True)[0:5]

                Action_list=[Action_candidate[idx[0]] for idx in shortlist]

                '''if the number of available actions is less than 5, magnify the searching area'''

                while len(Action_list)<self.param_limit:

                    travel_time=1+int(layer/5)

                    Action_candidate=[str(travel_time)+'-'+g for g in spatial_calculation.grid_neighbor(grid, layer-1, layer) if g in self.Grid]

                    '''Generate Action state'''

                    Action_state=list()

                    for a in Action_candidate:

                        dest_step=t_step+travel_time

                        dest_state=a.split('-')[1]+'-'+str(dest_step) 

                        Action_state.append(dest_state)

                    Action_weight={}

                    '''Calculate the Action's weight'''

                    for i in range(len(Action_state)):

                        if Action_state[i] in self.Reward_dic.keys():

                            if self.Reward_dic[Action_state[i]]>0:

                                Action_weight[i]=self.Reward_dic[Action_state[i]]*self.Capacity_dic[Action_state[i]]

                    shortlist=sorted(Action_weight.items(),key=lambda item:item[1],reverse=True)[0:5]

                    Action_list=Action_list+[Action_candidate[idx[0]] for idx in shortlist]

                    if layer>self.layer_limit:

                        if len(Action_list)==0:

                            Action_list=['1'+'-'+grid]

                        break

                    layer+=1
                    
                count+=1

                print(count)

                self.Action[state]=Action_list

                
    def Get_Match_Prob(self):
        
        Get_match_prob=lambda x: round(x/5.0,2) if x<=5 else 1.0

        Match_PROB_Dic={}

        for state in self.State:

            if state in self.Capacity_dic.keys():

                Match_PROB_Dic[state]=Get_match_prob(self.Capacity_dic[state])

            else:

                Match_PROB_Dic[state]=0.0 
                
        return Match_PROB_Dic
    
    def Get_Gain(self):
        
        Gain_table_Dic={}

        for state in self.State:

            if state in self.Reward_dic.keys():

                Gain_table_Dic[state]=self.Reward_dic[state]

            else:

                Gain_table_Dic[state]=0.0  

        return Gain_table_Dic
    
    def Get_reward(self):

        Reward={}

        for state in self.State:

            Reward[state]={}

            current_grid=state.split('-')[0];

            current_stamp=int(state.split('-')[1])

            for action in self.Action[state]:

                next_grid=action.split('-')[1]

                travel=int(action.split('-')[0])

                next_stamp=int(current_stamp)+travel      

                next_state=str(next_grid)+'-'+str(next_stamp)

                if next_stamp<self.end_step:

                    '''cruise_cost'''

                    cruise_cost=travel

                    '''Match'''

                    Prob=self.Match_PROB[next_state]

                    gain=0

                    '''Dispatch'''

                    if next_state in self.Dispatch_PROB.keys():

                        for pickup_state,pickup_prob in self.Dispatch_PROB[next_state].items():

                            pickup_stamp=int(pickup_state.split('-')[1])

                            if pickup_stamp<self.end_step and pickup_state in self.Gain.keys():

                                    pickup_award=self.Gain[pickup_state]

                                    gain+=pickup_prob*pickup_award

                    pickup_order=Prob*gain

                    Reward[state][action]=round(pickup_order-cruise_cost,2)
                    
        return Reward
    
    def Dynamic_programming(self):
        
        diff=0
        
        Value_sum={}
        
        Pre_Value_sum=0

        for i in range(self.iteration):

            count=0

            for state in self.State:

                self.State_value[state]=self.Bellman(state)

                count+=1

                print(i,count,state,self.State_value[state])

            Value_sum[i]=sum(self.State_value.values())

            Origin_policy=self.Update_action()

            diff=abs(Value_sum[i]-Pre_Value_sum)

            if diff<5:

                break

            pre_V_sum=Value_sum[i]



        Value_statistic=pd.DataFrame(Value_sum.keys(),columns=['iteration'])
        Value_statistic['V_sum']=Value_sum.values()
        Value_statistic.to_csv(os.path.join(MODEL_PATH,'Value_sum_weekend.csv'))

        State_value_table=pd.DataFrame(self.State_value.keys(),columns=['State'])
        State_value_table['value']=self.State_value.values()
        State_value_table.to_csv(os.path.join(MODEL_PATH,'State_value_weekend.cscv'))

    
    def Bellman(self,state):
    
        '''current location and step'''
        current_grid=state.split('-')[0]
        current_stamp=int(state.split('-')[1])

        '''Bellman Iteration'''

        Action_prob=round(float(1)/len(self.Action[state]),3)

        V=0

        for action in self.Action[state]:

            next_grid=action.split('-')[1]
            travel_time=int(action.split('-')[0])
            next_stamp=int(current_stamp+travel_time)
            next_state=next_grid+'-'+str(next_stamp)

            if next_stamp>=self.end_step:

                continue

            '''reward'''

            R=Action_prob*self.Reward[state][action]

            '''fail to find an order'''

            match_prob=self.Match_PROB[next_state]

            V_=(1-match_prob)*self.State_value[next_state]

            '''find an order'''

            if next_state in self.Dispatch_PROB.keys():
                
                '''If Dispacth'''

                for pickup_state,pickup_prob in self.Dispatch_PROB[next_state].items():
                    
                        '''If Pickup'''

                        if pickup_state in self.Request_PROB.keys():

                            for dest_state,request_prob in self.Request_PROB[pickup_state].items():

                                dest_time=int(dest_state.split('-')[1])

                                if dest_time<end_step and dest_state in self.State_value.keys():

                                    V_+=match_prob*pickup_prob*request_prob*self.State_value[dest_state]

            V+= R+ Action_prob*self.gumma*V_                 

        return V
    
    def Update_action(self):
    
        '''Greedy selection'''

        for state,action_list in self.Action.items():

            Next_value={}

            current_grid=state.split('-')[0]

            current_stamp=int(state.split('-')[1])

            for action in action_list:

                next_grid=action.split('-')[1]

                travel_time=int(action.split('-')[0])

                next_stamp=int(current_stamp+travel_time)

                next_state=next_grid+'-'+str(next_stamp)

                if next_stamp>=self.end_step:

                    continue

                Next_value[action]=self.State_value[next_state]

            if len(Next_value)!=0:

                Max_value=max(Next_value.values())

                for action,action_value in Next_value.items():

                    if action_value<Max_value:

                        self.Action[state].remove(action)  


if __name__ == '__main__':

    '''Time range'''

    end_step=144

    spatial_calculation=Spatial_calculation(13)

    stamp_transition=Stamp_transition()

    '''Reward table'''

    MODEL_PATH='./kddcup-testing/model/modelfile/'

    All_grid = np.load(os.path.join(MODEL_PATH,'All_grid_weekend.npy'))

    Reward_dic = np.load(os.path.join(MODEL_PATH,'Reward_dic_weekend.npy')).item()

    Capacity_dic = np.load(os.path.join(MODEL_PATH,'Capacity_dic_weekend.npy')).item()

    Dispatch_PROB = np.load(os.path.join(MODEL_PATH,'Dispatch_PROB_weekend.npy')).item()

    Request_PROB = np.load(os.path.join(MODEL_PATH,'Request_PROB_weekend.npy')).item()


    mdp=MDP(All_grid,end_step,Reward_dic,Capacity_dic,Dispatch_PROB,Request_PROB)

    mdp.Dynamic_programming()

    


















            
            


