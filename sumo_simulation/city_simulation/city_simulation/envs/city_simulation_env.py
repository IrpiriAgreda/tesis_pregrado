import gym
import pandas as pd
import os, sys
import _pickle as cPickle
import numpy as np
import torch
import random
from gc import collect

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci

class CitySimulation(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.state = np.zeros(12)
        self.done = 0
        self.reward = 0
        self.sumoCmd = ["/usr/bin/sumo/bin/sumo", "-c", "../sumo_simulation/sim_config/osm.sumocfg"]
        self.steps_in_hour = 500
        self.zone_lane_mapper = None#cPickle.load(open('../sumo_simulation/input/action_to_zone.pkl', 'rb'))
        self.fis = cPickle.load(open('../sumo_simulation/input/fis_v2.pkl', 'rb'))
        self.umap = None #cPickle.load(open('../sumo_simulation/input/umap.pkl', 'rb'))
        #self.pca = cPickle.load(open('../sumo_simulation/input/pca_v2.pkl', 'rb')) #95% variance PCA shape (6,1)
        self.action_space = gym.spaces.MultiDiscrete([2 for i in range(5981)])

        self.observation_space = gym.spaces.Box(low=0, high=np.infty, shape=(6, 5981))
        self.id_list = []
        self.scenarios = ['km2_centro', 'km2_centro_2', 'km2_jesusmaria', 'km2_lince',] #'km2_miraflores', 'km2_sani']
        self.sim_configs = {
            'km2_centro': "../sumo_simulation/sim_config/km2_centro/scenario/osm.sumocfg",
            'km2_centro_2': "../sumo_simulation/sim_config/km2_centro_2/scenario/osm.sumocfg",
            'km2_jesusmaria': "../sumo_simulation/sim_config/km2_jesusmaria/scenario/osm.sumocfg",
            'km2_lince': "../sumo_simulation/sim_config/km2_lince/scenario/osm.sumocfg",
            'km2_miraflores': "../sumo_simulation/sim_config/km2_miraflores/scenario/osm.sumocfg",
            'km2_sani': "../sumo_simulation/sim_config/km2_sani/scenario/osm.sumocfg"
        }
        self.action_dicts = {
            'km2_centro': cPickle.load(open('../sumo_simulation/input/action_to_zone_km2_centro.pkl', 'rb')),
            'km2_centro_2': cPickle.load(open('../sumo_simulation/input/action_to_zone_km2_centro_2.pkl', 'rb')),
            'km2_jesusmaria': cPickle.load(open('../sumo_simulation/input/action_to_zone_km2_jesusmaria.pkl', 'rb')),
            'km2_lince': cPickle.load(open('../sumo_simulation/input/action_to_zone_km2_lince.pkl', 'rb')),
            'km2_miraflores': cPickle.load(open('../sumo_simulation/input/action_to_zone_km2_miraflores.pkl', 'rb')),
            'km2_sani': cPickle.load(open('../sumo_simulation/input/action_to_zone_km2_sani.pkl', 'rb'))
        }

        self.umaps = {
            'km2_centro': cPickle.load(open('../sumo_simulation/input/umap_km2_centro.pkl', 'rb')),
            'km2_centro_2': cPickle.load(open('../sumo_simulation/input/umap_km2_centro_2.pkl', 'rb')),
            'km2_jesusmaria': cPickle.load(open('../sumo_simulation/input/umap_km2_jesusmaria.pkl', 'rb')),
            'km2_lince': cPickle.load(open('../sumo_simulation/input/umap_km2_lince.pkl', 'rb')),
            'km2_miraflores': cPickle.load(open('../sumo_simulation/input/umap_km2_miraflores.pkl', 'rb')),
            'km2_sani': cPickle.load(open('../sumo_simulation/input/umap_km2_sani.pkl', 'rb'))
        }

    def step(self, action):
        '''
        Action represents logits from Agent. Probably MultiDescrete input.
        Must check if tf-agents outputs 1s and 0s or if a step function is needed.
        '''

        self.assignAllowedVehicles(action)
        reward_means = self.runSimulationSteps()
        #self.iteration_counter += 1
        #self.reward = self.get_reward(reward_means)
        rew = [traci.edge.getLastStepMeanSpeed(edge_id) for edge_id in id_list]
        self.reward = sum(rew)/len(rew)
        collect()

        return [self.state, self.reward, self.done, {}] #key: measure for key, measure in reward_means.iteritems()

    def reset(self):
        '''
        Generate simulation connection with libsumo.
        Creates subscriptions to emission and fuel consumption per lane for performance
        '''
        # try:
        #     traci.close()
        # except:
        #     pass

        chosen_scenario = random.choice(self.scenarios)
        self.sumoCmd[-1] = self.sim_configs[chosen_scenario]
        self.zone_lane_mapper = self.action_dicts[chosen_scenario]
        self.umap = self.umaps[chosen_scenario]

        traci.start(self.sumoCmd)

        self.id_list = traci.edge.getIDList()
        # for edge_id in traci.lane.getIDList():
        #     traci.lane.subscribe(edge_id, [traci.constants.VAR_CO2EMISSION,
        #                                    traci.constants.VAR_COEMISSION,
        #                                    traci.constants.VAR_PMXEMISSION,
        #                                    traci.constants.VAR_NOXEMISSION,
        #                                    traci.constants.VAR_NOISEEMISSION,
        #                                    traci.constants.VAR_FUELCONSUMPTION])

        self.state = np.zeros(12)
        self.done = 0
        self.reward = 0
        collect()

    def render(self):
        'Returns simulation state transformed via PCA'
        return self.state

    #Helper functions
    def assignAllowedVehicles(self, action):
        '''
        Recieve an action tensor and alter allowed vehicles per lane selected
        Must recieve 1s and 0s in array
        '''
        #NumPy version
        #lane_indices = np.where(action == 1)[0]

        lane_indices = (action == 1).nonzero().view(-1)

        for lane_id in lane_indices:
            if self.zone_lane_mapper[lane_id.item()] is not None:
                traci.lane.setDisallowed(self.zone_lane_mapper[lane_id.item()], ['truck'])

        return 1

    def runSimulationSteps(self):
        '''
        This function runs a block of SUMO simulations and returns the emission and
        fuel consumption state. The state the agent percieves will be
        '''

        subscriptions_list = []

        for _ in range(self.steps_in_hour):
            #Check if all vehicles have left the simulation
            if traci.simulation.getMinExpectedNumber() == 0:
                self.done = 1
                sim_results = pd.DataFrame.from_dict(traci.lane.getAllSubscriptionResults())
                # subscriptions_list.append(pd.DataFrame.from_dict(traci.lane.getAllSubscriptionResults()).values)
                # sim_results = np.sum(subscriptions_list, axis=0)
                self.state =  self.umap.transform(sim_results.values) #sim_results.values # self.pca.transform(sim_results.values)
                traci.close(False)
                collect()
                #Returns state means for reward calculation
                return sim_results.T.mean()

            else:
                traci.simulationStep()
                #subscriptions_list.append(pd.DataFrame.from_dict(traci.lane.getAllSubscriptionResults()).values)
                collect()

        sim_results = pd.DataFrame.from_dict(traci.lane.getAllSubscriptionResults()) #np.sum(subscriptions_list, axis=0)
        self.state = self.umap.transform(sim_results.values)  #self.umap.transform(sim_results.values) self.pca.transform(sim_results.values)
        collect()

        return sim_results.T.mean() #pd.DataFrame(sim_results, index=[96,97,99,100,101,102]).T.mean()

    def runSimulationSteps_v2(self):
        if traci.simulation.getMinExpectedNumber() == 0:
            self.done = 1
            #sim_results = pd.DataFrame.from_dict(traci.lane.getAllSubscriptionResults())

            co2 = [traci.edge.getCO2Emission(edge_id) for edge_id in id_list]
            co = [traci.edge.getCOEmission(edge_id) for edge_id in id_list]
            nox = [traci.edge.getNOxEmission(edge_id) for edge_id in id_list]
            pmx = [traci.edge.getPMxEmission(edge_id) for edge_id in id_list]
            noise = [traci.edge.getNoiseEmission(edge_id) for edge_id in id_list]
            fuel = [traci.edge.getFuelConsumption(edge_id) for edge_id in id_list]

            sim_results = np.array([co2, co, pmx, nox, noise, fuel])

            self.state =  self.umap.transform(sim_results.values)
            traci.close(False)
            collect()
            return 1#sim_results.T.mean()
        else:
            traci.simulationStep(500)
            co2 = [traci.edge.getCO2Emission(edge_id) for edge_id in id_list]
            co = [traci.edge.getCOEmission(edge_id) for edge_id in id_list]
            nox = [traci.edge.getNOxEmission(edge_id) for edge_id in id_list]
            pmx = [traci.edge.getPMxEmission(edge_id) for edge_id in id_list]
            noise = [traci.edge.getNoiseEmission(edge_id) for edge_id in id_list]
            fuel = [traci.edge.getFuelConsumption(edge_id) for edge_id in id_list]


            collect()

        sim_results = np.array([co2, co, pmx, nox, noise, fuel])
        self.state = self.umap.transform(sim_results.values)
        return 1#sim_results.T.mean()

    def get_reward(self, means):
        '''
        This function evaluates the state of a simulation with a Fuzzy Inference System (FIS)
        The input indices are based on the SUMO variable id
        @param means: pd.Series of co2, co, pmx, nox, noise and fuel means in the simulation state
        returns: [0,100] score based on the FIS rules and membership functions (see notebooks/Fuzzy Models)
        '''

        self.fis.input['co2'] = means[96]
        self.fis.input['co'] = means[97]
        self.fis.input['pmx'] = means[99]
        self.fis.input['nox'] = means[100]
        self.fis.input['noise'] = means[101]
        self.fis.input['fuel'] = means[102]
        self.fis.compute()

        return self.fis.output['output']
