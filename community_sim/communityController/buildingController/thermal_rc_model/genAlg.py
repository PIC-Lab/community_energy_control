import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt

from neuromancer.psl.coupled_systems import *
from neuromancer.dynamics import integrators, ode, physics, interpolation
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable
from neuromancer.problem import Problem
from neuromancer.loss import PenaltyLoss
from neuromancer.system import Node, System
from neuromancer.loggers import BasicLogger
from neuromancer.trainer import Trainer
from neuromancer.modules import blocks
from neuromancer.callbacks import Callback

from galibrate.sampled_parameter import SampledParameter
from galibrate import GAO

def MainPrefab():
    param_names = list(["R", "C", "a"])
    sampled_parameters = [SampledParameter(name=p, loc=0, width=20.0) for p in param_names]

    population_size = 200

    wrapper = RCWrapper()

    gao = GAO(sampled_parameters, wrapper.fitness, population_size, generations=100, mutation_rate=0.1)

    best_theta, best_theta_fitness = gao.run()

    print("Fittest theta {} with fitness value {}".format(best_theta, best_theta_fitness))

    fig, ax = plt.subplots(figsize=(12, 18))
    ax.plot(gao.best_fitness_per_generation)
    fig.savefig('Saved Figures/gal_generations.png')

    fig, axs = plt.subplots(3, 1, figsize=(12, 18))
    axs[0].scatter(gao.final_population[:,0], gao.final_population_fitness, color='blue', label='a')
    axs[0].set_ylabel('a', color='blue')
    axs[0].legend(loc='upper left')
    
    axs[1].scatter(gao.final_population[:,1], gao.final_population_fitness, color='green', label='b')
    axs[1].set_ylabel('b', color='green')
    axs[1].legend(loc='upper left')
    
    axs[2].scatter(gao.final_population[:,2], gao.final_population_fitness, color='red', label='c')
    axs[2].set_ylabel('c', color='red')
    axs[2].set_xlabel('Individual Index')
    axs[2].legend(loc='upper left')
    
    axs[0].set_title(f'Final Generation ({gao.generations}) Population Solutions')
    fig.savefig('Saved Figures/gal_lastGeneration.png')

    trajectories = wrapper.model(wrapper.data)
    pred_traj = trajectories['xn'][:,:-1,:].detach().numpy().reshape(-1, 1)

    true_traj = wrapper.data['X'].detach().numpy().reshape(-1,1)
    input_traj = wrapper.data['U'].detach().numpy().reshape(-1,2)

    fig, ax = plt.subplots(2, figsize=(10,5))
    u = torch.from_numpy(input_traj).float()
    sol = torch.zeros((true_traj.shape[0],1))
    ic = torch.tensor(true_traj[0,:])
    for j in range(sol.shape[0]-1):
        if j==0:
            sol[[0],:] = ic.float()
            sol[[j+1],:] = wrapper.fxRK4(sol[[j],:],u[[j],:])
        else:
            sol[[j+1],:] = wrapper.fxRK4(sol[[j],:],u[[j],:])

        
    ax[0].plot(sol.detach().numpy(), label='model', color='black')
    ax[0].plot(pred_traj, label='pred')
    ax[0].plot(true_traj, label='data', color='red')
    ax[0].legend()
    ax[1].plot(input_traj, label='input')
    for x in ax:
        x.set_xlim([0,1000])
    plt.legend()
    fig.savefig('./Saved Figures/gal_tempPred', dpi=fig.dpi)

class RCWrapper():
    def __init__(self):
        nx = 1
        nu = 2
        ts = 1
        nsteps = 60
        bs = 32

        df = pd.read_csv('1_out.csv', nrows=57600, usecols=['Site Outdoor Air Temperature', 'living space Air Temperature', 'Cooling:Electricity', 'Whole Building Electricity'])
        dataset = pd.DataFrame()
        dataset['X1'] = df['living space Air Temperature']
        dataset['U1'] = df['Site Outdoor Air Temperature']
        dataset['U2'] = df['Cooling:Electricity']
        # dataset['U3'] = df['Whole Building Electricity']
        print(dataset.describe())
        
        # Possible include an on/off input

        trainLoader, devLoader, self.data = GetData(dataset, nsteps, bs, nx, nu)

        self.rcModel = RCNetwork(insize=nx+nu, outsize=nx)
        self.fxRK4 = integrators.RK4(self.rcModel, h=ts)
        self.model = System([Node(self.fxRK4, ['xn', 'U'], ['xn'])], nsteps=nsteps)

    def fitness(self, chromosome):
        self.rcModel.R, self.rcModel.C, self.rcModel.a = chromosome
        trajectories = self.model(self.data)
        pred_traj = trajectories['xn'][:,:-1,:].detach().numpy().reshape(-1, 1)
        true_traj = self.data['X'].detach().numpy().reshape(-1,1)
        return np.mean(np.abs(pred_traj - true_traj))


def Main():
    nx = 1
    nu = 2
    ts = 1
    nsteps = 60
    bs = 32

    df = pd.read_csv('1_out.csv', nrows=57600, usecols=['Site Outdoor Air Temperature', 'living space Air Temperature', 'Cooling:Electricity', 'Whole Building Electricity'])
    dataset = pd.DataFrame()
    dataset['X1'] = df['living space Air Temperature']
    dataset['U1'] = df['Site Outdoor Air Temperature']
    dataset['U2'] = df['Cooling:Electricity']
    # dataset['U3'] = df['Whole Building Electricity']
    print(dataset.describe())
    
    # Possible include an on/off input

    trainLoader, devLoader, testData = GetData(dataset, nsteps, bs, nx, nu)

    rcModel = RCNetwork(insize=nx+nu, outsize=nx)
    fxRK4 = integrators.RK4(rcModel, h=ts)
    dynamics_model = System([Node(fxRK4, ['xn', 'U'], ['xn'])], nsteps=nsteps)

    genAlg = ParamEstGenAlg(dynamics_model, testData)

    genAlg.run()

class ParamEstGenAlg():
    '''
    '''
    def __init__(self, model, data):
        '''
        Constructor
        '''
        self.generations = 200
        self.population = 100
        self.opts = {
            "mutationLoops": 2,
            "mutationDepth": 8,
            "mutationRate": 0.5, 
            "numberMax": 10,
            "numberDelims": 0.5,
            "elitismNum": 1,
            "crossoverNum": 10,
            "replicationNum": 1
        }
        self.model = model
        self.data = data

    def run(self):
        '''
        '''
        population = self.createInitPop(0, 100)

        bestPerformers = []
        allPopulations = []

        for i in range(0, self.generations):
            fitness = [self.loss(ind) for ind in population]
            bestIndividual = min(population, key=self.loss)
            bestFitness = self.loss(bestIndividual)
            bestPerformers.append((bestIndividual, bestFitness))
            allPopulations.append(population[:])

            population = self.selection(population, fitness)

            nextPopulation = []
            for j in range(0, len(population), 2):
                parent1 = population[j]
                parent2 = population[j+1]

                child1, child2 = self.crossover(parent1, parent2)

                nextPopulation.append(self.mutation(child1, 0, 1000))
                nextPopulation.append(self.mutation(child2, 0, 1000))

            nextPopulation[0] = bestIndividual
            population = nextPopulation

        finalPopulation = allPopulations[-1]
        finalFitness = [self.loss(ind) for ind in finalPopulation]

        fig, axs = plt.subplots(3, 1, figsize=(12, 18))
        axs[0].scatter(range(len(finalPopulation)), [ind[0] for ind in finalPopulation], color='blue', label='R')
        axs[0].scatter([finalPopulation.index(bestIndividual)], [bestIndividual[0]], color='cyan', s=100, label='Best Individual R')
        axs[0].text(finalPopulation.index(bestIndividual), bestIndividual[0]+1, "{:.3f}".format(bestIndividual[0]))
        axs[0].set_ylabel('R', color='blue')
        axs[0].legend(loc='upper left')
        
        axs[1].scatter(range(len(finalPopulation)), [ind[1] for ind in finalPopulation], color='green', label='C')
        axs[1].scatter([finalPopulation.index(bestIndividual)], [bestIndividual[1]], color='magenta', s=100, label='Best Individual C')
        axs[1].text(finalPopulation.index(bestIndividual), bestIndividual[1]+1, "{:.3f}".format(bestIndividual[1]))
        axs[1].set_ylabel('C', color='green')
        axs[1].legend(loc='upper left')
        
        axs[2].scatter(range(len(finalPopulation)), [ind[2] for ind in finalPopulation], color='red', label='a')
        axs[2].scatter([finalPopulation.index(bestIndividual)], [bestIndividual[2]], color='yellow', s=100, label='Best Individual a')
        axs[2].text(finalPopulation.index(bestIndividual), bestIndividual[2]+1, "{:.3f}".format(bestIndividual[2]))
        axs[2].set_ylabel('a', color='red')
        axs[2].set_xlabel('Individual Index')
        axs[2].legend(loc='upper left')
        
        axs[0].set_title(f'Final Generation ({self.generations}) Population Solutions')
        fig.savefig('Saved Figures/GA_lastGeneration.png')

        # Plot the values of a, b, and c over generations
        generations_list = range(1, len(bestPerformers) + 1)
        a_values = [ind[0][0] for ind in bestPerformers]
        b_values = [ind[0][1] for ind in bestPerformers]
        c_values = [ind[0][2] for ind in bestPerformers]
        fig, ax = plt.subplots()
        ax.plot(generations_list, a_values, label='R', color='blue')
        ax.plot(generations_list, b_values, label='C', color='green')
        ax.plot(generations_list, c_values, label='a', color='red')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Parameter Values')
        ax.set_title('Parameter Values Over Generations')
        ax.legend()
        fig.savefig('Saved Figures/GA_params.png')

        # Plot the fitness values over generations
        best_fitness_values = [fit[1] for fit in bestPerformers]
        min_fitness_values = [min([self.loss(ind) for ind in population]) for population in allPopulations]
        max_fitness_values = [max([self.loss(ind) for ind in population]) for population in allPopulations]
        largest = 0
        for i,fitness in enumerate(max_fitness_values):
            if fitness > 1e10:
                max_fitness_values[i] = largest
            else:
                if fitness > largest:
                    largest = fitness

        fig, ax = plt.subplots()
        ax.plot(generations_list, best_fitness_values, label='Best Fitness', color='black')
        # ax.fill_between(generations_list, min_fitness_values, max_fitness_values, color='gray', alpha=0.5, label='Fitness Range')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness')
        ax.set_title('Fitness Over Generations')
        ax.legend()
        fig.savefig('Saved Figures/GA_fitness.png')

        trajectories = self.model(self.data)
        pred_traj = trajectories['xn'][:,:-1,:].detach().numpy().reshape(-1, 1)

        true_traj = self.data['X'].detach().numpy().reshape(-1,1)
        input_traj = self.data['U'].detach().numpy().reshape(-1,2)

        fig, ax = plt.subplots(2, figsize=(10,5))
        u = torch.from_numpy(input_traj).float()
        sol = torch.zeros((true_traj.shape[0],1))
        ic = torch.tensor(true_traj[0,:])
        for j in range(sol.shape[0]-1):
            if j==0:
                sol[[0],:] = ic.float()
                sol[[j+1],:] = self.model.nodes[0].callable(sol[[j],:],u[[j],:])
            else:
                sol[[j+1],:] = self.model.nodes[0].callable(sol[[j],:],u[[j],:])

            
        ax[0].plot(sol.detach().numpy(), label='model', color='black')
        ax[0].plot(pred_traj, label='pred')
        ax[0].plot(true_traj, label='data', color='red')
        ax[0].legend()
        ax[1].plot(input_traj, label='input')
        for x in ax:
            x.set_xlim([0,1000])
        plt.legend()
        fig.savefig('./Saved Figures/GA_tempPred', dpi=fig.dpi)

        print(f'Test mae: {np.mean(np.abs(pred_traj - true_traj))}')

    def loss(self, params):
        '''
        '''
        rc = self.model.nodes[0].callable.block
        rc.R, rc.C, rc.a = params
        trajectories = self.model(self.data)
        pred_traj = trajectories['xn'][:,:-1,:].detach().numpy().reshape(-1, 1)
        true_traj = self.data['X'].detach().numpy().reshape(-1,1)
        return np.mean(np.abs(pred_traj - true_traj))

    def createInitPop(self, lowerBound, upperBound):
        '''
        '''
        population = []
        for _ in range(self.population):
            individ = (random.uniform(lowerBound, upperBound),
                      random.uniform(lowerBound, upperBound),
                      random.uniform(lowerBound, upperBound))
            population.append(individ)
        return population

    def selection(self, population, fitness, tournamentSize=3):
        '''
        '''
        selected = []
        for _ in range(len(population)):
            tournament = random.sample(list(zip(population, fitness)), tournamentSize)
            winner = min(tournament, key=lambda x: x[1])[0]
            selected.append(winner)
        return selected
    
    def crossover(self, parent1, parent2):
        '''
        '''
        alpha = random.random()
        child1 = tuple(alpha * p1 + (1 - alpha) * p2 for p1,p2 in zip(parent1, parent2))
        child2 = tuple(alpha * p1 + (1 - alpha) * p2 for p1,p2 in zip(parent1, parent2))
        return child1, child2
    
    def mutation(self, individual, upperBound, lowerBound):
        '''
        '''
        individual = list(individual)
        for i in range(len(individual)):
            if random.random() < self.opts['mutationRate']:
                mutationAmount = random.uniform(-5,5)
                individual[i] += mutationAmount
                if individual[i] <= 1e-10:
                    individual[i] += 1e-4
                individual[i] = max(min(individual[i], upperBound), lowerBound)
        return tuple(individual)

class RCNetwork(ode.ODESystem):
    def __init__(self, insize=2, outsize=2):
        super().__init__(insize=insize, outsize=outsize)
        self.R = 10.0
        self.C = 1.0
        self.a = 1.0
        # self.p = nn.Parameter(torch.tensor([5.0]), requires_grad=True)
        # self.f = nn.Parameter(torch.tensor([5.0]), requires_grad=True)

    def ode_equations(self, x, u):
        x1 = x[:, [0]]
        # x2 = x[:, [-1]]
        u1 = u[:,[0]]
        u2 = u[:, [1]]
        # u3 = u[:, [-1]]
        dx1 = -1/(self.C*self.R)*x1 + 1/(self.C*self.R)*u1 - self.a*u2
        return torch.cat([dx1], dim=1)
    
def GetData(data_df, nsteps, bs, nx, nu):
    n = len(data_df)
    mean = data_df.mean()
    std = data_df.std()
    # data_df = (data_df - mean) / std
    data_df.fillna(0, inplace=True)
    
    train_df = data_df[:int(np.round(n*0.7))]
    dev_df = data_df[int(np.round(n*0.7)):int(np.round(n*0.9))]
    test_df = data_df[int(np.round(n*0.9)):]

    nbatch = len(test_df)//nsteps

    train = {}
    train['X'] = np.column_stack([train_df[f'X{i}'] for i in range(1,nx+1)])
    train['U'] = np.column_stack([train_df[f'U{i}'] for i in range(1,nu+1)])

    dev = {}
    dev['X'] = np.column_stack([dev_df[f'X{i}'] for i in range(1,nx+1)])
    dev['U'] = np.column_stack([dev_df[f'U{i}'] for i in range(1,nu+1)])

    test = {}
    test['X'] = np.column_stack([test_df[f'X{i}'] for i in range(1,nx+1)])
    test['U'] = np.column_stack([test_df[f'U{i}'] for i in range(1,nu+1)])


    trainX = train['X'][:n].reshape(nbatch*7, nsteps, nx)
    trainX = torch.tensor(trainX, dtype=torch.float32)
    trainU = train['U'][:n].reshape(nbatch*7, nsteps, nu)
    trainU = torch.tensor(trainU, dtype=torch.float32)
    trainData = DictDataset({'X': trainX, 'xn': trainX[:,0:1,:],
                             'U': trainU}, name='train')
    trainLoader = DataLoader(trainData, batch_size=bs,
                             collate_fn=trainData.collate_fn, shuffle=True)
    
    devX = dev['X'][:n].reshape(nbatch*2, nsteps, nx)
    devX = torch.tensor(devX, dtype=torch.float32)
    devU = dev['U'][:n].reshape(nbatch*2, nsteps, nu)
    devU = torch.tensor(devU, dtype=torch.float32)
    devData = DictDataset({'X': devX, 'xn': devX[:,0:1,:],
                             'U': devU}, name='dev')
    devLoader = DataLoader(devData, batch_size=bs,
                             collate_fn=devData.collate_fn, shuffle=True)
    
    testX = test['X'][:n].reshape(nbatch, nsteps, nx)
    testX = torch.tensor(testX, dtype=torch.float32)
    testU = test['U'][:n].reshape(nbatch, nsteps, nu)
    testU = torch.tensor(testU, dtype=torch.float32)
    testData = {'X': testX, 'xn': testX[:,0:1,:], 'U': testU}

    return trainLoader, devLoader, testData

if __name__ == '__main__':
    Main()
    # MainPrefab()