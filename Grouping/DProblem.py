import geatpy as ea
from util import help_Proposal
import numpy as np
from bench import benchmark


class MyProblem(ea.Problem):
    def __init__(self, Dim, gene_len, modify_fitness, model, scaler_X, random_Pop, intercept, dif):
        name = 'MyProblem'
        M = 1
        maxormins = [1]
        self.random_Pop = random_Pop
        self.intercept = intercept
        self.dif = dif
        self.modify_fitness = modify_fitness
        self.Gene_len = gene_len
        self.model = model
        self.scaler_X = scaler_X
        self.Dim = Dim
        self.varTypes = [1] * Dim
        self.lb = [0] * Dim
        self.ub = [2 ** gene_len-1] * Dim
        self.lbin = [1] * Dim
        self.ubin = [1] * Dim
        ea.Problem.__init__(self, name, M, maxormins, Dim, self.varTypes, self.lb, self.ub, self.lbin, self.ubin)

    def aimFunc(self, pop):
        Vars = pop.Phen
        Objs = []
        for var in Vars:
            # Decompose the problem depending on
            Empty_groups = help_Proposal.Phen_Groups(var, help_Proposal.empty_groups(2 ** self.Gene_len))

            groups_fitness = benchmark.groups_fitness(Empty_groups, self.random_Pop, self.model,
                                                                 self.scaler_X, self.intercept, self.dif)
            Objs.append([benchmark.object_function(self.modify_fitness, groups_fitness)])
        pop.ObjV = np.array(Objs, dtype='double').reshape(-1, 1)



