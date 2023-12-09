import random
import numpy as np
import copy



class Wolf:
    def __init__(self, fitness, dim, minx, maxx, seed) -> None:
        self.rnd = random.Random(seed)
        self.position = [0.0 for i in range(dim)]
        for i in range(dim):
            self.position[i] = ((maxx - minx) * self.rnd.random() + minx)
        self.fitness = fitness(self.position) # curr fitness

    def get_position(self):
        return self.position
        
def pos(a:Wolf):
    return a.get_position()


def GWO(fitness, max_iter, n, dim, minx, maxx):
    rnd = random.Random(0)

    # create n random wolves 
    population = [ Wolf(fitness, dim, minx, maxx, i) for i in range(n)]

    # On the basis of fitness values of wolves 
    # sort the population in asc order
    population = sorted(population, key = lambda temp: temp.fitness)

    # best 3 solutions will be called as 
    # alpha, beta and gaama
    alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[: 3])

    # main loop of gwo
    Iter = 0
    while Iter < max_iter:
        # after every 10 iterations 
        # print iteration number and best fitness value so far
        if Iter % 10 == 0 and Iter > 1:
            print("Iter = " + str(Iter) + " best fitness = %.3f" % alpha_wolf.fitness)

        # linearly decreased from 2 to 0
        a = 2*(1 - Iter/max_iter)

        # updating each population member with the help of best three members 
        for i in range(n):
            A1, A2, A3 = a * (2 * rnd.random() - 1), a * (
            2 * rnd.random() - 1), a * (2 * rnd.random() - 1)
            C1, C2, C3 = 2 * rnd.random(), 2*rnd.random(), 2*rnd.random()

            X1 = [0.0 for i in range(dim)]
            X2 = [0.0 for i in range(dim)]
            X3 = [0.0 for i in range(dim)]
            Xnew = [0.0 for i in range(dim)]
            for j in range(dim):
                X1[j] = alpha_wolf.position[j] - A1 * abs(
                C1 * alpha_wolf.position[j] - population[i].position[j])
                X2[j] = beta_wolf.position[j] - A2 * abs(
                C2 * beta_wolf.position[j] - population[i].position[j])
                X3[j] = gamma_wolf.position[j] - A3 * abs(
                C3 * gamma_wolf.position[j] - population[i].position[j])
                Xnew[j]+= X1[j] + X2[j] + X3[j]
            
            for j in range(dim):
                Xnew[j]/=3.0
            
            # fitness calculation of new solution
            fnew = fitness(Xnew)

            # greedy selection
            if fnew < population[i].fitness:
                population[i].position = Xnew
                population[i].fitness = fnew
                
        # On the basis of fitness values of wolves 
        # sort the population in asc order
        population = sorted(population, key = lambda temp: temp.fitness)

        # best 3 solutions will be called as 
        # alpha, beta and gaama
        alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[: 3])
        features = [i>=1 for i in alpha_wolf.position]
        print(features)
        Iter+= 1
    # end-while
    # returning the best solution
    return pos(alpha_wolf)