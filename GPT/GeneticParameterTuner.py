import os
import multiprocessing
import numpy as np
from time import time

from GPT.multiprocessing_helpers import DillProcess


class GPT:
    default_parameter_values = {'int': 0, 'float': 0.0, 'bool': False}

    def __init__(self, fitness_function, parameter_types: dict, parameter_mutations: dict = None,
                 projections: dict = None,
                 initial_set: dict = None,
                 mutation_probability: float = 0.1, crossover_probability: float = 0.1,
                 num_generations: int = 100, num_agents: int = 100, num_workers: int = -1, print_progress: bool = False):
        """
        Used to initialize the Genetic Parameter Tuner
        :param fitness_function: function to evaluate the fitness of the agents. It should accept a single parameter,
        a dictionary of the form {parameter_name: value} for each parameter.
        :param parameter_types: dictionary of form: {parameter_name: type}, where type is one of 'int', 'float', 'bool'
        :param parameter_mutations: dictionary of functions to use when mutating parameters. If None, defaults to mutation with
        a normal distribution with mean of previous value, and standard deviation of 25% of the previous value
        :param projections: if the parameters need to be in a certain range, this dictionary should contain functions that
        will be applied to the parameters before evaluation. If None, defaults to identity function
        :param initial_set: dictionary of values for the parameters to start with. Each starting agent will be initialized with
        mutation of these values. If None, defaults to random values within the range of the parameter type
        :param mutation_rate: probability of mutation for each parameter.
        :param crossover_rate: probability of crossover between two chosen agents.
        :param num_generations: number of epochs to run the genetic algorithm
        :param num_agents: number of agents to train for each epoch
        :param num_workers: number of workers to use for parallel processing. If -1, defaults to number of cores
        """
        self.fitness_function = fitness_function
        self.parameter_types = parameter_types
        # type checking
        for key in self.parameter_types.keys():
            if self.parameter_types[key] not in self.default_parameter_values.keys():
                raise ValueError(f'Parameter type {self.parameter_types[key]} is not supported')

        self.mutation_probability = mutation_probability
        self.crossover_probability = crossover_probability

        # setting the mutation operators
        if parameter_mutations is None:
            self.parameter_mutations = {}
        else:
            self.parameter_mutations = parameter_mutations
        for key in self.parameter_types.keys():
            if key not in self.parameter_mutations:
                self.parameter_mutations[key] = self._default_mutation

        # setting the projection operators
        if projections is None:
            self.projections = {}
        else:
            self.projections = projections
        for key in self.parameter_types.keys():
            if key not in self.projections:
                self.projections[key] = self._identity_projection

        # setting the initial set
        if initial_set is None:
            self.initial_set = {}
        else:
            self.initial_set = initial_set
        for key in self.parameter_types.keys():
            if key not in self.initial_set:
                self.initial_set[key] = self.default_parameter_values[self.parameter_types[key]]

        self.num_generations = num_generations
        self.num_agents = num_agents
        self.num_workers = num_workers if num_workers != -1 else os.cpu_count()

        # initializing the first set of agents
        self.agents = [self._mutate(self.initial_set, mutation_probability=1) for _ in range(self.num_agents)]

        if print_progress:
            print('Initialization done, evaluating initial agents...')
        start_time = time()
        if self.num_workers > 1:
            max_workers = min(self.num_workers, self.num_agents)
            index_splits = np.array_split(range(self.num_agents), max_workers)

            with multiprocessing.Manager() as manager:
                agents = manager.list(self.agents)
                fitnesses = manager.list([0 for _ in range(self.num_agents)])

                workers = []
                for j in range(max_workers):
                    worker = DillProcess(target=self._evaluate_parallel, args=(agents, fitnesses, index_splits[j]))
                    workers.append(worker)
                    worker.start()

                for worker in workers:
                    worker.join()

                self.agents = list(agents)
                self.fitnesses = list(fitnesses)

        end_time = time()
        if print_progress:
            print(f'Initialization done, took {end_time - start_time:.2f} seconds')

        if print_progress:
            print('Starting the genetic algorithm...')
        for i in range(self.num_generations):
            start_time = time()
            self.agents = self._reproduce(self.agents, self.fitnesses)
            self.agents = [self._mutate(agent) for agent in self.agents]

            if self.num_workers > 1:
                max_workers = min(self.num_workers, self.num_agents)
                index_splits = np.array_split(range(self.num_agents), max_workers)

                with multiprocessing.Manager() as manager:
                    agents = manager.list(self.agents)
                    fitnesses = manager.list(self.fitnesses)

                    workers = []
                    for j in range(max_workers):
                        worker = DillProcess(target=self._evaluate_parallel, args=(agents, fitnesses, index_splits[j]))
                        workers.append(worker)
                        worker.start()

                    for worker in workers:
                        worker.join()

                    self.agents = list(agents)
                    self.fitnesses = list(fitnesses)

            else:
                self.fitnesses = [self._evaluate(agent) for agent in self.agents]

            end_time = time()
            if print_progress:
                print(f'Generation {i + 1}/{self.num_generations} done, best fitness: {np.max(self.fitnesses)}\n'
                      f'It took {end_time - start_time:.2f} seconds\n')

        self.best_agent = self.agents[np.argmax(self.fitnesses)]
        self.best_fitness = np.max(self.fitnesses)

    @staticmethod
    def _identity_projection(value):
        return value

    @staticmethod
    def _default_mutation(parameter_type, value):
        if parameter_type == 'int':
            return int(np.random.normal(value, 0.25 * value))
        elif parameter_type == 'float':
            return np.random.normal(value, 0.25 * value)
        else:
            # the only way to mutate a boolean is to flip it
            return not value

    def _mutate(self, agent, mutation_probability=None):
        if mutation_probability is None:
            mutation_probability = self.mutation_probability
        new_agent = agent.copy()
        for key in agent.keys():
            if np.random.rand() < mutation_probability:
                new_agent[key] = self.parameter_mutations[key](self.parameter_types[key], agent[key])
        return new_agent

    def _crossover(self, agent1, agent2):
        new_agent1 = {}
        new_agent2 = {}

        if np.random.rand() < self.crossover_probability:
            # selecting the random crossover point
            crossover_point = np.random.randint(0, len(agent1))

            for i, key in enumerate(agent1.keys()):
                if i < crossover_point:
                    new_agent1[key] = agent1[key]
                    new_agent2[key] = agent2[key]
                else:
                    new_agent1[key] = agent2[key]
                    new_agent2[key] = agent1[key]

        else:
            new_agent1 = agent1
            new_agent2 = agent2

        return new_agent1, new_agent2

    def _select(self, agents, fitnesses):
        # normalizing the fitnesses to be in the range [0, 1], to represent probabilities
        # copy the fitnesses to avoid changing the original list
        fitnesses = np.array(fitnesses)
        fitnesses = fitnesses / np.sum(fitnesses)

        # randomly selecting two agents based on their fitness
        best_indices = np.random.choice(range(len(agents)), size=2, p=fitnesses)

        return agents[best_indices[0]], agents[best_indices[1]]

    def _evaluate_parallel(self, agents, fitnesses, indices):
        for i in indices:
            fitnesses[i] = self._evaluate(agents[i])

    def _evaluate(self, agent):
        projected_agent = {key: self.projections[key](value) for key, value in agent.items()}
        return self.fitness_function(projected_agent)

    def _reproduce(self, agents, fitnesses):
        new_agents = []
        while len(new_agents) < self.num_agents:
            agent1, agent2 = self._select(agents, fitnesses)
            new_agent1, new_agent2 = self._crossover(agent1, agent2)
            new_agents.append(new_agent1)
            new_agents.append(new_agent2)
        if len(new_agents) > self.num_agents:
            # if the number of new agents is odd, we need to remove one
            new_agents = new_agents[:self.num_agents]
        return new_agents

    def get_best_agent(self):
        return self.best_agent