import os

import numpy as np


class GPT:
    default_parameter_values = {'int': 0, 'float': 0.0, 'bool': False}

    def __init__(self, fitness_function, parameter_types: dict, parameter_mutations: dict = None,
                 projections: dict = None,
                 initial_set: dict = None,
                 num_epochs: int = 100, num_agents: int = 100, num_workers: int = -1):
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
        :param num_epochs: number of epochs to run the genetic algorithm
        :param num_agents: number of agents to train for each epoch
        :param num_workers: number of workers to use for parallel processing. If -1, defaults to number of cores
        """
        self.fitness_function = fitness_function
        self.parameter_types = parameter_types
        # type checking
        for key in self.parameter_types.keys():
            if self.parameter_types[key] not in self.default_parameter_values.keys():
                raise ValueError(f'Parameter type {self.parameter_types[key]} is not supported')

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

        self.num_epochs = num_epochs
        self.num_agents = num_agents
        self.num_workers = num_workers if num_workers != -1 else os.cpu_count()

    @staticmethod
    def _identity_projection(value):
        return value

    @staticmethod
    def _default_mutation(parameter_type, value):
        if parameter_type == 'int':
            return np.random.randint(value - 0.25 * value, value + 0.25 * value)
        elif parameter_type == 'float':
            return np.random.normal(value, 0.25 * value)
        else:
            # the only way to mutate a boolean is to flip it
            return not value
