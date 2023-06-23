import csv
import random
import numpy as np
from typing import List
from shapely import affinity
from joblib import Parallel, delayed
from random import randint, uniform, choice
from shapely.geometry import Point, Polygon
from sklearn.model_selection import ParameterGrid

square_size = 1000

class Container:
    def __init__(self, square_size):
        self.batteries = []
        self.square_size = square_size * 1000
        self.container_polygon = Polygon([(0, 0), (0, self.square_size), (self.square_size, self.square_size), (self.square_size, 0)])

    def add_battery(self, battery):
        if not self.check_overlap(battery) and self.is_inside(battery):
            self.batteries.append(battery)
            return True
        return False

    def remove_battery(self, battery):
        self.batteries.remove(battery)

    def check_overlap(self, battery):
        for other_battery in self.batteries:
            if other_battery is not battery and other_battery.shape.intersects(battery.shape):
                return True
        return False

    def is_inside(self, battery):
        return self.container_polygon.contains(battery.shape)

    def largest_empty_space(self, a, b):
        positions = [(x, y) for x in range(0, self.square_size - a * 1000, 1000) for y in range(0, self.square_size - b * 1000, 1000)]
        random.shuffle(positions)
        for theta in range(0, 1000, 1):
            for x, y in positions:
                candidate_battery = Battery(x, y, a * 1000, b * 1000, theta, self, True)
                if self.is_inside(candidate_battery) and not candidate_battery.is_overlapping():
                    return x, y, theta
        print("Warning: no large enough empty space found")
        return None

    def validate(self):
        for battery in self.batteries:
            if self.check_overlap(battery):
                print(f"Validation failed: Battery {battery} is overlapping with another battery.")
                return False
            if not self.is_inside(battery):
                print(f"Validation failed: Battery {battery} is outside the container.")
                return False
        print("Validation passed: All batteries are inside the container and there are no overlaps.")
        return True

container = Container(square_size)

class Battery:
    def __init__(self, x, y, a, b, rotation, container, add_to_container=True):
        self.x = x/1000
        self.y = y/1000
        self.a = a
        self.b = b
        self.rotation = rotation
        self.shape = self.ellipse_as_polygon()
        self.container = container
        if add_to_container:
            self.container.add_battery(self)

    def ellipse_as_polygon(self, resolution=100):
        t = np.linspace(0, 2*np.pi, resolution)
        st = self.a/2.*np.sin(t)
        ct = self.b/2.*np.cos(t)
        coords = np.column_stack((self.x+st, self.y+ct))
        ellipse = Polygon(coords)
        return affinity.rotate(ellipse, np.degrees(self.rotation), origin=(self.x, self.y))

    def is_overlapping(self):
        return self.container.check_overlap(self)

    def mutate(self, step_size=1, angle_step=1):
        for _ in range(self.container.mutation_attempts):
            old_values = (self.x, self.y, self.rotation)
            self.x = self.ensure_within_bounds(self.x + step_size * randint(-1, 1)/1000, self.container.square_size - self.a/1000)
            self.y = self.ensure_within_bounds(self.y + step_size * randint(-1, 1)/1000, self.container.square_size - self.b/1000)
            self.rotation = (self.rotation + angle_step * uniform(-1, 1)/1000) % np.pi
            self.shape = self.ellipse_as_polygon()
            self.container.remove_battery(self)
            if not self.container.is_inside(self) or self.is_overlapping():
                print(f"Mutation failed at ({self.x}, {self.y}, {self.rotation}): overlap or outside container")
                self.x, self.y, self.rotation = old_values
            else:
                print(f"Mutation success at ({self.x}, {self.y}, {self.rotation})")
                self.container.add_battery(self)
                return self
        print(f"Mutation failed after {self.container.mutation_attempts} attempts")
        return None

    def ensure_within_bounds(self, value, max_value):
        return max(0, min(value, max_value))

class GeneticAlgorithm:
    def __init__(self, mutation_rate, crossover_operator, selection_operator, mutation_attempts=10):
        self.mutation_rate = mutation_rate
        self.crossover_operator = crossover_operator
        self.selection_operator = selection_operator
        self.mutation_attempts = mutation_attempts

    def adjust_mutation_rate(self, scores, previous_average):
        current_average = sum(scores) / len(scores)
        improvement = previous_average - current_average
        if previous_average != 0:
            mutation_rate = min(0.1, abs(improvement) / previous_average)
        else:
            mutation_rate = self.mutation_rate
        return mutation_rate, current_average

    def check_stagnation(self, scores, previous_average):
        current_average = sum(scores) / len(scores)
        if current_average == previous_average:
            return True
        return False

def uniform_crossover(parent1, parent2):
    child1, child2 = [], []
    for gene1, gene2 in zip(parent1, parent2):
        if random.random() < 0.5:
            child1.append(Battery(gene1.x, gene1.y, gene1.a, gene1.b, gene1.rotation, add_to_container=False))
            child2.append(Battery(gene2.x, gene2.y, gene2.a, gene2.b, gene2.rotation, add_to_container=False))
        else:
            child1.append(Battery(gene2.x, gene2.y, gene2.a, gene2.b, gene2.rotation, add_to_container=False))
            child2.append(Battery(gene1.x, gene1.y, gene1.a, gene1.b, gene1.rotation, add_to_container=False))
    return child1, child2

def rank_selection(population, scores):
    sorted_indices = np.argsort(scores)
    sorted_population = [population[i] for i in sorted_indices]
    cumulative_sum = np.cumsum(np.arange(len(population), 0, -1))
    r = np.random.uniform(0, cumulative_sum[-1])
    individual = sorted_population[np.searchsorted(cumulative_sum, r)]
    return individual

def generate_initial_population(population_size, battery_types, counts):
    population = []
    for _ in range(population_size):
        battery_types_temp = battery_types.copy()
        counts_temp = counts.copy()
        batteries = []
        for _ in range(sum(counts)):
            battery = generate_random_battery(battery_types_temp, counts_temp, method='LES')
            if battery is not None:
                batteries.append(battery)
        population.append(batteries)
    return population

def generate_random_battery(battery_types: List[tuple], counts: List[int], method='random'):
    if not battery_types:
        return None
    index = randint(0, len(battery_types) - 1)
    a, b = battery_types[index]
    if a > square_size or b > square_size:
        return None
    counts[index] -= 1
    if counts[index] <= 0:
        battery_types.pop(index)
        counts.pop(index)
    coords = None
    if method == 'LES':
        coords = container.largest_empty_space(a, b)
    if coords is None:
        x = randint(0, square_size - a)
        y = randint(0, square_size - b)
    else:
        x, y = coords
    rotation = uniform(0, np.pi)
    return Battery(x, y, a, b, rotation)

def evaluate_solution(batteries):
    overlapping_area = 0
    coverage_area = 0
    total_distance = 0
    print(f"Initial values: overlapping area: {overlapping_area}, coverage area: {coverage_area}, total distance: {total_distance}")
    for i, battery in enumerate(batteries):
        print(f"Checking battery {i}")
        if not container.is_inside(battery):
            print(f"Evaluation failed: Battery {i} is outside the container.")
            return float('inf')
        for j, other_battery in enumerate(batteries):
            if j > i and other_battery.shape.intersects(battery.shape):
                overlapping_area += other_battery.shape.intersection(battery.shape).area
                print(f"Overlapping area increased to {overlapping_area}")
        coverage_area += battery.shape.area
        print(f"Coverage area increased to {coverage_area}")
        if i > 0:
            total_distance += battery.distance_to(batteries[i-1])
            print(f"Total distance increased to {total_distance}")
    print(f"Final values: overlapping area: {overlapping_area}, coverage area: {coverage_area}, total distance: {total_distance}")
    result = overlapping_area - coverage_area - total_distance
    print(f"Evaluation result: {result}")
    return result

def crossover(parent1, parent2, max_attempts=100):
    if len(parent1) <= 1 or len(parent2) <= 1:
        return parent1, parent2
    for _ in range(max_attempts):
        crossover_point = random.randint(1, len(parent1)-1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        print(f"Attempting crossover at point {crossover_point}")
        if evaluate_solution(child1) and evaluate_solution(child2):
            print(f"Children after crossover: {child1}, {child2}")
            return child1, child2
    print(f"Crossover failed, parents returned")
    return parent1, parent2

def tournament_selection(population, scores, k=3):
    selected_indices = random.sample(range(len(population)), k)
    selected = [population[i] for i in selected_indices]
    best_individual_index = min(selected_indices, key=lambda i: scores[i])
    return population[best_individual_index]

def save_best_solution(batteries):
    with open("best_solution.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "a", "b", "rotation"])
        for battery in batteries:
            writer.writerow([int(battery.x*1000), int(battery.y*1000), int(battery.a*1000), int(battery.b*1000), int(battery.rotation)])

def validate_solution(batteries):
    for i, battery in enumerate(batteries):
        if not container.is_inside(battery):
            print(f"Validation failed: Battery {i} is outside the container.")
            return False
        for j, other_battery in enumerate(batteries):
            if j > i and other_battery.shape.intersects(battery.shape):
                print(f"Validation failed: Batteries {i} and {j} are overlapping.")
                return False
    print("Validation passed: All batteries are inside the container and there are no overlaps.")
    return True

def main():
    battery_types = [(40000, 20000), (20000, 10000)]
    counts = [39, 161]
    population_size = 100
    generation_limit = 1000
    scores = []
    parameters = {
        'mutation_rate': [0.01, 0.05, 0.1],
        'crossover_operator': [crossover, uniform_crossover],
        'selection_operator': [tournament_selection, rank_selection]
    }
    parameter_grid = list(ParameterGrid(parameters))
    best_score = float('inf')
    best_parameters = None
    best_individual = None
    population = generate_initial_population(population_size, battery_types, counts)
    for parameter_set in parameter_grid:
        genetic_algorithm = GeneticAlgorithm(**parameter_set)
        previous_average = 0
        for generation in range(generation_limit):
            scores = Parallel(n_jobs=-1)(delayed(evaluate_solution)(individual) for individual in population)
            mutation_rate, previous_average = genetic_algorithm.adjust_mutation_rate(scores, previous_average)
            parent1 = genetic_algorithm.selection_operator(population, scores)
            parent2 = genetic_algorithm.selection_operator(population, scores)
            child1, child2 = genetic_algorithm.crossover_operator(parent1, parent2)
            if random.random() < mutation_rate:
                child1 = [battery.mutate(step_size=mutation_rate, angle_step=mutation_rate) for battery in child1 if battery]
                child2 = [battery.mutate(step_size=mutation_rate, angle_step=mutation_rate) for battery in child2 if battery]
            population[random.randint(0, len(population) - 1)] = child1
            population[random.randint(0, len(population) - 1)] = child2
            best_individual = min(population, key=evaluate_solution)
            if best_individual not in [parent1, parent2, child1, child2]:
                population[random.randint(0, len(population) - 1)] = best_individual
            best_individual_score = min(scores)
            if best_individual_score < best_score:
                best_score = best_individual_score
                best_parameters = parameter_set
                best_individual = min(population, key=evaluate_solution)
            print(f"Generation {generation + 1} of {generation_limit} for current parameter set")
            print(f"Best score in this generation: {best_individual_score}")
            if validate_solution(best_individual):
                print("Best solution is valid.")
            else:
                print("Best solution is invalid.")
                print("\n")
    if best_individual:
        population[0] = best_individual
    print("Best parameters:", best_parameters)
    print("Best score:", best_score)
    if container.validate():
        print("All batteries are inside the container and there are no overlaps.")
    else:
        print("Some batteries are overlapping or outside the container.")

if __name__ == "__main__":
    main()
