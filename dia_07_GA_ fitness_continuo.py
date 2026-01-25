"""
📘 DIA 07 — Função de Fitness em Problemas Contínuos
-----------------------------------------------------

GA básico para otimização de funções reais usando:

- Representação contínua (float)
- Crossover aritmético
- Mutação gaussiana
- Seleção por torneio

Problema: maximizar f(x) = x * sin(10x) + 1
em x ∈ [-1, 2]
"""

import random
import math
import matplotlib.pyplot as plt

random.seed(42)

# -----------------------------------------------------------
# 1️⃣ Parâmetros gerais
# -----------------------------------------------------------
POP_SIZE = 50
NUM_GENERATIONS = 60
TOURNAMENT_K = 3
MUTATION_RATE = 0.2
MUTATION_STD = 0.1  # desvio-padrão da mutação gaussiana
LOWER_BOUND, UPPER_BOUND = -1, 2


# -----------------------------------------------------------
# 2️⃣ Função de fitness (problema contínuo)
# -----------------------------------------------------------
def fitness(x):
    return x * math.sin(10 * x) + 1


# -----------------------------------------------------------
# 3️⃣ Inicialização (valores reais aleatórios)
# -----------------------------------------------------------
def initialize_population():
    return [random.uniform(LOWER_BOUND, UPPER_BOUND) for _ in range(POP_SIZE)]


# -----------------------------------------------------------
# 4️⃣ Seleção por torneio
# -----------------------------------------------------------
def tournament_selection(population):
    competitors = random.sample(population, TOURNAMENT_K)
    return max(competitors, key=fitness)


# -----------------------------------------------------------
# 5️⃣ Crossover aritmético
# -----------------------------------------------------------
def crossover(p1, p2):
    alpha = random.random()  # peso entre 0 e 1
    child = alpha * p1 + (1 - alpha) * p2
    return child


# -----------------------------------------------------------
# 6️⃣ Mutação gaussiana
# -----------------------------------------------------------
def mutate(x):
    if random.random() < MUTATION_RATE:
        x = x + random.gauss(0, MUTATION_STD)
    return max(LOWER_BOUND, min(UPPER_BOUND, x))  # clamping


# -----------------------------------------------------------
# 7️⃣ Loop principal do GA
# -----------------------------------------------------------
def run_ga():
    population = initialize_population()
    best_history = []

    for gen in range(NUM_GENERATIONS):

        new_population = []

        # elitismo
        best_individual = max(population, key=fitness)
        new_population.append(best_individual)

        while len(new_population) < POP_SIZE:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)

            child = crossover(parent1, parent2)
            child = mutate(child)

            new_population.append(child)

        population = new_population
        best_f = fitness(max(population, key=fitness))
        best_history.append(best_f)

        if gen % 10 == 0:
            print(f"Geração {gen} | Melhor fitness = {best_f:.4f}")

    return best_history


# -----------------------------------------------------------
# 8️⃣ Execução e visualização
# -----------------------------------------------------------
if __name__ == "__main__":
    history = run_ga()

    plt.plot(history)
    plt.title("Evolução do GA em domínio contínuo")
    plt.xlabel("Geração")
    plt.ylabel("Melhor fitness")
    plt.grid(True)
    plt.show()

