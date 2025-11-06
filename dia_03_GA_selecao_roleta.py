"""
📘 DIA 3 — Seleção por Roleta (Roulette Wheel Selection)
--------------------------------------------------------

Objetivo:
---------
Comparar o impacto de diferentes métodos de seleção no desempenho de um
Algoritmo Genético (GA) aplicado à maximização da função:

    f(x) = x * sin(10πx) + 1.0

no intervalo [-1, 2].

Serão comparados:
- Seleção por Torneio (como referência)
- Seleção por Roleta (proporcional ao fitness)

Ao final, será plotado o desempenho (melhor fitness) de cada método.
"""

import random
import math
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# 1️⃣ Parâmetros do GA
# -----------------------------------------------------------
POP_SIZE = 40
NUM_GENERATIONS = 60
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.9
X_MIN, X_MAX = -1, 2
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# -----------------------------------------------------------
# 2️⃣ Função objetivo
# -----------------------------------------------------------
def objective_function(x: float) -> float:
    """Função multimodal usada para testar o desempenho do GA."""
    return x * math.sin(10 * math.pi * x) + 1.0


# -----------------------------------------------------------
# 3️⃣ Inicialização da população
# -----------------------------------------------------------
def initialize_population():
    """Gera uma população inicial aleatória."""
    return [random.uniform(X_MIN, X_MAX) for _ in range(POP_SIZE)]


# -----------------------------------------------------------
# 4️⃣ Seleção por Torneio
# -----------------------------------------------------------
def tournament_selection(population, k=2):
    """Seleciona o melhor de k indivíduos aleatórios."""
    candidates = random.sample(population, k)
    return max(candidates, key=objective_function)


# -----------------------------------------------------------
# 5️⃣ Seleção por Roleta (Roulette Wheel)
# -----------------------------------------------------------
def roulette_selection(population):
    """
    Seleciona um indivíduo proporcional ao seu fitness.
    Implementa o conceito de 'roleta viciada' usado em GAs clássicos.
    """
    # Avalia fitness e soma total
    fitness_values = [objective_function(ind) for ind in population]
    total_fitness = sum(fitness_values)

    # Normaliza fitness (probabilidade de seleção)
    probs = [f / total_fitness for f in fitness_values]

    # Seleciona aleatoriamente conforme probabilidade acumulada
    r = random.random()
    cumulative = 0
    for ind, p in zip(population, probs):
        cumulative += p
        if r <= cumulative:
            return ind
    return population[-1]  # segurança


# -----------------------------------------------------------
# 6️⃣ Cruzamento e Mutação
# -----------------------------------------------------------
def blend_crossover(parent1, parent2):
    """Blend crossover (BLX-α)."""
    if random.random() > CROSSOVER_RATE:
        return parent1
    alpha = 0.5
    diff = abs(parent1 - parent2)
    low = min(parent1, parent2) - alpha * diff
    high = max(parent1, parent2) + alpha * diff
    child = random.uniform(low, high)
    return max(min(child, X_MAX), X_MIN)


def mutate(x):
    """Mutação gaussiana."""
    if random.random() < MUTATION_RATE:
        x += random.gauss(0, 0.1)
    return max(min(x, X_MAX), X_MIN)


# -----------------------------------------------------------
# 7️⃣ Algoritmo Genético com método de seleção escolhido
# -----------------------------------------------------------
def run_genetic_algorithm(selection_method, label):
    """Executa o GA completo usando o método de seleção especificado."""
    population = initialize_population()
    best_scores = []

    for generation in range(NUM_GENERATIONS):
        new_population = []

        # elitismo simples
        best = max(population, key=objective_function)
        new_population.append(best)

        while len(new_population) < POP_SIZE:
            parent1 = selection_method(population)
            parent2 = selection_method(population)

            child = blend_crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

        best_ind = max(population, key=objective_function)
        best_val = objective_function(best_ind)
        best_scores.append(best_val)

        if generation % 10 == 0:
            print(f"[{label}] Geração {generation:02d} | Melhor f(x) = {best_val:.5f}")

    return best_scores


# -----------------------------------------------------------
# 8️⃣ Execução e comparação dos métodos
# -----------------------------------------------------------
if __name__ == "__main__":
    scores_tournament = run_genetic_algorithm(tournament_selection, "Torneio")
    scores_roulette = run_genetic_algorithm(roulette_selection, "Roleta")

    # Comparação visual
    plt.plot(scores_tournament, label="Seleção por Torneio")
    plt.plot(scores_roulette, label="Seleção por Roleta", linestyle="--")
    plt.title("Comparação de Métodos de Seleção no GA")
    plt.xlabel("Geração")
    plt.ylabel("Melhor f(x)")
    plt.legend()
    plt.grid(True)
    plt.show()
