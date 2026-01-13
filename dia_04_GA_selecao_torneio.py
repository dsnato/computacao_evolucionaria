"""
📘 DIA 4 — Seleção por Torneio e Pressão Seletiva
--------------------------------------------------

Objetivo:
---------
Demonstrar como a pressão seletiva pode ser ajustada no GA modificando
o tamanho do torneio (k). Quanto maior o k, maior a chance de selecionar
indivíduos muito bons → convergência mais rápida, porém risco maior de
perder diversidade.

Função usada:
    f(x) = x * sin(10πx) + 1.0
em [-1, 2], mesma dos dias anteriores.

Serão comparados:
- Torneio k = 2  (pressão baixa)
- Torneio k = 3  (pressão moderada)
- Torneio k = 5  (pressão alta)
- Torneio k = 10 (pressão muito alta → elitismo extremo)
"""

import random
import math
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# 1️⃣ Parâmetros Base do GA
# -----------------------------------------------------------
POP_SIZE = 40
NUM_GENERATIONS = 60
MUTATION_RATE = 0.12
CROSSOVER_RATE = 0.9
X_MIN, X_MAX = -1, 2
random.seed(42)


# -----------------------------------------------------------
# 2️⃣ Função objetivo
# -----------------------------------------------------------
def objective_function(x):
    """Função multimodal de teste."""
    return x * math.sin(10 * math.pi * x) + 1.0


# -----------------------------------------------------------
# 3️⃣ Inicialização
# -----------------------------------------------------------
def initialize_population():
    return [random.uniform(X_MIN, X_MAX) for _ in range(POP_SIZE)]


# -----------------------------------------------------------
# 4️⃣ Seleção por Torneio
# -----------------------------------------------------------
def tournament_selection(population, k=2):
    """Seleciona o melhor indivíduo entre k escolhidos aleatoriamente."""
    candidates = random.sample(population, k)
    return max(candidates, key=objective_function)


# -----------------------------------------------------------
# 5️⃣ Operadores do GA
# -----------------------------------------------------------
def crossover(parent1, parent2):
    """Cruzamento BLX-α."""
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
# 6️⃣ Loop do GA
# -----------------------------------------------------------
def run_ga(k_tournament):
    population = initialize_population()
    best_scores = []

    for gen in range(NUM_GENERATIONS):

        new_population = []

        # Elitismo explícito: sempre preservamos o melhor
        best = max(population, key=objective_function)
        new_population.append(best)

        while len(new_population) < POP_SIZE:
            p1 = tournament_selection(population, k=k_tournament)
            p2 = tournament_selection(population, k=k_tournament)

            child = crossover(p1, p2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

        best_gen = max(population, key=objective_function)
        best_scores.append(objective_function(best_gen))

        if gen % 10 == 0:
            print(f"[k={k_tournament}] Geração {gen:02d} | Melhor f(x) = {best_scores[-1]:.5f}")

    return best_scores


# -----------------------------------------------------------
# 7️⃣ Execução e comparação
# -----------------------------------------------------------
if __name__ == "__main__":
    results = {
        2: run_ga(2),
        3: run_ga(3),
        5: run_ga(5),
        10: run_ga(10),
    }

    # Gráfico comparativo
    for k, scores in results.items():
        plt.plot(scores, label=f"Torneio k={k}")

    plt.title("Comparação da Pressão Seletiva (Torneio k)")
    plt.xlabel("Geração")
    plt.ylabel("Melhor Fitness")
    plt.legend()
    plt.grid(True)
    plt.show()

