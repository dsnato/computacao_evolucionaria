"""
📘 DIA 2 — Algoritmo Genético Clássico (GA) - Maximização de Função Simples
----------------------------------------------------------------------------

Objetivo:
---------
Implementar um **Algoritmo Genético (GA)** clássico em Python
para **maximizar a função**:

    f(x) = x * sin(10πx) + 1.0

no intervalo **[-1, 2]**.

Este é um exemplo clássico da literatura (Goldberg, 1989)
usado para demonstrar o comportamento do GA em funções multimodais.

O algoritmo segue as etapas básicas:
1. Geração da população inicial (valores reais)
2. Avaliação do fitness
3. Seleção (torneio)
4. Cruzamento (blend crossover - média ponderada)
5. Mutação (pequena perturbação gaussiana)
6. Substituição
7. Registro do melhor indivíduo por geração

O script imprime o progresso e plota a curva de convergência.
"""

import random
import math
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# 1️⃣ Parâmetros do GA
# -----------------------------------------------------------
POP_SIZE = 30           # Tamanho da população
NUM_GENERATIONS = 50    # Número de gerações
MUTATION_RATE = 0.1     # Probabilidade de mutação
CROSSOVER_RATE = 0.9    # Probabilidade de cruzamento
X_MIN, X_MAX = -1, 2    # Intervalo de busca (domínio)
RANDOM_SEED = 42        # Para reprodutibilidade

random.seed(RANDOM_SEED)


# -----------------------------------------------------------
# 2️⃣ Função objetivo (a ser maximizada)
# -----------------------------------------------------------
def objective_function(x: float) -> float:
    """
    Função multimodal usada para testar GAs.
    Possui vários máximos locais.
    """
    return x * math.sin(10 * math.pi * x) + 1.0


# -----------------------------------------------------------
# 3️⃣ Inicialização da população
# -----------------------------------------------------------
def initialize_population():
    """Gera uma lista de valores aleatórios (indivíduos) dentro do intervalo definido."""
    return [random.uniform(X_MIN, X_MAX) for _ in range(POP_SIZE)]


# -----------------------------------------------------------
# 4️⃣ Seleção (torneio de 2)
# -----------------------------------------------------------
def tournament_selection(population):
    """Seleciona dois indivíduos aleatórios e retorna o melhor."""
    a, b = random.sample(population, 2)
    return a if objective_function(a) > objective_function(b) else b


# -----------------------------------------------------------
# 5️⃣ Cruzamento (Blend Crossover - média ponderada)
# -----------------------------------------------------------
def blend_crossover(parent1, parent2):
    """
    Cruzamento do tipo BLX-α (simplificado).
    Gera um filho dentro da faixa entre os pais, com pequena extrapolação.
    """
    if random.random() > CROSSOVER_RATE:
        return parent1  # sem cruzamento

    alpha = 0.5  # controle do peso
    diff = abs(parent1 - parent2)
    low = min(parent1, parent2) - alpha * diff
    high = max(parent1, parent2) + alpha * diff
    child = random.uniform(low, high)
    return max(min(child, X_MAX), X_MIN)


# -----------------------------------------------------------
# 6️⃣ Mutação (adição de ruído gaussiano)
# -----------------------------------------------------------
def mutate(x):
    """Aplica mutação gaussiana com pequena variância."""
    if random.random() < MUTATION_RATE:
        x += random.gauss(0, 0.1)
    return max(min(x, X_MAX), X_MIN)


# -----------------------------------------------------------
# 7️⃣ Loop evolutivo principal
# -----------------------------------------------------------
def genetic_algorithm():
    population = initialize_population()
    best_scores = []

    for generation in range(NUM_GENERATIONS):
        new_population = []

        # Elitismo simples — mantém o melhor da geração anterior
        best = max(population, key=objective_function)
        new_population.append(best)

        # Gera nova população
        while len(new_population) < POP_SIZE:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)

            child = blend_crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        # Atualiza população
        population = new_population

        # Avalia o melhor indivíduo
        best_individual = max(population, key=objective_function)
        best_value = objective_function(best_individual)
        best_scores.append(best_value)

        print(f"Geração {generation+1:02d} | Melhor x = {best_individual:.5f} | f(x) = {best_value:.5f}")

    # Gráfico da convergência
    plt.plot(best_scores, label="Melhor Fitness")
    plt.title("Convergência — Algoritmo Genético Clássico")
    plt.xlabel("Geração")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.show()


# -----------------------------------------------------------
# 8️⃣ Execução
# -----------------------------------------------------------
if __name__ == "__main__":
    genetic_algorithm()

