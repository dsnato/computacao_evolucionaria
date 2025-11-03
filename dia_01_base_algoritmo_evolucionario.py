"""
📘 DIA 1 — Estrutura Base de um Algoritmo Evolucionário
------------------------------------------------------

Objetivo:
---------
Criar a estrutura genérica de um algoritmo evolucionário (EA),
incluindo:
- Criação da população inicial
- Avaliação (fitness)
- Seleção
- Cruzamento (recombinação)
- Mutação
- Substituição e iteração por gerações

Usaremos um exemplo simples de otimização da função f(x) = x²
buscando o valor máximo de x² dentro de um intervalo [-10, 10].
"""

import random
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# 1️⃣ Parâmetros do algoritmo
# -----------------------------------------------------------

POP_SIZE = 20           # Tamanho da população
NUM_GENERATIONS = 30    # Número de gerações
MUTATION_RATE = 0.1     # Probabilidade de mutação
X_MIN, X_MAX = -10, 10  # Intervalo de busca


# -----------------------------------------------------------
# 2️⃣ Função de avaliação (fitness)
# -----------------------------------------------------------
def fitness(x: float) -> float:
    """
    Calcula o valor de aptidão (fitness) de um indivíduo.
    Aqui queremos maximizar f(x) = x².
    """
    return x ** 2


# -----------------------------------------------------------
# 3️⃣ Inicialização da população
# -----------------------------------------------------------
def initialize_population() -> list:
    """
    Gera uma lista inicial de indivíduos (valores aleatórios entre X_MIN e X_MAX).
    """
    return [random.uniform(X_MIN, X_MAX) for _ in range(POP_SIZE)]


# -----------------------------------------------------------
# 4️⃣ Seleção
# -----------------------------------------------------------
def selection(population: list) -> float:
    """
    Seleciona um indivíduo da população com base no fitness (proporcional à qualidade).
    Método simples: escolha de 2 aleatórios e seleção do melhor (torneio).
    """
    a, b = random.sample(population, 2)
    return a if fitness(a) > fitness(b) else b


# -----------------------------------------------------------
# 5️⃣ Cruzamento (recombinação)
# -----------------------------------------------------------
def crossover(parent1: float, parent2: float) -> float:
    """
    Gera um novo indivíduo (filho) combinando dois pais.
    Aqui usamos média simples entre os pais.
    """
    return (parent1 + parent2) / 2


# -----------------------------------------------------------
# 6️⃣ Mutação
# -----------------------------------------------------------
def mutate(x: float) -> float:
    """
    Aplica uma mutação aleatória ao indivíduo com uma pequena perturbação.
    """
    if random.random() < MUTATION_RATE:
        x += random.uniform(-1, 1)  # Pequena variação
    return max(min(x, X_MAX), X_MIN)  # Garante que x fique nos limites


# -----------------------------------------------------------
# 7️⃣ Loop evolutivo principal
# -----------------------------------------------------------
def evolutionary_algorithm():
    population = initialize_population()
    best_scores = []

    for generation in range(NUM_GENERATIONS):
        new_population = []

        # Cria a nova geração
        for _ in range(POP_SIZE):
            # Seleção dos pais
            parent1 = selection(population)
            parent2 = selection(population)

            # Cruzamento e mutação
            child = crossover(parent1, parent2)
            child = mutate(child)

            new_population.append(child)

        # Substituição da população antiga pela nova
        population = new_population

        # Avaliação do melhor indivíduo
        best = max(population, key=fitness)
        best_fitness = fitness(best)
        best_scores.append(best_fitness)

        print(f"Geração {generation+1:02d} | Melhor indivíduo: {best:.4f} | Fitness: {best_fitness:.4f}")

    # Visualização da convergência
    plt.plot(best_scores)
    plt.title("Convergência do Algoritmo Evolucionário")
    plt.xlabel("Geração")
    plt.ylabel("Melhor Fitness")
    plt.show()


# -----------------------------------------------------------
# 8️⃣ Execução do código
# -----------------------------------------------------------
if __name__ == "__main__":
    evolutionary_algorithm()
