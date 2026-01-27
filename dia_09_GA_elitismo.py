"""
📘 DIA 09 — Algoritmo Genético com Elitismo
--------------------------------------------

Objetivo:
---------
Mostrar como preservar as melhores soluções de uma geração
para outra usando elitismo.

Estratégia:
-----------
- Representação: binária (5 bits)
- Seleção: torneio
- Crossover: 1 ponto
- Mutação: bit-flip
- Elitismo: preserva top N indivíduos

Função: f(x) = x^2, com x ∈ [0, 31]
"""

import random
import matplotlib.pyplot as plt

random.seed(42)

# -----------------------------------------------------------
# 1️⃣ Parâmetros
# -----------------------------------------------------------
POP_SIZE = 40
N_BITS = 5
NUM_GENERATIONS = 50
MUTATION_RATE = 0.05
TOURNAMENT_K = 3
ELITE_SIZE = 2  # número de melhores indivíduos preservados


# -----------------------------------------------------------
# 2️⃣ Funções básicas
# -----------------------------------------------------------
def decode(chromosome):
    """Converte binário -> inteiro."""
    return int(chromosome, 2)


def fitness(chromosome):
    """Função de aptidão."""
    x = decode(chromosome)
    return x ** 2


# -----------------------------------------------------------
# 3️⃣ Inicialização da população
# -----------------------------------------------------------
def random_chromosome():
    """Cria um cromossomo binário aleatório."""
    return ''.join(random.choice(['0', '1']) for _ in range(N_BITS))


def initialize_population():
    return [random_chromosome() for _ in range(POP_SIZE)]


# -----------------------------------------------------------
# 4️⃣ Seleção por torneio
# -----------------------------------------------------------
def tournament_selection(population):
    """Seleciona o melhor de k candidatos."""
    competitors = random.sample(population, TOURNAMENT_K)
    return max(competitors, key=fitness)


# -----------------------------------------------------------
# 5️⃣ Crossover e mutação
# -----------------------------------------------------------
def crossover(p1, p2):
    """Cruzamento de 1 ponto."""
    point = random.randint(1, N_BITS - 1)
    return p1[:point] + p2[point:]


def mutate(chromosome):
    """Mutação bit-flip."""
    bits = []
    for bit in chromosome:
        if random.random() < MUTATION_RATE:
            bits.append('1' if bit == '0' else '0')
        else:
            bits.append(bit)
    return ''.join(bits)


# -----------------------------------------------------------
# 6️⃣ Elitismo
# -----------------------------------------------------------
def get_elite(population, n):
    """Retorna os n melhores indivíduos."""
    return sorted(population, key=fitness, reverse=True)[:n]


# -----------------------------------------------------------
# 7️⃣ Loop principal
# -----------------------------------------------------------
def run_ga():
    population = initialize_population()
    best_history = []

    for gen in range(NUM_GENERATIONS):
        new_population = []

        # 🔹 Elitismo: preserva os melhores
        elites = get_elite(population, ELITE_SIZE)
        new_population.extend(elites)

        # 🔹 Reprodução
        while len(new_population) < POP_SIZE:
            p1 = tournament_selection(population)
            p2 = tournament_selection(population)
            child = crossover(p1, p2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

        # 🔹 Melhor da geração
        best_ind = max(population, key=fitness)
        best_val = fitness(best_ind)
        best_history.append(best_val)

        if gen % 10 == 0:
            print(f"Geração {gen:02d} | Melhor x = {decode(best_ind):2d} | f(x) = {best_val}")

    return best_history


# -----------------------------------------------------------
# 8️⃣ Execução
# -----------------------------------------------------------
if __name__ == "__main__":
    history = run_ga()

    plt.plot(history)
    plt.title("GA com Elitismo — Evolução do Fitness")
    plt.xlabel("Geração")
    plt.ylabel("Melhor Fitness")
    plt.grid(True)
    plt.show()

