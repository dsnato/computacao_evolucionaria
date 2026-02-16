"""
📘 DIA 08 — Algoritmo Genético Binário
---------------------------------------

GA que usa:
- Cromossomos binários (strings de '0' e '1')
- Crossover de 1 ponto
- Mutação bit-flip
- Seleção por torneio

Problema: maximizar f(x) = x^2
com x representado em 5 bits.
"""

import random
import matplotlib.pyplot as plt

random.seed(42)

# ----------------------------------------------------------
# 1️⃣ Parâmetros do GA
# ----------------------------------------------------------
POP_SIZE = 30
N_BITS = 5                  # tamanho do cromossomo
NUM_GENERATIONS = 40
MUTATION_RATE = 0.05
TOURNAMENT_K = 3


# -----------------------------------------------------------
# 2️⃣ Converte cromossomo binário -> inteiro
# -----------------------------------------------------------
def decode(chromosome):
    return int(chromosome, 2)


# -----------------------------------------------------------
# 3️⃣ Função de fitness
# -----------------------------------------------------------
def fitness(chromosome):
    x = decode(chromosome)
    return x ** 2  # otimizar x^2


# -----------------------------------------------------------
# 4️⃣ Inicialização aleatória (bitstring)
# -----------------------------------------------------------
def random_chromosome():
    return ''.join(random.choice(['0', '1']) for _ in range(N_BITS))


def initialize_population():
    return [random_chromosome() for _ in range(POP_SIZE)]


# -----------------------------------------------------------
# 5️⃣ Seleção por torneio
# -----------------------------------------------------------
def tournament_selection(population):
    competitors = random.sample(population, TOURNAMENT_K)
    winner = max(competitors, key=fitness)
    return winner


# -----------------------------------------------------------
# 6️⃣ Crossover de 1 ponto
# -----------------------------------------------------------
def crossover(p1, p2):
    point = random.randint(1, N_BITS - 1)
    child = p1[:point] + p2[point:]
    return child


# -----------------------------------------------------------
# 7️⃣ Mutação bit-flip
# -----------------------------------------------------------
def mutate(chromosome):
    new_bits = []
    for bit in chromosome:
        if random.random() < MUTATION_RATE:
            new_bits.append('1' if bit == '0' else '0')
        else:
            new_bits.append(bit)
    return ''.join(new_bits)


# -----------------------------------------------------------
# 8️⃣ Loop principal do GA
# -----------------------------------------------------------
def run_ga():
    population = initialize_population()
    best_history = []

    for gen in range(NUM_GENERATIONS):
        new_population = []

        # elitismo
        elite = max(population, key=fitness)
        new_population.append(elite)

        # reprodução
        while len(new_population) < POP_SIZE:
            p1 = tournament_selection(population)
            p2 = tournament_selection(population)
            child = crossover(p1, p2)
            child = mutate(child)
            new_population.append(child)

        population = new_population
        best_f = fitness(max(population, key=fitness))
        best_history.append(best_f)

        if gen % 5 == 0:
            print(f"Geração {gen} | Melhor fitness: {best_f}")

    return best_history


# -----------------------------------------------------------
# 9️⃣ Execução + gráfico
# -----------------------------------------------------------
if __name__ == "__main__":
    history = run_ga()

    plt.plot(history)
    plt.title("Evolução do Fitness (GA Binário)")
    plt.xlabel("Geração")
    plt.ylabel("Melhor Fitness")
    plt.grid(True)
    plt.show()


