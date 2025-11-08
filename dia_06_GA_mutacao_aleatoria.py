"""
📘 DIA 6 — Mutação Aleatória (Random Mutation)
----------------------------------------------

Operador de mutação simples para manter diversidade em
Algoritmos Genéticos clássicos com codificação binária.

- Problema: maximizar f(x) = x²
- Representação: 8 bits (inteiros de 0 a 255)
- Mutação aplicada bit a bit com taxa definida.
"""

import random
import matplotlib.pyplot as plt

random.seed(42)

# -----------------------------------------------------------
# 1️⃣ Parâmetros do GA
# -----------------------------------------------------------
POP_SIZE = 40
NUM_GENERATIONS = 50
MUTATION_RATE = 0.02  # probabilidade por bit
CHROMOSOME_LENGTH = 8


# -----------------------------------------------------------
# 2️⃣ Funções auxiliares
# -----------------------------------------------------------
def decode(binary_string):
    return int(binary_string, 2)

def objective_function(x):
    return x ** 2


# -----------------------------------------------------------
# 3️⃣ Inicialização
# -----------------------------------------------------------
def initialize_population():
    population = []
    for _ in range(POP_SIZE):
        chromosome = "".join(random.choice("01") for _ in range(CHROMOSOME_LENGTH))
        population.append(chromosome)
    return population


# -----------------------------------------------------------
# 4️⃣ Seleção simples (torneio)
# -----------------------------------------------------------
def tournament_selection(population, k=3):
    candidates = random.sample(population, k)
    return max(candidates, key=lambda c: objective_function(decode(c)))


# -----------------------------------------------------------
# 5️⃣ Mutação aleatória (bit-flip)
# -----------------------------------------------------------
def mutate(chromosome):
    """
    Percorre cada bit do cromossomo e, com uma probabilidade MUTATION_RATE,
    troca "0" por "1" ou "1" por "0".
    """
    new_bits = []
    for bit in chromosome:
        if random.random() < MUTATION_RATE:
            new_bits.append("1" if bit == "0" else "0")
        else:
            new_bits.append(bit)
    return "".join(new_bits)


# -----------------------------------------------------------
# 6️⃣ Loop do GA (sem cruzamento neste dia)
# -----------------------------------------------------------
def run_ga():
    population = initialize_population()
    best_scores = []

    for gen in range(NUM_GENERATIONS):

        new_population = []

        # elitismo
        best = max(population, key=lambda c: objective_function(decode(c)))
        new_population.append(best)

        while len(new_population) < POP_SIZE:
            p = tournament_selection(population)
            mutated = mutate(p)
            new_population.append(mutated)

        population = new_population

        best_ind = max(population, key=lambda c: objective_function(decode(c)))
        best_val = objective_function(decode(best_ind))
        best_scores.append(best_val)

        if gen % 10 == 0:
            print(f"Geração {gen:02d} | Melhor f(x) = {best_val}")

    return best_scores


# -----------------------------------------------------------
# 7️⃣ Execução e plot
# -----------------------------------------------------------
if __name__ == "__main__":
    scores = run_ga()

    plt.plot(scores)
    plt.title("Mutação Aleatória — Convergência do GA")
    plt.xlabel("Geração")
    plt.ylabel("Melhor f(x)")
    plt.grid(True)
    plt.show()
