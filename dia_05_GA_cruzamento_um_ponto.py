"""
📘 DIA 5 — Cruzamento de 1 Ponto (One-Point Crossover)
------------------------------------------------------

Objetivo:
---------
Demonstrar o funcionamento do crossover clássico de 1 ponto usando
representação binária. Essa é a forma mais tradicional de recombinação
na literatura de algoritmos genéticos.

Problema usado:
---------------
Maximizar a função f(x) = x^2
com x codificado em 8 bits (0 a 255).

- Cromossomos: strings binárias de tamanho 8
- Seleção: torneio (k=3)
- Cruzamento: 1 ponto
- Mutação: flip bit
"""

import random
import matplotlib.pyplot as plt

random.seed(42)

# ----------------------------------------------------------
# 1️⃣ Parâmetros do GA
# ----------------------------------------------------------
POP_SIZE = 40
NUM_GENERATIONS = 50
MUTATION_RATE = 0.02
CROSSOVER_RATE = 0.9
CHROMOSOME_LENGTH = 8  # representamos x ∈ [0, 255]


# ----------------------------------------------------------
# 2️⃣ Funções auxiliares
# ----------------------------------------------------------
def decode(binary_string: str) -> int:
    """Converte binário para inteiro."""
    return int(binary_string, 2)


def objective_function(x: int) -> int:
    """Função objetivo simples para testar recombinação."""
    return x ** 2


# ----------------------------------------------------------
# 3️⃣ Inicialização da população
# ----------------------------------------------------------
def initialize_population():
    population = []
    for _ in range(POP_SIZE):
        # cromossomo binário aleatório
        chromosome = "".join(random.choice("01") for _ in range(CHROMOSOME_LENGTH))
        population.append(chromosome)
    return population


# ----------------------------------------------------------
# 4️⃣ Seleção por torneio
# ----------------------------------------------------------
def tournament_selection(pop, k=3):
    candidates = random.sample(pop, k)
    return max(candidates, key=lambda c: objective_function(decode(c)))


# ----------------------------------------------------------
# 5️⃣ Cruzamento de 1 ponto
# ----------------------------------------------------------
def one_point_crossover(parent1, parent2):
    """
    Realiza crossover:
    - seleciona um ponto entre 1 e n-1
    - troca os segmentos
    """
    if random.random() > CROSSOVER_RATE:
        return parent1, parent2  # sem crossover

    point = random.randint(1, CHROMOSOME_LENGTH - 1)

    # recombinação
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]

    return child1, child2


# ----------------------------------------------------------
# 6️⃣ Mutação: flip bit
# ----------------------------------------------------------
def mutate(chromosome):
    new_bits = []
    for bit in chromosome:
        if random.random() < MUTATION_RATE:
            new_bits.append("1" if bit == "0" else "0")
        else:
            new_bits.append(bit)
    return "".join(new_bits)


# ----------------------------------------------------------
# 7️⃣ Execução do GA
# ----------------------------------------------------------
def run_ga():
    population = initialize_population()
    best_scores = []

    for gen in range(NUM_GENERATIONS):
        new_population = []

        # elitismo
        best = max(population, key=lambda c: objective_function(decode(c)))
        new_population.append(best)

        # gerar novos indivíduos
        while len(new_population) < POP_SIZE:
            p1 = tournament_selection(population)
            p2 = tournament_selection(population)

            c1, c2 = one_point_crossover(p1, p2)

            c1 = mutate(c1)
            c2 = mutate(c2)

            new_population.extend([c1, c2])

        population = new_population[:POP_SIZE]

        # registrar melhor da geração
        best_ind = max(population, key=lambda c: objective_function(decode(c)))
        best_val = objective_function(decode(best_ind))
        best_scores.append(best_val)

        if gen % 10 == 0:
            print(f"Geração {gen:02d} | Melhor x = {decode(best_ind):3d} | f(x) = {best_val}")

    return best_scores


# ----------------------------------------------------------
# 8️⃣ Gráfico de convergência
# ----------------------------------------------------------
if __name__ == "__main__":
    scores = run_ga()

    plt.plot(scores)
    plt.title("Cruzamento de 1 Ponto — Convergência do GA")
    plt.xlabel("Geração")
    plt.ylabel("Melhor f(x)")
    plt.grid(True)
    plt.show()

