"""
📘 DIA 11 — Algoritmos Meméticos (GA + Local Search)
----------------------------------------------------

Objetivo:
---------
Implementar um Algoritmo Memético (híbrido): um Algoritmo Genético real-valued
com a adição de uma busca local (hill-climbing / busca por vizinhança) aplicada
aos descendentes para refinar soluções (memes = refinamentos locais).

Problema de teste:
------------------
O algoritmo otimiza a função Rastrigin (multimodal) em dimensão D:
    f(x) = A*D + sum(x_i^2 - A*cos(2π x_i)), com A=10
Objetivo: minimizar f(x) (mínimo global em x=0 com f=0)

Características:
- Representação: vetor de floats (real-valued)
- Seleção: torneio
- Crossover: blend arithmetic (real)
- Mutação: gaussiana por dimensão
- Busca local: hill-climbing aleatório (pequenos passos gaussianos) aplicada com probabilidade LOCAL_SEARCH_PROB ao filho
- Elitismo: preserva melhores
- Saída: imprime progresso e plota convergência (melhor fitness por geração)
"""
import random
import math
import copy
import matplotlib.pyplot as plt

# -----------------------------
# 1) Hiperparâmetros
# -----------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

POP_SIZE = 60              # população
NUM_GENERATIONS = 120      # gerações
DIM = 5                    # dimensão do problema (vetor x ∈ R^DIM)
LOWER_BOUND = -5.12        # domínio Rastrigin
UPPER_BOUND = 5.12

TOURNAMENT_K = 3           # torneio
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.2        # probabilidade de mutar cada gene
MUTATION_STD = 0.2         # desvio-padrão da mutação gaussiana

ELITE_SIZE = 2             # preserva os top N indivíduos

LOCAL_SEARCH_PROB = 0.6    # probabilidade de aplicar busca local em um filho
LOCAL_SEARCH_ITERS = 20    # iterações de hill-climbing por aplicação
LOCAL_STEP_STD = 0.1       # escala dos passos locais (gaussiano)

# -----------------------------
# 2) Função Rastrigin (a minimizar)
# -----------------------------
def rastrigin(x):
    """
    Rastrigin function: multimodal com muitos ótimos locais.
    f(x) >= 0; f(0)=0.
    """
    A = 10.0
    return A * len(x) + sum([(xi ** 2 - A * math.cos(2 * math.pi * xi)) for xi in x])

# -----------------------------
# 3) Inicialização (população de vetores reais)
# -----------------------------
def random_individual():
    """Cria indivíduo aleatório (lista de floats de dimensão DIM)."""
    return [random.uniform(LOWER_BOUND, UPPER_BOUND) for _ in range(DIM)]

def initialize_population():
    return [random_individual() for _ in range(POP_SIZE)]

# -----------------------------
# 4) Avaliação (fitness) - aqui, lower is better
# -----------------------------
def evaluate(individual):
    """Avalia um indivíduo pela função objetivo (fitness = valor a minimizar)."""
    return rastrigin(individual)

# -----------------------------
# 5) Seleção: torneio
# -----------------------------
def tournament_selection(population, k=TOURNAMENT_K):
    """Retorna uma cópia do vencedor do torneio (melhor entre k amostras)."""
    candidates = random.sample(population, k)
    winner = min(candidates, key=evaluate)  # min, pois queremos minimizar
    return copy.deepcopy(winner)

# -----------------------------
# 6) Crossover: blend/arithmetic
# -----------------------------
def blend_crossover(p1, p2, alpha=0.5):
    """
    BLX-like / arithmetic blend: cria um filho pontual entre p1 e p2.
    Simples: child = alpha*p1 + (1-alpha)*p2 (alpha sorteado).
    Retorna um único filho (pode-se criar 2 usando inversão de pais).
    """
    child = []
    for a, b in zip(p1, p2):
        a0 = min(a, b)
        b0 = max(a, b)
        # BLX-alpha-like sampling com extrapolação controlada
        interval = b0 - a0
        low = a0 - alpha * interval
        high = b0 + alpha * interval
        val = random.uniform(low, high)
        # clamp para limites definidos
        val = max(min(val, UPPER_BOUND), LOWER_BOUND)
        child.append(val)
    return child

# -----------------------------
# 7) Mutação gaussiana (per-gene)
# -----------------------------
def mutate(individual):
    """Aplica mutação gaussiana por gene com probabilidade MUTATION_RATE."""
    mutant = []
    for gene in individual:
        if random.random() < MUTATION_RATE:
            gene = gene + random.gauss(0, MUTATION_STD)
        # garante limites
        gene = max(min(gene, UPPER_BOUND), LOWER_BOUND)
        mutant.append(gene)
    return mutant

# -----------------------------
# 8) Busca local (hill-climbing aleatório)
# -----------------------------
def local_search_hillclimb(individual, iters=LOCAL_SEARCH_ITERS, step_std=LOCAL_STEP_STD):
    """
    Aplica uma busca local simples: em cada iteração gera uma vizinhança
    por pequenos passos gaussianos; se encontrar vizinho melhor (menor fitness),
    aceita o vizinho. Retorna indivíduo refinado.
    """
    current = copy.deepcopy(individual)
    current_f = evaluate(current)

    for _ in range(iters):
        # gera vizinho por perturbação gaussiana
        neighbor = [max(min(g + random.gauss(0, step_std), UPPER_BOUND), LOWER_BOUND) for g in current]
        neighbor_f = evaluate(neighbor)
        if neighbor_f < current_f:
            current, current_f = neighbor, neighbor_f
    return current

# -----------------------------
# 9) Geração nova com elitismo + memética
# -----------------------------
def create_new_generation(population):
    """
    Gera nova população:
      - preserva elites
      - cria filhos por seleção, crossover, mutação
      - aplica busca local em alguns filhos (memetic refinement)
    """
    # ordena população pelo fitness (menor é melhor)
    sorted_pop = sorted(population, key=evaluate)
    new_pop = [copy.deepcopy(ind) for ind in sorted_pop[:ELITE_SIZE]]  # preserva elites

    while len(new_pop) < POP_SIZE:
        # seleção de pais
        parent1 = tournament_selection(population)
        parent2 = tournament_selection(population)

        # crossover (com probabilidade)
        if random.random() < CROSSOVER_RATE:
            child = blend_crossover(parent1, parent2)
        else:
            # sem crossover, copia um dos pais
            child = copy.deepcopy(parent1 if random.random() < 0.5 else parent2)

        # mutação
        child = mutate(child)

        # busca local (memética) com probabilidade
        if random.random() < LOCAL_SEARCH_PROB:
            child = local_search_hillclimb(child)

        new_pop.append(child)

    return new_pop[:POP_SIZE]

# -----------------------------
# 10) Loop principal do Algoritmo Memético
# -----------------------------
def memetic_algorithm():
    population = initialize_population()
    best_history = []

    for gen in range(1, NUM_GENERATIONS + 1):
        # registra melhor atual
        best = min(population, key=evaluate)
        best_f = evaluate(best)
        best_history.append(best_f)

        if gen % 10 == 0 or gen == 1:
            print(f"Geração {gen:03d} | Melhor fitness (mín) = {best_f:.6f}")

        # gerar próxima geração
        population = create_new_generation(population)

    # retorno do melhor e histórico
    best = min(population, key=evaluate)
    return best, best_history

# -----------------------------
# 11) Execução do experimento
# -----------------------------
if __name__ == "__main__":
    best_ind, history = memetic_algorithm()

    print("\nMelhor indivíduo encontrado:")
    print([round(v, 6) for v in best_ind])
    print("Fitness (Rastrigin) =", evaluate(best_ind))

    # plot da convergência (menor fitness por geração)
    plt.figure(figsize=(9, 4))
    plt.plot(history, label="Melhor fitness (mín) por geração")
    plt.xlabel("Geração")
    plt.ylabel("Fitness (Rastrigin) — menor é melhor")
    plt.title("Algoritmo Memético — GA + Busca Local")
    plt.grid(True)
    plt.legend()
    plt.show()


