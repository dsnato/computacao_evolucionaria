"""
📘 DIA 13 — Evolução Diferencial (DE) — DE/rand/1/bin
----------------------------------------------------

Objetivo:
---------
Implementar o algoritmo Differential Evolution (DE) para minimizar
uma função multimodal (Rastrigin). Usamos a variante clássica DE/rand/1/bin.

Características:
- Representação: indivíduos são vetores reais de dimensão D
- Mutação: vetores diferenciais (v = a + F*(b - c))
- Crossover: binomial (CR)
- Seleção: competição entre alvo e trial (sobrevive o melhor)
- Minimização: fitness = função objetivo (quanto menor, melhor)

Como executar:
    python dia_13_de.py

Dependências:
    numpy, matplotlib
"""
import numpy as np
import random
import math
import matplotlib.pyplot as plt

# -----------------------------
# 0) Reprodutibilidade
# -----------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# -----------------------------
# 1) Hiperparâmetros do DE
# -----------------------------
POP_SIZE = 60        # NP, número de vetores (população)
DIM = 10             # dimensão do problema (cada indivíduo tem DIM componentes)
GENS = 200           # número de gerações
F = 0.8              # fator de escala (mutation factor)
CR = 0.9             # probabilidade de crossover (crossover rate)
LOWER_BOUND = -5.12  # limites do domínio (Rastrigin típico)
UPPER_BOUND = 5.12

# -----------------------------
# 2) Função objetivo: Rastrigin (multimodal)
# -----------------------------
def rastrigin(vec):
    """
    Rastrigin function: multimodal, muitas armadilhas locais.
    f(x) = A*n + sum(x_i^2 - A*cos(2*pi*x_i)), A=10
    Mínimo global em x = 0 -> f = 0
    Input: 1D numpy array
    """
    A = 10.0
    return A * vec.size + np.sum(vec**2 - A * np.cos(2 * math.pi * vec))

# -----------------------------
# 3) Inicialização da população
# -----------------------------
def initialize_population(pop_size, dim, lb, ub):
    """
    Gera uma matriz (pop_size x dim) com valores uniformes entre lb e ub.
    Cada linha é um indivíduo.
    """
    return np.random.uniform(lb, ub, size=(pop_size, dim))

# -----------------------------
# 4) Mutação (DE/rand/1)
# -----------------------------
def mutation_rand_1(pop, idx, F):
    """
    Mutação tipo 'rand/1':
      - escolhe aleatoriamente 3 índices distintos: a, b, c != idx
      - devolve vetor mutante v = a + F * (b - c)
    Recebe a população (np.array), o índice do alvo e F.
    """
    pop_size = pop.shape[0]
    # escolhe 3 índices distintos e diferentes de idx
    indices = list(range(pop_size))
    indices.remove(idx)
    a, b, c = random.sample(indices, 3)
    mutant = pop[a] + F * (pop[b] - pop[c])
    return mutant

# -----------------------------
# 5) Crossover binomial (bin)
# -----------------------------
def crossover_binomial(target, mutant, CR):
    """
    Crossover binomial: para cada dimensão, escolhe gene do mutant com prob CR,
    caso contrário mantém do target. Garante que pelo menos um gene venha do mutant
    (jrand).
    """
    dim = target.size
    trial = target.copy()
    jrand = random.randrange(dim)  # garante pelo menos um gene trocado
    for j in range(dim):
        if random.random() < CR or j == jrand:
            trial[j] = mutant[j]
    return trial

# -----------------------------
# 6) Boundary handling (clamping)
# -----------------------------
def ensure_bounds(vec, lb, ub):
    """
    Garante que cada componente esteja dentro dos limites [lb, ub].
    Aqui usamos 'clamp' simples.
    """
    return np.minimum(np.maximum(vec, lb), ub)

# -----------------------------
# 7) Loop principal do DE
# -----------------------------
def differential_evolution(
    pop_size=POP_SIZE, dim=DIM, gens=GENS, F=F, CR=CR, lb=LOWER_BOUND, ub=UPPER_BOUND
):
    # inicializa população (matriz pop_size x dim)
    pop = initialize_population(pop_size, dim, lb, ub)

    # avalia fitness inicial (vectorizado)
    fitness_vals = np.array([rastrigin(ind) for ind in pop])

    best_history = []
    best_idx = np.argmin(fitness_vals)
    best_vec = pop[best_idx].copy()
    best_val = fitness_vals[best_idx]
    best_history.append(best_val)

    for g in range(1, gens + 1):
        for i in range(pop_size):
            # 1) mutação
            mutant = mutation_rand_1(pop, i, F)
            mutant = ensure_bounds(mutant, lb, ub)

            # 2) crossover
            trial = crossover_binomial(pop[i], mutant, CR)
            trial = ensure_bounds(trial, lb, ub)

            # 3) seleção: comparamos fitness(target) vs fitness(trial)
            f_trial = rastrigin(trial)
            if f_trial <= fitness_vals[i]:
                # trial substitui target
                pop[i] = trial
                fitness_vals[i] = f_trial

                # atualização do melhor global se necessário
                if f_trial < best_val:
                    best_val = f_trial
                    best_vec = trial.copy()

        best_history.append(best_val)

        # logging simples a cada 10 gerações
        if g % 10 == 0 or g == 1:
            print(f"Geração {g:03d} | Melhor fitness = {best_val:.6f}")

    return best_vec, best_val, best_history, pop, fitness_vals

# -----------------------------
# 8) Execução
# -----------------------------
if __name__ == "__main__":
    best_vec, best_val, history, final_pop, final_fitness = differential_evolution()
    print("\n=== Resultado final ===")
    print("Melhor fitness encontrado:", best_val)
    print("Melhor vetor (primeiras 6 componentes):", np.round(best_vec[:6], 6))

    # plot da convergência
    plt.figure(figsize=(8,4))
    plt.plot(history, label="Melhor fitness (mín) por geração")
    plt.xlabel("Geração")
    plt.ylabel("Fitness (Rastrigin) — menor é melhor")
    plt.title("Differential Evolution — Convergência")
    plt.grid(True)
    plt.legend()
    plt.show()

    # histograma dos fitness finais
    plt.figure(figsize=(6,3))
    plt.hist(final_fitness, bins=20)
    plt.title("Distribuição dos fitness na população final")
    plt.xlabel("Fitness")
    plt.ylabel("Contagem")
    plt.grid(True)
    plt.show()
