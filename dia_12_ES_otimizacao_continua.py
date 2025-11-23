"""
📘 DIA 12 — Estratégias Evolutivas (ES)
----------------------------------------

Implementação do clássico (μ + λ)-ES para otimização contínua.

Problema:
---------
Minimizar f(x,y) = (x - 3)^2 + (y + 2)^2

Indivíduo:
----------
(x, y, σ_x, σ_y)

Mutação:
--------
x' = x + σ_x * N(0,1)
σ' = σ * exp(t * N(0,1))   # adaptação do passo

Parâmetros:
-----------
μ = 10   pais
λ = 40   filhos
"""

import random
import math
import matplotlib.pyplot as plt

random.seed(42)

# -------------------------------------------------------------
# 1️⃣ Função objetivo
# -------------------------------------------------------------
def fitness(ind):
    """Retorna o valor da função f(x,y) a ser minimizada."""
    x, y, _, _ = ind
    return (x - 3)**2 + (y + 2)**2


# -------------------------------------------------------------
# 2️⃣ Inicialização da população
# -------------------------------------------------------------
def initialize_population(mu=10):
    """
    Cada indivíduo é representado como:
    (x, y, sigma_x, sigma_y)
    """

    population = []
    for _ in range(mu):
        x = random.uniform(-10, 10)
        y = random.uniform(-10, 10)
        sigma_x = random.uniform(0.1, 1.0)
        sigma_y = random.uniform(0.1, 1.0)
        population.append((x, y, sigma_x, sigma_y))
    return population


# -------------------------------------------------------------
# 3️⃣ Mutação Gaussian + adaptação de σ
# -------------------------------------------------------------
def mutate(ind):
    """
    Estratégias Evolutivas usam:
      - Mutação gaussiana nos genes reais
      - Adaptação multiplicativa em σ via exp
    """
    x, y, sigma_x, sigma_y = ind

    # parâmetros clássicos
    t = 1 / math.sqrt(2)

    # adapta sigma
    sigma_x_new = sigma_x * math.exp(t * random.gauss(0, 1))
    sigma_y_new = sigma_y * math.exp(t * random.gauss(0, 1))

    # garante valores mínimos
    sigma_x_new = max(sigma_x_new, 0.001)
    sigma_y_new = max(sigma_y_new, 0.001)

    # aplica mutação real
    x_new = x + sigma_x_new * random.gauss(0, 1)
    y_new = y + sigma_y_new * random.gauss(0, 1)

    return (x_new, y_new, sigma_x_new, sigma_y_new)


# -------------------------------------------------------------
# 4️⃣ Reprodução λ filhos
# -------------------------------------------------------------
def reproduce(population, lambd=40):
    children = []
    for _ in range(lambd):
        parent = random.choice(population)
        child = mutate(parent)
        children.append(child)
    return children


# -------------------------------------------------------------
# 5️⃣ Ciclo principal (μ + λ)-ES
# -------------------------------------------------------------
def evolution_strategy(mu=10, lambd=40, generations=80):

    population = initialize_population(mu)
    best_history = []

    for g in range(generations):

        # gera λ filhos
        children = reproduce(population, lambd=lambd)

        # união μ + λ
        combined = population + children

        # selecionar os μ melhores
        combined_sorted = sorted(combined, key=lambda ind: fitness(ind))
        population = combined_sorted[:mu]

        # melhor da geração
        best = population[0]
        best_history.append(fitness(best))

        if g % 10 == 0:
            print(f"Geração {g:02d} | Melhor f = {fitness(best):.4f} | x,y = {best[0]:.3f}, {best[1]:.3f}")

    return population[0], best_history


# -------------------------------------------------------------
# 6️⃣ Execução + visualização
# -------------------------------------------------------------
if __name__ == "__main__":
    best, hist = evolution_strategy()

    print("\nMelhor solução final:")
    print(f"x = {best[0]:.4f}, y = {best[1]:.4f}, fitness = {fitness(best):.6f}")

    plt.plot(hist)
    plt.title("Estratégia Evolutiva (μ + λ)-ES — Convergência")
    plt.xlabel("Geração")
    plt.ylabel("Melhor fitness")
    plt.grid(True)
    plt.show()
