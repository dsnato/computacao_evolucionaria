"""
📘 DIA 16 — Particle Swarm Optimization (PSO)
---------------------------------------------

Minimização da função Rastrigin.
"""

import numpy as np
import random
import math
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1️⃣ Parâmetros do PSO
# ---------------------------------------------------------
NUM_PARTICLES = 40
DIM = 10
ITERATIONS = 150

w = 0.7      # peso de inércia
c1 = 1.5     # coeficiente cognitivo
c2 = 1.5     # coeficiente social

LOWER_BOUND = -5.12
UPPER_BOUND = 5.12

random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------
# 2️⃣ Função Rastrigin
# ---------------------------------------------------------
def rastrigin(x):
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * math.pi * x))


# ---------------------------------------------------------
# 3️⃣ Inicialização
# ---------------------------------------------------------
positions = np.random.uniform(LOWER_BOUND, UPPER_BOUND, (NUM_PARTICLES, DIM))
velocities = np.random.uniform(-1, 1, (NUM_PARTICLES, DIM))

pbest_positions = positions.copy()
pbest_values = np.array([rastrigin(p) for p in positions])

gbest_index = np.argmin(pbest_values)
gbest_position = pbest_positions[gbest_index].copy()
gbest_value = pbest_values[gbest_index]

history = [gbest_value]

# ---------------------------------------------------------
# 4️⃣ Loop principal do PSO
# ---------------------------------------------------------
for iteration in range(ITERATIONS):

    for i in range(NUM_PARTICLES):

        r1 = np.random.rand(DIM)
        r2 = np.random.rand(DIM)

        # Atualiza velocidade
        velocities[i] = (
            w * velocities[i]
            + c1 * r1 * (pbest_positions[i] - positions[i])
            + c2 * r2 * (gbest_position - positions[i])
        )

        # Atualiza posição
        positions[i] = positions[i] + velocities[i]

        # Mantém dentro dos limites
        positions[i] = np.clip(positions[i], LOWER_BOUND, UPPER_BOUND)

        # Avalia nova posição
        fitness = rastrigin(positions[i])

        # Atualiza pbest
        if fitness < pbest_values[i]:
            pbest_values[i] = fitness
            pbest_positions[i] = positions[i].copy()

    # Atualiza gbest
    gbest_index = np.argmin(pbest_values)
    if pbest_values[gbest_index] < gbest_value:
        gbest_value = pbest_values[gbest_index]
        gbest_position = pbest_positions[gbest_index].copy()

    history.append(gbest_value)

    if iteration % 10 == 0:
        print(f"Iteração {iteration} | Melhor fitness: {gbest_value:.6f}")

# ---------------------------------------------------------
# 5️⃣ Resultado
# ---------------------------------------------------------
print("\nMelhor solução encontrada:")
print("Fitness:", gbest_value)
print("Primeiras componentes:", gbest_position[:5])

# Gráfico de convergência
plt.plot(history)
plt.title("PSO — Convergência")
plt.xlabel("Iteração")
plt.ylabel("Melhor Fitness")
plt.grid(True)
plt.show()