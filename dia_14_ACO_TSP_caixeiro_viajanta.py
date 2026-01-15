import numpy as np
import random


# -------------------------------------------------------------
# Gerar cidades aleatórias (coordenadas 2D)
# -------------------------------------------------------------
def gerar_cidades(n=10, limite=100):
    return np.random.rand(n, 2) * limite


# --------------------------------------------------------------
# Função de distância Euclidiana
# --------------------------------------------------------------
def distancia(a, b):
    return np.linalg.norm(a - b)


# --------------------------------------------------------------
# Construir rota usando probabilidade baseada em
# feromônio ^ alpha  *  (1 / distância) ^ beta
# --------------------------------------------------------------
def construir_rota(feromonio, distancias, alpha=1.0, beta=2.0):
    n = len(distancias)
    visitados = [random.randint(0, n - 1)]

    for _ in range(n - 1):
        atual = visitados[-1]
        nao_visitados = [i for i in range(n) if i not in visitados]

        # Probabilidade proporcional ao feromônio e à heurística
        probs = []
        for j in nao_visitados:
            tau = feromonio[atual][j] ** alpha
            eta = (1 / distancias[atual][j]) ** beta
            probs.append(tau * eta)

        probs = np.array(probs)
        probs = probs / probs.sum()

        # Escolher próxima cidade conforme distribuição de probabilidade
        proxima = np.random.choice(nao_visitados, p=probs)
        visitados.append(proxima)

    return visitados


# --------------------------------------------------------------
# Calcular o tamanho total da rota
# --------------------------------------------------------------
def calcular_custo(rota, distancias):
    total = 0
    for i in range(len(rota)):
        total += distancias[rota[i]][rota[(i + 1) % len(rota)]]
    return total


# --------------------------------------------------------------
# Atualizar feromônio com evaporação + reforço por boas rotas
# --------------------------------------------------------------
def atualizar_feromonio(feromonio, rotas, custos, evaporacao=0.5, Q=100):
    n = len(feromonio)
    feromonio *= (1 - evaporacao)

    for rota, custo in zip(rotas, custos):
        deposito = Q / custo
        for i in range(len(rota)):
            a = rota[i]
            b = rota[(i + 1) % len(rota)]
            feromonio[a][b] += deposito
            feromonio[b][a] += deposito

    return feromonio


# --------------------------------------------------------------
# Algoritmo principal ACO
# --------------------------------------------------------------
def aco_tsp(num_cidades=10, num_formigas=20, iteracoes=50):
    cidades = gerar_cidades(num_cidades)

    # Matriz de distâncias
    distancias = np.zeros((num_cidades, num_cidades))
    for i in range(num_cidades):
        for j in range(num_cidades):
            distancias[i][j] = distancia(cidades[i], cidades[j])

    # Feromônio inicial
    feromonio = np.ones((num_cidades, num_cidades))

    melhor_custo = float("inf")
    melhor_rota = None

    for it in range(iteracoes):
        rotas = []
        custos = []

        for _ in range(num_formigas):
            rota = construir_rota(feromonio, distancias)
            custo = calcular_custo(rota, distancias)
            rotas.append(rota)
            custos.append(custo)

            if custo < melhor_custo:
                melhor_custo = custo
                melhor_rota = rota

        feromonio = atualizar_feromonio(feromonio, rotas, custos)

        print(f"Iteração {it + 1} | Melhor custo até agora = {melhor_custo:.2f}")

    return melhor_rota, melhor_custo


# --------------------------------------------------------------
# Executar quando rodar o arquivo
# --------------------------------------------------------------
if __name__ == "__main__":
    rota, custo = aco_tsp()
    print("\nMelhor rota encontrada:", rota)
    print("Custo total:", custo)

