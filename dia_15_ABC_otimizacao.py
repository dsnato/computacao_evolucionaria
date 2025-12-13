import numpy as np
import random

# -------------------------------------------
# Função objetivo (mínimos)
# Exemplo: esfera (x^2 + y^2 + ...)
# -------------------------------------------
def esfera(x):
    return np.sum(x ** 2)

# -------------------------------------------
# Gera uma solução aleatória dentro do intervalo
# -------------------------------------------
def gerar_solucao(dim, minimo, maximo):
    return np.random.uniform(minimo, maximo, dim)

# -------------------------------------------
# Gera solução vizinha (movimento local)
# x_new = x + phi * (x - x_k)
# phi in [-1, 1]
# -------------------------------------------
def gerar_vizinho(x, populacao, minimo, maximo):
    dim = len(x)
    k = random.randint(0, len(populacao)-1)
    while np.array_equal(populacao[k], x):
        k = random.randint(0, len(populacao)-1)

    phi = np.random.uniform(-1, 1, dim)
    x_new = x + phi * (x - populacao[k])

    # Limitar ao intervalo
    x_new = np.clip(x_new, minimo, maximo)
    return x_new

# -------------------------------------------
# Algoritmo ABC
# -------------------------------------------
def ABC(
    funcao,
    num_fontes=20,
    limite=20,
    iteracoes=100,
    dim=2,
    minimo=-5,
    maximo=5
):

    # Inicialização
    populacao = [gerar_solucao(dim, minimo, maximo) for _ in range(num_fontes)]
    aptidoes = [funcao(s) for s in populacao]
    contador_sem_melhora = [0] * num_fontes

    melhor_solucao = populacao[np.argmin(aptidoes)]
    melhor_valor = min(aptidoes)

    # -----------------------------
    # Loop principal
    # -----------------------------
    for it in range(iteracoes):

        # --- Fase das Employed Bees ---
        for i in range(num_fontes):
            vizinho = gerar_vizinho(populacao[i], populacao, minimo, maximo)
            f_vizinho = funcao(vizinho)

            if f_vizinho < aptidoes[i]:
                populacao[i] = vizinho
                aptidoes[i] = f_vizinho
                contador_sem_melhora[i] = 0
            else:
                contador_sem_melhora[i] += 1

        # --- Fase das Onlooker Bees ---
        apt_inverse = 1 / (1 + np.array(aptidoes))
        probs = apt_inverse / apt_inverse.sum()

        for _ in range(num_fontes):
            i = np.random.choice(range(num_fontes), p=probs)
            vizinho = gerar_vizinho(populacao[i], populacao, minimo, maximo)
            f_vizinho = funcao(vizinho)

            if f_vizinho < aptidoes[i]:
                populacao[i] = vizinho
                aptidoes[i] = f_vizinho
                contador_sem_melhora[i] = 0
            else:
                contador_sem_melhora[i] += 1

        # --- Fase das Scout Bees ---
        for i in range(num_fontes):
            if contador_sem_melhora[i] >= limite:
                populacao[i] = gerar_solucao(dim, minimo, maximo)
                aptidoes[i] = funcao(populacao[i])
                contador_sem_melhora[i] = 0

        # Atualiza melhor solução global
        idx = np.argmin(aptidoes)
        if aptidoes[idx] < melhor_valor:
            melhor_valor = aptidoes[idx]
            melhor_solucao = populacao[idx]

        print(f"Iteração {it+1} | Melhor valor: {melhor_valor:.6f}")

    return melhor_solucao, melhor_valor

# ------------------------------
# Execução do script
# ------------------------------
if __name__ == "__main__":
    sol, valor = ABC(funcao=esfera)
    print("\nMelhor solução encontrada:", sol)
    print("Valor:", valor)

