import random
import math
import copy
import matplotlib.pyplot as plt
from typing import Callable, Union, List, Tuple
import sys # Import the sys module

random.seed(42)

# Increase the recursion limit to handle potentially deep trees
sys.setrecursionlimit(2000)

# -------------------------------------------
# Definição do espaço de funções e terminais
# -------------------------------------------
# Cada função é um tuple (callable, aridade, símbolo)
FUNCTION_SET = [
    (lambda a, b: a + b, 2, "+"),
    (lambda a, b: a - b, 2, "-"),
    (lambda a, b: a * b, 2, "*"),
    # divisão protegida: se divisor próximo de 0, retorna 1.0
    (lambda a, b: a / b if abs(b) > 1e-6 else 1.0, 2, "/"),
    (lambda a: math.sin(a), 1, "sin"),
    (lambda a: math.cos(a), 1, "cos"),
    (lambda a: math.exp(a) if a < 50 else math.exp(50), 1, "exp"),  # evita overflow
]

TERMINAL_SET = ["x"]  # variável
# constantes podem ser terminais também; usamos constantes randômicas ao gerar folhas
CONST_MIN, CONST_MAX = -5.0, 5.0

# -------------------------
# Estrutura de nó da árvore
# -------------------------
class Node:
    """
    Representa um nó em uma árvore de expressão.
    - Se node_type == 'func', aridade indica quantos filhos e func é a callable.
    - Se node_type == 'term', value guarda 'x' ou um número (constante).
    """

    def __init__(self,
                 node_type: str,
                 func: Callable = None,
                 arity: int = 0,
                 symbol: str = "",
                 value: Union[float, str] = None):
        self.node_type = node_type  # 'func' ou 'term'
        self.func = func            # função (para 'func')
        self.arity = arity          # número de filhos (para 'func')
        self.symbol = symbol        # representação textual (por ex. '+', 'sin')
        self.value = value          # valor do terminal (por ex. 'x' ou 3.14)
        self.children: List[Node] = []  # lista de nós filhos

    def is_terminal(self) -> bool:
        return self.node_type == "term"

    def copy(self):
        """Retorna uma cópia profunda do nó (subárvore)."""
        return copy.deepcopy(self)

    def __str__(self) -> str:
        """Representação em infix (legível) da subárvore neste nó."""
        if self.is_terminal():
            return str(self.value)
        # função unária
        if self.arity == 1:
            return f"{self.symbol}({str(self.children[0])})"
        # função binária (infix)
        left = str(self.children[0])
        right = str(self.children[1])
        return f"({left} {self.symbol} {right})"

# -------------------------
# Geração de árvores (crescimento randômico)
# -------------------------
def generate_random_tree(max_depth: int, grow: bool = True) -> Node:
    """
    Gera uma árvore aleatória.
    - max_depth: profundidade máxima permitida (1 = apenas folha)
    - grow=True: mistura funções e terminais até a profundidade; grow=False: full (funcional até penúltimo nível)
    """
    # Se chegamos na profundidade 1, precisamos gerar um terminal
    if max_depth == 1:
        return generate_random_terminal()

    # Decidir se criaremos função ou terminal (para variar forma)
    if grow:
        # maior probabilidade de terminais conforme profundidade decresce
        p_term = 0.3
        if random.random() < p_term:
            return generate_random_terminal()
    else:
        # full: força função até profundidade-1
        pass

    # cria função aleatória
    func, arity, symbol = random.choice(FUNCTION_SET)
    node = Node(node_type="func", func=func, arity=arity, symbol=symbol)
    for _ in range(arity):
        node.children.append(generate_random_tree(max_depth - 1, grow))
    return node

def generate_random_terminal() -> Node:
    """Gera um terminal: 'x' ou uma constante aleatória."""
    if random.random() < 0.6:
        return Node(node_type="term", value="x")
    else:
        val = random.uniform(CONST_MIN, CONST_MAX)
        return Node(node_type="term", value=round(val, 4))

# -------------------------
# Avaliação de árvore para um valor x (execução da expressão)
# -------------------------
def evaluate_tree(node: Node, x_value: float) -> float:
    """
    Avalia recursivamente a árvore para x = x_value.
    - Para terminais 'x' retorna x_value
    - Para constantes retorna o número
    - Para funções aplica func nos valores avaliados dos filhos
    """
    if node.is_terminal():
        if node.value == "x":
            return x_value
        else:
            return float(node.value)
    # calcula valores dos filhos
    child_vals = [evaluate_tree(child, x_value) for child in node.children]
    try:
        # chama a função armazenada com unpack dos argumentos
        return node.func(*child_vals)
    except Exception:
        # caso algum erro numérico ocorra, penalizamos retornando grande valor
        return float("inf")

# -------------------------
# Geração de população inicial
# -------------------------
def initialize_population(pop_size: int, max_depth: int) -> List[Node]:
    """Cria lista de árvores (população inicial) usando método grow/full aleatório."""
    population = []
    for i in range(pop_size):
        # misturamos grow e full para diversidade (50% cada)
        grow = (random.random() < 0.5)
        tree = generate_random_tree(max_depth=max_depth, grow=grow)
        population.append(tree)
    return population

# -------------------------
# Fitness: MSE (quanto menor, melhor)
# -------------------------
def fitness(tree: Node, dataset: List[Tuple[float, float]]) -> float:
    """
    Calcula o erro quadrático médio entre predição da árvore e valores reais.
    Retorna um valor >= 0 (quanto menor, melhor).
    """
    errors = []
    for x, y_true in dataset:
        y_pred = evaluate_tree(tree, x)
        # se avaliação falhou (inf), retorna valor alto
        if not math.isfinite(y_pred):
            return 1e6
        errors.append((y_true - y_pred) ** 2)
    return sum(errors) / len(errors)

# -------------------------
# Seleção: torneio
# -------------------------
def tournament_selection(population: List[Node], dataset: List[Tuple[float, float]], k: int) -> Node:
    """Seleciona o melhor entre k amostras aleatórias usando fitness (menor é melhor)."""
    candidates = random.sample(population, k)
    candidates_sorted = sorted(candidates, key=lambda t: fitness(t, dataset))
    return candidates_sorted[0].copy()  # retorna cópia para evitar aliasing

# -------------------------
# Operadores genéticos: crossover (troca de subárvore) e mutação (substituição)
# -------------------------
def get_all_nodes(node: Node) -> List[Node]:
    """
    Retorna lista de todas as referências de nós (preorder).
    Usado para escolher um ponto de crossover ou mutação.
    """
    nodes = [node]
    if not node.is_terminal():
        for child in node.children:
            nodes.extend(get_all_nodes(child))
    return nodes

def replace_node(parent: Node, target: Node, replacement: Node) -> bool:
    """
    Substitui a subárvore 'target' em 'parent' por 'replacement'.
    Retorna True se substituiu; False caso não encontrasse (usado para navegar a árvore).
    """
    if parent.is_terminal():
        return False
    for i, child in enumerate(parent.children):
        if child is target:
            parent.children[i] = replacement
            return True
        else:
            if replace_node(child, target, replacement):
                return True
    return False

def subtree_crossover(parent1: Node, parent2: Node, max_depth: int) -> Tuple[Node, Node]:
    """
    Seleciona um nó aleatório em cada pai e troca as subárvores.
    Retorna dois filhos (cópias dos pais modificados).
    """
    # cópias para não modificar pais originais
    p1_copy = parent1.copy()
    p2_copy = parent2.copy()

    nodes1 = get_all_nodes(p1_copy)
    nodes2 = get_all_nodes(p2_copy)

    # evitamos trocar a raiz em ambos para preservar diversidade (mas é permitido)
    node1 = random.choice(nodes1)
    node2 = random.choice(nodes2)

    # se o nó selecionado for a raiz, simplesmente troca árvores inteiras
    if node1 is p1_copy:
        child1 = node2.copy()
    else:
        child1 = p1_copy
        replace_node(child1, node1, node2.copy())

    if node2 is p2_copy:
        child2 = node1.copy()
    else:
        child2 = p2_copy
        replace_node(child2, node2, node1.copy())

    # opcional: controlar profundidade - se exceder, pode cortar (não implementado rigorosamente aqui)
    return child1, child2

def subtree_mutation(tree: Node, max_depth: int) -> Node:
    """
    Substitui uma subárvore aleatória por uma nova árvore gerada aleatoriamente.
    """
    t_copy = tree.copy()
    nodes = get_all_nodes(t_copy)
    node_to_replace = random.choice(nodes)
    new_subtree = generate_random_tree(max_depth=max_depth, grow=True)

    if node_to_replace is t_copy:
        return new_subtree
    else:
        replace_node(t_copy, node_to_replace, new_subtree)
        return t_copy

# -------------------------
# Função utilitária: imprime expressão em string limpa
# -------------------------
def tree_to_string(tree: Node) -> str:
    return str(tree)

# -------------------------
# Algoritmo GP principal
# -------------------------
def genetic_programming(
    pop_size: int,
    generations: int,
    dataset: List[Tuple[float, float]],
    max_depth: int = 5,
    tournament_k: int = 3,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.3,
    elite_size: int = 1
) -> Tuple[Node, List[float]]:
    """
    Executa o loop evolutivo do GP:
    - inicializa população
    - avalia fitness
    - preserva elites
    - aplica seleção, crossover e mutação
    Retorna o melhor indivíduo e histórico (melhor fitness por geração).
    """
    population = initialize_population(pop_size, max_depth)
    history = []

    for gen in range(generations):
        # calcula fitness de toda população
        pop_fitness = [(ind, fitness(ind, dataset)) for ind in population]
        pop_fitness.sort(key=lambda x: x[1])  # ordena por fitness crescente (menor é melhor)

        # registra melhor
        best_ind, best_fit = pop_fitness[0]
        history.append(best_fit)

        # imprime resumo
        if gen % 5 == 0 or gen == generations - 1:
            print(f"Geração {gen:03d} | Melhor MSE = {best_fit:.6f} | Expr = {tree_to_string(best_ind)}")

        # elitismo: preserva top N
        new_population = [ind.copy() for ind, _ in pop_fitness[:elite_size]]

        # preenche restante da população
        while len(new_population) < pop_size:
            # seleção
            parent1 = tournament_selection(population, dataset, tournament_k)
            parent2 = tournament_selection(population, dataset, tournament_k)

            # reprodução
            if random.random() < crossover_rate:
                child1, child2 = subtree_crossover(parent1, parent2, max_depth)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # mutação
            if random.random() < mutation_rate:
                child1 = subtree_mutation(child1, max_depth)
            if random.random() < mutation_rate and len(new_population) + 1 < pop_size:
                child2 = subtree_mutation(child2, max_depth)

            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)

        population = new_population

    # ao final, retorna melhor indivíduo (reavaliado)
    final_pop_fitness = [(ind, fitness(ind, dataset)) for ind in population]
    final_pop_fitness.sort(key=lambda x: x[1])
    best_individual, best_fit = final_pop_fitness[0]
    return best_individual, history

# -------------------------
# Exemplo de dataset sintético (regressão simbólica)
# -------------------------
def make_dataset(func: Callable[[float], float], n_samples: int = 40, x_min: float = -1.0, x_max: float = 1.0):
    xs = [random.uniform(x_min, x_max) for _ in range(n_samples)]
    xs.sort()
    return [(x, func(x)) for x in xs]

# função alvo para testar (não trivial, multimodal):
def target_function(x):
    # combinação de termos simples para criar curva não-trivial
    return x * math.sin(3 * x) + 0.5 * x

# ------------------------
# Execução (script)
# ------------------------
if __name__ == "__main__":
    # gera dataset
    dataset = make_dataset(target_function, n_samples=50, x_min=-2.0, x_max=2.0)

    # parâmetros do GP
    POP_SIZE = 200
    GENERATIONS = 60
    MAX_DEPTH = 5

    best_tree, hist = genetic_programming(
        pop_size=POP_SIZE,
        generations=GENERATIONS,
        dataset=dataset,
        max_depth=MAX_DEPTH,
        tournament_k=5,
        crossover_rate=0.9,
        mutation_rate=0.35,
        elite_size=2
    )

    # imprime resultado final
    print("\n=== Melhor indivíduo final ===")
    print("Expressão:", tree_to_string(best_tree))
    print("MSE final:", fitness(best_tree, dataset))

    # plota histórico de fitness (MSE)
    plt.figure(figsize=(10, 4))
    plt.plot(hist)
    plt.title("Convergência do GP (MSE)")
    plt.xlabel("Geração")
    plt.ylabel("MSE (menor é melhor)")
    plt.grid(True)
    plt.show()

    # plota predição vs real
    xs = [x for x, _ in dataset]
    ys_true = [y for _, y in dataset]
    ys_pred = [evaluate_tree(best_tree, x) for x in xs]

    plt.figure(figsize=(8, 5))
    plt.scatter(xs, ys_true, label="Real (target)", s=20)
    plt.plot(xs, ys_pred, label="Predição (melhor GP)", linewidth=2)
    plt.title("Predição vs Real — GP (melhor indivíduo)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)

    plt.show()

