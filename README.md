# 🧬 30 Dias de Computação Evolucionária

Este repositório contém **30 exercícios práticos** sobre **Computação Evolucionária**, abrangendo algoritmos inspirados em processos naturais de evolução, adaptação e comportamento coletivo.  
O objetivo é **aprender e implementar do zero** os principais paradigmas dessa área, com **códigos comentados linha a linha**, **explicações teóricas** e **exemplos executáveis**.

---

## 🎯 Objetivos do Projeto

- Entender os fundamentos teóricos da **Computação Evolucionária** e seus subcampos.  
- Implementar **30 algoritmos evolutivos**, cada um em um arquivo independente (`dia_X_nome_do_algoritmo.py`).  
- Comentar detalhadamente cada linha e função, explicando:  
  - O papel de cada operador (seleção, cruzamento, mutação etc.);
  - As estruturas de dados utilizadas;
  - O comportamento esperado e os resultados obtidos.  
- Executar cada código com **exemplos simples e reprodutíveis** (datasets sintéticos, funções matemáticas, listas aleatórias etc.).  
- Comparar abordagens, performances e aplicações práticas.  

---

## 📅 Cronograma

| Dia | Tema Principal | Tipo de Algoritmo | Objetivo |
|-----|----------------|------------------|-----------|
| 1 | Introdução e Estrutura Base | — | Criar o template base de um algoritmo evolucionário |
| 2 | Algoritmo Genético Clássico | GA | Maximizar função simples |
| 3 | Seleção por Roleta | GA | Comparar tipos de seleção |
| 4 | Seleção por Torneio | GA | Ajustar pressão seletiva |
| 5 | Cruzamento de 1 ponto | GA | Recombinar soluções |
| 6 | Mutação Aleatória | GA | Introduzir diversidade |
| 7 | Função de Fitness em problemas contínuos | GA | Otimizar função matemática |
| 8 | Algoritmo Genético Binário | GA | Representação de bits |
| 9 | Algoritmo Genético com Elitismo | GA | Preservar melhores soluções |
| 10 | Programação Genética | GP | Evoluir árvores de expressão |
| 11 | Algoritmos Meméticos | GA + Local Search | Híbrido evolutivo |
| 12 | Algoritmo de Estratégia Evolutiva (ES) | ES | Otimização contínua |
| 13 | Algoritmo de Evolução Diferencial | DE | Minimizar função multimodal |
| 14 | Colônia de Formigas | ACO | Resolver problema do caixeiro viajante |
| 15 | Colônia de Abelhas | ABC | Otimização inspirada em abelhas |
| 16 | Algoritmo do Enxame de Partículas | PSO | Otimização contínua |
| 17 | Algoritmo de Bat | Metaheurística Bioinspirada | Explorar soluções |
| 18 | Algoritmo de Cuckoo Search | Metaheurística Bioinspirada | Usar comportamento de parasitismo |
| 19 | Algoritmo de Fogo | Metaheurística Física | Minimizar função de energia |
| 20 | Algoritmo de Simulated Annealing | SA | Busca local com resfriamento |
| 21 | Algoritmo de Evolução Cultural | CE | Aprendizagem de população |
| 22 | Algoritmo de Coevolução | GA | Competição entre espécies |
| 23 | Algoritmo Multiobjetivo (NSGA-II) | MOEA | Otimização com múltiplos objetivos |
| 24 | Programação Evolutiva | EP | Evoluir parâmetros |
| 25 | Island Model GA | GA Distribuído | Migração entre populações |
| 26 | Algoritmo Genético Paralelo | GA Distribuído | Dividir e conquistar |
| 27 | Evolução com Redes Neurais (NEAT) | Neuroevolução | Evoluir topologias |
| 28 | Evolução de Autômatos Celulares | EC | Regras emergentes |
| 29 | Hibridização com Deep Learning | GA + DL | Otimizar hiperparâmetros |
| 30 | Projeto Final | — | Resolver um problema real com abordagem evolucionária |

---

## 🧠 Conceitos-Chave Abordados

- **Seleção natural e adaptação**
- **Operadores genéticos (cruzamento, mutação, elitismo)**
- **População e fitness**
- **Exploração vs Exploração**
- **Metaheurísticas bioinspiradas**
- **Otimização contínua e combinatória**
- **Evolução de funções e estruturas de código (Programação Genética)**
- **Integração com Aprendizado de Máquina**

---

## 🧰 Tecnologias Utilizadas

- **Python 3.11+**
- **NumPy**, **Matplotlib**, **random**, **math**
- Eventualmente: **DEAP**, **SciPy**, **Pandas**, **Seaborn**
- Todos os códigos serão **autossuficientes** e **reprodutíveis**

---

## 🗂️ Estrutura do Repositório

```bash
📦 30-dias-computacao-evolucionaria
├── README.md
├── dia_01_base_algoritmo_evolucionario.py
├── dia_02_GA_classico_maximizar_funcao.py
├── dia_03_GA_selecao_roleta.py
│
├── ...
│
└── dia_30_projeto_final.py
```

---

## 📖 Referências Clássicas

1. **Eiben, A. E., & Smith, J. E. (2015).** *Introduction to Evolutionary Computing*. Springer.  
2. **Mitchell, M. (1998).** *An Introduction to Genetic Algorithms*. MIT Press.  
3. **Holland, J. H. (1975).** *Adaptation in Natural and Artificial Systems*. University of Michigan Press.  
4. **Dorigo, M., & Stützle, T. (2004).** *Ant Colony Optimization*. MIT Press.  
5. **Kennedy, J., & Eberhart, R. (1995).** *Particle Swarm Optimization*. IEEE.  
6. **Simon, D. (2013).** *Evolutionary Optimization Algorithms*. Wiley.

---

## 🧩 Como Executar

1. Clone o repositório:
   ```bash
   git clone https://github.com/ds_nato/computacao_evolucionaria.git
   cd computacao_evolucionaria
   
2. Crie um ambiente virtual e instale as dependências:
   ```python -m venv venv
   source venv/bin/activate  # ou venv\Scripts\activate no Windows
   pip install -r requirements.txt
   ```

3. Execute o código do dia:
   ```bash
   python dia_05_mutacao.py
   ```

4. Observe a saída e gráficos de convergência para análise do desempenho.

---

## 🧪 Licença

Este projeto é distribuído sob a licença **MIT** — veja o arquivo `LICENSE` para mais detalhes.

---

## ✍️ Autor

**Renato Samico**  
Estudante de **Sistemas de Informação** e **Ciência da Informação** | Pesquisador em **IA e Computação Evolucionária**  
👨‍💻 Foco atual: *Problemas de Roteamento, Algoritmos Genéticos e Modelos de Linguagem de Grande Escala (LLMs)*  

---

> “A evolução é o algoritmo mais poderoso do universo — e a Computação Evolucionária é a forma de reproduzi-lo digitalmente.”  
> — *John H. Holland*

