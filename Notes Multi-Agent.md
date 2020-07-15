# Multi-Agent DRL

Molt del RL es concentra en un agent que ha de demostrar competència en una tasca. En aquest escenari no hi ha altres agents.
Tot i això, si volem que els nostres agents siguin inteligents, haurien de poder comunicar-se amb, i aprendre de, altres agents. Multi-agent RL té moltes aplicacions en el mon real, des de cotxes autònoms a gestió de magatzems.

## Introducció al Multi-agent RL

[Video]()

**Motivació**:
- vivim en un món amb multi agents
- els agents intel·ligents han d'interactuar amb humants
- els agents necessiten treballar en entorns complexes

**Beneficis**:
- els agents poden compartir les experiències entre ells
- els agents es poden substituir
- escalabilitat (tot i què, com més agents, més complexe)

**MARL**: Multi Agent Reinforcement Learning

## Markov Games
joint set of actions

Markov Game Framework, generalització de MDP a multi agent


\begin{align}
&<n,S,A_1,\ldots,A_n,O_1,\ldots,O_n,R_1,\ldots,R_n,\pi_i,\ldots,\pi_n,T>\\
&\text{$n$: número d'agents}  \\
&\text{$S$: conjunt d'estats de l'entorn} \\ 
&\text{$R_i$: $S \times A_i \rightarrow R$} \\
&\text{$\pi_i$: $O_i \rightarrow A_i$} \\
&\text{$T$: $S \times A \rightarrow S$ } \\
\end{align}

La funció de transició depèn d'aquestes accions juntes (joint actions)

## Aproximacions a MARL

2 aproximacions:
- La simple, entrenar els agents de forma independent, sense considerar l'existència d'altres agents. En aquesta aproximació, l'agent considera els altres com a part de l'entorn, i apren la seva política. Com cada un apren independenment, l'entorn es veu de forma prospectiva i canvia dinàmicament. Entorn (non-stationary). En molts algoritmes s'assumeix que l'entorn es estacionari (stationary) i es pot garantir la convergència. En un entorn no estacionari, això no es pot garantir.
- meta-agent. Considera l'existència de múltiples agents, i es genera una única política. Pren l'estat i l'acció de l'agent es la *joint action* de totes les accions $\text{Policy: } S \rightarrow A_1 \times A_2 \ldots \times A_n$, i la recompensa que entrega l'entorn es global: $R: S \times A \rightarrow \text{Real Number}$. Aquesta aproximació només funciona quan els agents poden veure tot l'entorn (no POMP)

## Cooperació, Competició, Entorns mixtos


## Research topics
OpenAI 5

## Descripció del paper que usarem
En aquesta lliçó ens centrarem en aquest paper [Multi Agent Actor Critic for Mixed Cooperative Competitive environments](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf)

Provarem d'implementar part del paper:

- OpenAI multi-agent escenari. physical deception

# Classe 3. Case Study: AlphaZero

1. AlphaGo
2. AlphaZero. Es simple: Monte Carlo tree search, guiada per una DNN.

Papers: [alphago zero](https://deepmind.com/documents/119/agz_unformatted_nature.pdf) i [alphazero](https://arxiv.org/abs/1712.01815)

### AlphaZero

Zero-Sum Game. 

TicTacToe