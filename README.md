# ACO_NMO
Inteligencja roju, czy też inteligencja rozproszona to pojęcie z zakresu sztucznej inteligencji. Wyjaśnia ona jak grupa prymitywnych agentów, działających według prostych reguł, jest zdolna do wykonywania skomplikowanych zadań i podejmowania decyzji dzięki współpracy i komunikacji między sobą. 
Algorytm mrówkowy (ACO), opracowany przez M. Dorigo w 1992 roku w pracy "Optimization, Learning and Natural Algorithms" na Politecnico di Milano, opiera się na sposobie budowania rozwiązań przez przemieszczające się "mrówki". Mrówki te przechodzą od węzła do węzła na grafie (np. od miasta do miasta), wybierając kolejne kroki na podstawie intensywności feromonów oraz lokalnych heurystyk na krawędziach. Każda mrówka (k) przechodzi przez graf, tworząc rozwiązanie (S_k). Przemieszczając się z węzła i do j, mrówka wybiera ścieżkę z prawdopodobieństwem  p_{ij}^k danym przez : 

p_{ij}^k=\frac{\left[\tau_{ij}\right]^\alpha\left[\eta_{ij}\right]^\beta}{\sum_{l\in N_i}{\left[\tau_{il}\right]^\alpha\left[\eta_{il}\right]^\beta}}

gdzie: 
- \tau_{ij}: intensywność feromonów na krawędzi (i, j), 
- \eta_{ij}: wartość heurystyczna (np. odwrotność odległości (d_{ij})), 
- \alpha i \beta: współczynniki wpływu feromonów (\alpha) i heurystyki (\beta), 
- N_i:  zbiór węzłów dostępnych z węzła (i)

Komunikacja między agentami odbywa się poprzez środowisko, czyli macierz feromonów, Po zbudowaniu rozwiązań przez wszystkie mrówki, feromony na krawędziach są aktualizowane według wzoru:

\tau_{ij}=\left(1-\rho\right)\tau_{ij}+\sum_{k=1}^{m}{\Delta\tau_{ij}^k}

gdzie: 
-  \rho:  współczynnik parowania feromonów, 
- \Delta\tau_{ij}^k: ilość feromonów dodanych przez mrówkę k, często zdefiniowana jako odwrotność długości trasy  S_k : 

\Delta\tau_{ij}^k = QLk             jeśli mrówka k przeszła przez i,j0                                            w przeciwnym razie 

gdzie: 
-  Q: stała, 
- \ L_k: długość trasy S_k.

Taki sposób działania prowadzi do wysokiego prawdopodobieństwa szybkiej konwergencji do optymalnych lub bliskich optymalnych rozwiązań. Jednak istnieje ryzyko utknięcia w lokalnych minimach. Algorytm ACO jest przystosowany do optymalizacji kombinatorycznej i jest skuteczny w problemach, które można przedstawić za pomocą grafu.
