import random
import numpy as np

random.seed(12344)
np.random.seed(12344)

class AntColony:
    def __init__(self, distances, n_ants, n_best, n_iterations, decay, alpha=1, beta=1,patience=100):
        self.distances = distances
        self.pheromone = np.ones(self.distances.shape) / len(distances)
        self.all_inds = range(len(distances))
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.history = []
        self.patience = patience
    def run(self):
        all_time_shortest_path = ("placeholder", np.inf)
        shortest_path = None
        no_improve_counter = 0
        for i in range(self.n_iterations):
            all_paths = self.gen_all_paths()
            self.spread_pheromone(all_paths, self.n_best)

            # Track the shortest path at this iteration
            shortest_path = min(all_paths, key=lambda x: x[1])
            self.history.append(shortest_path)  # Append shortest path to history

            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path
                no_improve_counter = 0  # reset gdy jest poprawa
            else:
                no_improve_counter += 1
            self.pheromone *= (1 - self.decay)
            if no_improve_counter >= self.patience:
                print(f"Early stopping at iteration {i + 1} (no improvement in {self.patience} iterations).")
                break
        return all_time_shortest_path

    def gen_path_dist(self, path):
        total_dist = 0
        for i in range(len(path)):
            total_dist += self.distances[path[i % len(path)]][path[(i + 1) % len(path)]]
        return total_dist

    def gen_all_paths(self):
        all_paths = []
        for i in range(self.n_ants):
            path = self.gen_path(0)
            all_paths.append((path, self.gen_path_dist(path)))
        return all_paths

    def gen_path(self, start):
        path = []
        visited = set()
        visited.add(start)
        prev = start
        for i in range(len(self.distances) - 1):
            move = self.pick_move(prev, visited)
            path.append(move)
            prev = move
            visited.add(move)
        # Ensure the last city connects back to the start city (forming a cycle)
        path.append(start)  # This ensures the loop is closed
        return path

    def pick_move(self, current, visited):
        unvisited = list(set(self.all_inds) - visited)
        pheromone = self.pheromone[current, unvisited]
        distances = self.distances[current, unvisited]

        with np.errstate(divide='ignore'):
            attractiveness = pheromone ** self.alpha * (1.0 / distances) ** self.beta
        attractiveness = np.nan_to_num(attractiveness, nan=0.0, posinf=0.0)

        if attractiveness.sum() == 0:
            probs = np.ones_like(attractiveness) / len(attractiveness)
        else:
            probs = attractiveness / attractiveness.sum()

        return np.random.choice(unvisited, p=probs)

    def spread_pheromone(self, all_paths, n_best):
        # Sort all paths based on distance
        sorted_paths = sorted(all_paths, key=lambda x: x[1])

        # Spread pheromone for the n_best paths
        for path, dist in sorted_paths[:n_best]:
            for i in range(len(path)):
                from_city = path[i]
                to_city = path[(i + 1) % len(path)]
                # Update pheromone levels
                self.pheromone[from_city][to_city] += 1.0 / dist
                self.pheromone[to_city][from_city] += 1.0 / dist