import numpy as np

class PufferFishOptimization:
    def __init__(self, population_size=10, num_generations=20, learning_rate_bounds=(1e-5, 1e-2),
                 batch_size_options=[32, 64, 128, 256], dropout_bounds=(0.1, 0.5), lstm_units_bounds=(32, 128)):
        self.population_size = population_size
        self.num_generations = num_generations
        self.learning_rate_bounds = learning_rate_bounds
        self.batch_size_options = batch_size_options
        self.dropout_bounds = dropout_bounds
        self.lstm_units_bounds = lstm_units_bounds

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = {
                "learning_rate": np.random.uniform(*self.learning_rate_bounds),
                "batch_size": np.random.choice(self.batch_size_options),
                "dropout_rate": np.random.uniform(*self.dropout_bounds),
                "lstm_units": np.random.randint(*self.lstm_units_bounds)
            }
            population.append(individual)
        return population

    def fitness_function(self, hyperparams):
        # Simulating an accuracy score for demonstration purposes
        accuracy = np.random.uniform(0.7, 0.95)  # Assume a range of model accuracy
        return accuracy

    def update_population(self, population):
        new_population = []
        best_individual = max(population, key=lambda x: x["fitness"])

        for individual in population:
            new_individual = individual.copy()

            # Pufferfish-inspired defense mechanism (small changes in hyperparameters)
            new_individual["learning_rate"] += np.random.uniform(-0.0005, 0.0005)
            new_individual["learning_rate"] = np.clip(new_individual["learning_rate"], *self.learning_rate_bounds)

            new_individual["dropout_rate"] += np.random.uniform(-0.05, 0.05)
            new_individual["dropout_rate"] = np.clip(new_individual["dropout_rate"], *self.dropout_bounds)

            new_individual["lstm_units"] += np.random.randint(-8, 8)
            new_individual["lstm_units"] = np.clip(new_individual["lstm_units"], *self.lstm_units_bounds)

            new_population.append(new_individual)

        return new_population

    def optimize(self):
        population = self.initialize_population()

        for generation in range(self.num_generations):
            for individual in population:
                individual["fitness"] = self.fitness_function(individual)

            population = self.update_population(population)

            best_individual = max(population, key=lambda x: x["fitness"])
            print(f"Generation {generation+1}/{self.num_generations} - Best Accuracy: {best_individual['fitness']:.4f}")

        return best_individual

if __name__ == "__main__":
    optimizer = PufferFishOptimization(population_size=10, num_generations=5)
    best_hyperparams = optimizer.optimize()
    print("Best Hyperparameters:", best_hyperparams)
