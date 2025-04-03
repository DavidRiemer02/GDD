from ganTraining import GAN

# Path to your real dataset
data_path = "TrainingData/realData/iris_data.csv"

# Initialize and Train the GAN
gan_model = GAN(data_path)
# Train the GAN model
gan_model.train()
  

# Generate Synthetic Data
gan_model.generate_data(num_samples=10000)  