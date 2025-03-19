# Particle-Swarm-Optimization

This project aims to visualize particle swarm optimization.
In this context, we are treating swarm of fish as particle and food as the destination.
This project depicts the use of simple particle swarm optimization using heuristic approach as well as the AI model approach.

Steps to run this project:

1. Git clone this project.
2. Install required dependencies in virtual environment after creating the virtual environment.
   Use Command python -m venv .venv
3. Run the file particle_swarm_optimization.py to see the particle swarm optimization visually without using the ai model.
4. Run new.py to create a dataset(.csv file) inorder to train the model which will be generated in same project directory.
5. The dataset generated is not in format to train the model so we flatten it using the file flatten_csv.py also the flattened file is generated inside same directory.
6. After flattening, run train.py in google colab or jupyter notebook as you wish.
   It takes some time depending on your processing power.
7. Train.py generates .joblib file which is our machine learning model which we put inside the project directory.
8. Now run the file pswarmoptusingmodel.py in order to visualize the particle swarm optimization using the trained ai model.
