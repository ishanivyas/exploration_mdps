# exploration_mdps

# Objective:
Compare the relative performance of different exploration strategies in a Markov Decision
Process using a simple simulated environment. Measures of performance include total reward
and distance between empirical estimates and optimal values after some number of iterations.
If time remains, it would be useful to extend the environment to see how the exploration
strategies scale with increases in the dimensionality of the environment.


# Outline:
 ● Select an MDP & a reward function
    ○ Gridworld
    ○ Extend Gridworld
       ■ Simplify to hexagonal grid (instead of 8 adjacent cells, there will be only 6)
       ■ Vary/extend the range of grid height values
■ 3D (i.e. Cubeworld, 26 adjacent cells)
■ Add more types of tiles in gridworld (i.e. hazards)
● Quicksand
● Water
● Implement RL Exploration Strategies:
○ Random
○ Greedy
○ Epsilon-Greedy
○ Boltzmann
● Measure RL strategies over the course of the roll-outs
○ Total reward (may not be possible if rewards are given at the end)
○ Difference between empirical values and optimal values
○ Number of steps to achieve goal
○ Convergence rate of trained expert (reward vs training epoch)
