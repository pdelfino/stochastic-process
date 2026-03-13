# Stochastic Processes -- Simulation Assignments

![The Cardsharps](https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Caravaggio_%28Michelangelo_Merisi%29_-_The_Cardsharps_-_Google_Art_Project.jpg/960px-Caravaggio_%28Michelangelo_Merisi%29_-_The_Cardsharps_-_Google_Art_Project.jpg)

*"The Cardsharps" (c. 1594) by Caravaggio — [Wikipedia](https://en.wikipedia.org/wiki/The_Cardsharps)*

Computational simulations of fundamental stochastic processes, developed for the "Stochastic Processes" course at EMAp/FGV (Escola de Matematica Aplicada, Fundacao Getulio Vargas).

## About

This repository contains seven programming exercises that bridge the gap between the theory of stochastic processes and computational experimentation. Each problem is solved both analytically and via Monte Carlo simulation, verifying that numerical results converge to theoretical values.

**Professor:** Dr. Yuri Saporito
**Semester:** 2019.2
**Collaborators:** Bruna Fistarol, Danillo Fiorenza
**Reference:** *Introduction to Stochastic Processes with R* by Robert Dobrow; lecture notes by Prof. Saporito

## Topics Covered

| Problem | Topic | Description |
|---------|-------|-------------|
| 1 | **Markov Chains -- Stationary Distributions** | Classifying transient/recurrent states and computing stationary distributions via simulation (10,000 runs) |
| 2 | **Markov Chains -- Ergodic Averages** | Estimating long-run time averages and verifying against the theoretical value via the ergodic theorem |
| 3 | **Martingales** | Simulating multiplicative martingale paths and demonstrating almost-sure convergence to zero |
| 4 | **Poisson Process** | Simulating Poisson arrivals via inter-arrival times and the order-statistics method; computing expected integrals |
| 5 | **Gaussian Process Regression** | Non-parametric regression with an RBF kernel, computing the posterior mean and 95% confidence intervals |
| 6 | **Brownian Motion** | Simulating sample paths and comparing the empirical distribution of the running maximum to its exact density |
| 7 | **Black-Scholes Option Pricing** | Monte Carlo estimation of European call option prices under geometric Brownian motion |

## Tech Stack

- Python 3.6.8
- NumPy, Matplotlib

## Repository Structure

```
questao-1.py .. questao-7.py   # One script per problem
problem-set.pdf                # Original assignment (in Portuguese)
diagrama.png                   # State transition diagram (Problem 1)
questao-*.png                  # Generated plots and figures
teorica-*.jpg                  # Handwritten analytical derivations
```

## How to Run

```bash
# Example: simulate Brownian motion paths
python questao-6.py

# Example: Black-Scholes Monte Carlo pricing
python questao-7.py
```

Note: Some simulations (especially Problem 1) run 10,000 iterations and may take a few moments to complete.
