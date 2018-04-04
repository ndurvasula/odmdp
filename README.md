# odmdp
## What is on-demand reinforcement learning?
Current reinforcement learning (RL) methods, while model-free, still require an agent to walk through the Markov Decision Process (MDP) many times. Thus, the MDP state transition model must be known or completely simulatable. However, in many real-world contexts (such as performing business negotations or interacting with people in general), we do not know the model, and thus cannot train with it. 

In an **on-demand** RL problem, we aim to approximate the optimal policy while only walking through the MDP a _single_ time.

## What is this code?
This code implements a general method for approximating the policy in the on-demand environment, applicable to _any_ RL problem. Without going into much detail, it works by approximating the MDP state transition model given state transitions that we have seen thus far, and calls a subsolver that finds the optimal policy given the approximated model.

## Usage
To approximate the optimal policy, import **solver.py** into your project. Calling `Solver.step()` will return the next action. After observing the state transition, update the solver using `Solver.update(parts)`.

To use this code, all application specific information (specified in the code stub provided) must be placed in **application.py**. This includes a subsolver that can find the optimal policy _given_ the model. The user specified `application.subsolver(transition)` is passed the function `transition(state,action)` which samples from the approximated state transition function.

In `/examples`, you can find example code for toy on-demand RL problems.

### Dependencies
To use this code, you'll need **numpy** and **GPy**, both of which can be installed through **pip**.
