### **agent.py**

It contains three distinct agents, which are `HumanAgent`, `RandomAgent`, `LearningAgent`. Also, in my version, I declared an abstract class called `BaseAgent`. This class has a single abstract method which is named as `action()`. This method is designed to be passed current state as a `numpy` array and to return the action of agent whose type is `Tuple[int, int]`. 

When `HumanAgent.action()` is called, the user is to select his or her choice of coordinate, and the method returns it. When `RandomAgent.action()` is called, using python `random` module, it returns randomly chosen coordinate as its action. Finally, when `LearningAgent.action()` is called, which was initialized with `state_dict`, it returns the most-proficient action calculated by the Deep Q Learning algorithm, which is named to be `QNet`. That is, the method works as a policy function. It gets the current state as a matrix, computes the q-values of each cell, and returns the `argmax` value of the computed matrix. More details are described in `network.py`. 

### game.py

This script is only for the game environment. The entire information for the environment is contained in the class which is called `TicTacToe`. This class has some public methods as follows. 

- `reset(opponent: BaseAgent, turn: bool)`: This method resets the environment. Set the board to be an empty state and set `opponent` for self-playing sake. If `turn` is false, do an action by the opponent.
- `step(action: Tuple[int, int])`: This method is almost the same as the `step()` method is ***openai*** `gym`. Both of the argument and the return value are the exactly the same. That is, type of action, which is an argument, is `Tuple[int, int]` and it implies the coordinate. Also, the return value is a Tuple-value as well, which contains `next_state`, `reward`, `done`, and `info`. The working mechanism is as follows.
    
    ```mermaid
    flowchart TD
    	1(function called) --> 2(if the action is within the bound) -->|true| 3(if it's possible to place on)
    	2 -.->|false| 4[return: copy, -1, True, _]
    	3 -->|true| 5(place on the board at the coord)
    	3 -->|false| 4
    	5 --> 6(if there's a winner)
    	6 -->|true| 7[return: copy, 1, True, _]
    	6 -->|false| 8(if there's no place to put on)
    	8 -->|true| 9[return: copy, 0, True, _]
    	8 -->|false| 10[get opponent's action with parameter passed with a negative sign]
    	10 -->|repeat with negative sign in front of `win` and `player` when returning and placing, respectively| 1
    	10 --> 11(Finally, return: copy, 0, False, _ => it means the placing has undergone successfully)
    	
    ```
    
- `render()`: It shows the current state of the environment. This method helps to print the state in a beautiful manner. It shows cells within grid-shaped board.

### network.py

This script contains two classes and three functions. Let’s get dive into the class first. 

**classes**

- `ReplayBuffer`: It uses a queue. It’s not necessarily be a queue, it’s totally fine to use a list instead. But by using a queue for data structure, we can set the maximum length, which is a capacity of the queue. If it reaches its limit, following the fundamental property of queue, FIFO, the transition which was inserted first will be removed. This class contains two methods. `put(transition: Tuple[np.ndarray, int, int, np.ndarray, float])` and `sample(n: int)`.
    - `put(transition: Tuple[np.ndarray, int, int, np.ndarray, float])`: By using this method, we can put a transition data into the buffer. The buffer stores the transition data in the queue in the form of raw tuple.
    - `sample(n: int)`: This method is for a sampling. This method returns a tuple of tensors which consists of state, action, reward, next state, and done mask. Also, when returning the tensor of action, reward, and done mask, because they are in the form of a list, they are unsqueezed. That is, The shape of the tensors is `(n, 1)`, which is two-dimensional, not `(n, )`. The shape of each return value is as follows.
        - state: `(n, 18)`
        - action: `(n, 1)`
        - reward: `(n, 1)`
        - next state: `(n, 18)`
        - done mask: `(n, 1)`
- `QNet`: It inherits from `torch.nn.Module`, so it’s a neural network. It’s consists of four layers. Every layer is linear, that is a fully-connected layer. Also, there is a ReLU activation function between layers. The output of `QNet` is raw output of the final layer.

**functions**

- `sample_action(q: QNet, state: np.ndarray)`: It gets the current state and returns the proper (not always) action. Within the `EPSILON` portion, it returns a random action within $[0,9)$. In general, it returns the most-proficient action using `QNet`. It gives the flattened state to the `QNet` and gets value of each action. And returns the $\text{arg}\max$ of the value matrix. The mathematical form is as follows.
    
    $$
    \text{action}=
    \begin{cases}
    \text{random action}, \space \text{if} \space prob < \epsilon \\
    $\text{arg}\max$ \space \text{net}(\text{state}), \space \text{otherwise}
    \end{cases}
    $$
    
- `train(q: QNet, q_target: QNet, memory: ReplayBuffer, optimizer: torch.optim.Optimizer)`: It trains the given model `q` on the given `memory`, ReplayBuffer using the `optimizer`. Where the model `q_target` is being used is to set the target tensor up. When computing the target tensor, a formula below is used.
    
    $$
    R(s,a) + \gamma \max Q(s',a')
    $$
    
    More details is described here. First, we use smooth L1 loss as a loss function, which is well-known as *Huber Loss*. This function is differentiable for all real number and quite robust for outliers. The formula for the loss is as follows. 
    
    $$
    Loss_{\delta}(y,f(x))=\begin{cases}{1\over2}((y_{i}-f(x_{i}))^2 & for \left\vert y_{i}-f(x_{i}) \right\vert \le \delta, \\ \delta\left\vert y_{i}-f(x_{i}) \right\vert - {1\over2}\delta^2 & otherwise.\end{cases}
    $$
    
    And we use a hyperparameter called `K`. It describes how many time should `train()` run a loop. So, the loop in `train()` function runs `K` times. Let’s look at the inside of the loop. Let me list up the process. 
    
    1. We randomly sample the history data in the history buffer by `BATCH_SIZE`. 
    2. Then, we compute the `q_out` by using `q` network. q_out implies the value of each cell (i.e., coordinate). 
    3. Compute the `q_a` by indexing from `q_out` by `a`, which was given at the *sampling* process. I think this process is needed because other cells except for the *being placed cells* are not that important, so they need to be ignored. In other word, this process can be seen as a kinda ***filtering***. The code used here is as follows: `q_a = q_out.gather(dim=1, index=a)`
    4. Calculate the target tensor using the formula above (policy gradient). The code snippet is here: `target: torch.Tensor = r + gamma * max_q_prime * done_mask`
    5. Finally, evaluate the loss value and back-propagate!
    6. Sum up accumulatively the loss values as looping over and over again. 
    7. Return the average value of the loop. 
- `train_loop()`: It trains model for a specific epochs (i.e., the number of episodes past). In each loop, it plays a game with a random agent and record the history. As soon as the number of elements in the history buffer, it calls `train()` to have a model learn. If the index of the loop is exactly equal to the `INTERVAL`, it saves the model state, the parameters of the model.

### play.py

This script is used for playing between a human and computer, learning agent. It uses `for` loop and the Tic-Tac-Toe environment with an opponent of `LearningAgent`. 

### utils.py

Only contains function of utility. Specifically, it contains `flat_state()` function. 

### exception.py

It contains some manually declared exceptions, `OutofRangeException` and `InvalidActionException`. 

### config.py

Originally, it contained the random-seed for playing. But currently, it only has information for `torch.device`, either `cpu` or `mps` (for m1).