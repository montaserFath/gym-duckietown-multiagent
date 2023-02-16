# Multi-agent Gym-Duckietown

Multi-agent version of [Duckietown](http://duckietown.org/) self-driving car simulator environments for OpenAI Gym.

### Done
1- add another bot (2 agents) to the simulator.

2- Multi-agent simulator: multi-agent gym env inherited from a single-agent simulator, change observation-space to include all agents, as well as action space.

3- Multi-agent simulator: `reset` multi-agent env, `step` all agents, `update_physics()` to move all agents, and `step()` to step multi-agent simulator.

4- Refactor single-agent simulator to fit multiple agents.

5- Test the multi-agent simulator with single and two agents. 

6- Train single agent policy where the observations as row images and discrete actions (it takes about 5 hours to learn to drive).

---
### WIP
1- Multi-agent to single-agent wrapper (wrap other agents' policy as part of the environment).

2- Simple RL wrapper: takes an image, detects lines, calculates bot distance to edges and the central line, as well as other obervation.

3- Process image to sensor observations.

4- Add reward functions for multi-agent env.

5- Documentation.

6- Linting and pass code tests.

7- collision between agents --> termnation (`done`).

---
### TODO
1- Test multi-agent to a single-agent wrapper with fix and random policy.

2- Train multi-agent with multi-agent PPO.

3- Add options for reward shaping and domain randomization.

4- Add unit test.

5- Add evaluation matrix (reward, image processing speed FPS, etc).

6- self-play training.
