Environments
- True Extinction Environment:
  Starts off with two arms, one arm more rewarding than the other.
  The reward frequecies of both arms add up to one. 
  So, is the most rewarding arm delivers reward .9 of the time,
  the other arm delivers reward .1 of the time.
  When extinction occurs, no reward is emitted from either arm.
  After extinction, reward resumes as before.

- Switching Evironment:
  Starts off just like true extinction environment.
  During extinction, the two arms are flipped, 
  so that the arm that previously had the highest 
  rate of reward, now has the reward frequency of the 
  less rewarding arm, and vice-versa. When extinction
  ends, the arms switch back to normal.
  
  first arm: .9 | second arm: .1 -> first arm: .1 | second arm: .9

- Per Arm Switching Environment:
  Identical to the switching environment; however, during extinction, 
  the arms switch back and forth randomly. The rate at which the arms 
  switch back and forth are determined by an argument to the class.
  
Bodies
- Null Body:
  Perceives every reward that is gained. If the agent picks an arm, 
  and the arm is rewarding during that time step, then that reward 
  is counted.
  
- Info Gain Body:
  Reward is not counted during learning phase. While the reward is still 
  registered by the agent (affects rates of arm pulling), the reward is 
  not counted towards the fitness. This allows our agent to practice before
  they are tested.
  
- Noisy Body:
  The ability to gain reward shuts of randomly throughout the simulation.
  During the time the ability to gain reward is shut off, reward cannot 
  be added to the final fitness of the agent, and the agent is also unable
  to perceive reward, causing any movements to be costly. 
