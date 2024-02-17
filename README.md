# Resilience_based_RL

Exothermic reactions carried out in batch reactors need a lot of attention to operate because any insufficient condition can lead to thermal runaway causing an explosion in the worst case.

Therefore a well-designed intervention action is necessary to avoid non-desired events. For this problem, we propose to use resilience-based reinforcement learning, where the artificial agent can decide whether to intervene or not based on the present state of the system.

One of our goals is to design resilient systems, which means to design such systems which can recover after a disruption. Hence we developed the calculation method of the resilience for reactors, where we suggest using dynamic predictive time to failure and recover for better resilience evaluation. Moreover, if the process state is out of the design parameters then we do not suggest calculating with the adaptation and recovery phase.

We suggest using Deep Q-learning to learn when to intervene in the system to avoid catastrophic events, where we propose to use the resilience metric as a reward function for the learning process.
The results show that the proposed methodology is applicable to developing resilient-based mitigation systems, and the agent can effectively distinguish between normal and hazardous states.

![Figure_structure](https://github.com/AgentKummer/Resilience_based_RL/assets/131676644/ecbbe139-0116-4774-aa77-baf2ca93c98d)
