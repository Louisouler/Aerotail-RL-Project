# References

Suggested reference list for the Part-E report, numbered in the order the topics
normally appear (algorithms, tooling, environment, curriculum, perception). Cite
only the ones the text actually refers to.

## Algorithms

[1] T. Haarnoja, A. Zhou, P. Abbeel and S. Levine, "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor," in Proc. 35th Int. Conf. on Machine Learning (ICML), 2018. https://arxiv.org/abs/1801.01290

[2] T. Haarnoja, A. Zhou, K. Hartikainen, G. Tucker, S. Ha, J. Tan, V. Kumar, H. Zhu, A. Gupta, P. Abbeel and S. Levine, "Soft Actor-Critic Algorithms and Applications," arXiv:1812.05905, 2018. (automatic entropy tuning, used via `ent_coef="auto"`) https://arxiv.org/abs/1812.05905

[3] J. Schulman, F. Wolski, P. Dhariwal, A. Radford and O. Klimov, "Proximal Policy Optimization Algorithms," arXiv:1707.06347, 2017. https://arxiv.org/abs/1707.06347

[4] J. Schulman, P. Moritz, S. Levine, M. Jordan and P. Abbeel, "High-Dimensional Continuous Control Using Generalized Advantage Estimation," in Proc. Int. Conf. on Learning Representations (ICLR), 2016. https://arxiv.org/abs/1506.02438

[5] S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," Neural Computation, vol. 9, no. 8, pp. 1735-1780, 1997.

[6] M. Hausknecht and P. Stone, "Deep Recurrent Q-Learning for Partially Observable MDPs," in AAAI Fall Symposium on Sequential Decision Making for Intelligent Agents, 2015. https://arxiv.org/abs/1507.06527 (motivation for a recurrent policy under partial observability)

## Software

[7] A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus and N. Dormann, "Stable-Baselines3: Reliable Reinforcement Learning Implementations," Journal of Machine Learning Research, vol. 22, no. 268, pp. 1-8, 2021. https://jmlr.org/papers/v22/20-1364.html (version 2.9.0 used)

[8] Stable-Baselines3 Contrib (sb3-contrib), RecurrentPPO implementation, https://github.com/Stable-Baselines-Team/stable-baselines3-contrib (version 2.9.0 used)

[9] M. Towers et al., "Gymnasium: A Standard Interface for Reinforcement Learning Environments," arXiv:2407.17032, 2024. https://arxiv.org/abs/2407.17032 (version 1.3.0 used)

[10] Plotly Dash, https://dash.plotly.com (dash 4.4.1, plotly 7.1.0; monitoring dashboard)

## Simulation environment

[11] S. Shah, D. Dey, C. Lovett and A. Kapoor, "AirSim: High-Fidelity Visual and Physical Simulation for Autonomous Vehicles," in Field and Service Robotics, Springer, 2018, pp. 621-635. https://arxiv.org/abs/1705.05065 (simulator binaries v1.7.0 for Linux: Blocks and AirSimNH environments; Python client 1.8.1)

## Curriculum learning and generalisation

[12] Y. Bengio, J. Louradour, R. Collobert and J. Weston, "Curriculum Learning," in Proc. 26th Int. Conf. on Machine Learning (ICML), 2009, pp. 41-48.

[13] S. Narvekar, B. Peng, M. Leonetti, J. Sinapov, M. E. Taylor and P. Stone, "Curriculum Learning for Reinforcement Learning Domains: A Framework and Survey," Journal of Machine Learning Research, vol. 21, no. 181, pp. 1-50, 2020. https://jmlr.org/papers/v21/20-212.html

[14] J. Tobin, R. Fong, A. Ray, J. Schneider, W. Zaremba and P. Abbeel, "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World," in Proc. IEEE/RSJ Int. Conf. on Intelligent Robots and Systems (IROS), 2017. https://arxiv.org/abs/1703.06907 (context for the Blocks -> AirSimNH transfer result)

## Perception side study

[15] P. Zhu, L. Wen, D. Du, X. Bian, H. Fan, Q. Hu and H. Ling, "Detection and Tracking Meet Drones Challenge," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 11, pp. 7380-7399, 2022. https://arxiv.org/abs/2001.06303 (VisDrone dataset)

[16] G. Jocher, J. Qiu and A. Chaurasia, Ultralytics YOLO, https://github.com/ultralytics/ultralytics (YOLOv8n and YOLO26l used)

Please check the page numbers and author lists against the publisher pages before
submission; they were written from memory and not looked up online.
