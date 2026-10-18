# References

This reading list favors primary papers, textbooks, and official documentation. Inclusion indicates relevance to the roadmap, not that Motion Lab has reproduced the work.

## Simulation and feedback control

1. Todorov, E., Erez, T., and Tassa, Y. (2012). “MuJoCo: A physics engine for model-based control.” *IEEE/RSJ International Conference on Intelligent Robots and Systems*. [DOI](https://doi.org/10.1109/IROS.2012.6386109)
2. MuJoCo documentation. [Official documentation](https://mujoco.readthedocs.io/)
3. Åström, K. J., and Murray, R. M. (2021). *Feedback Systems: An Introduction for Scientists and Engineers*, second edition. [Open textbook](https://fbswiki.org/wiki/index.php/Feedback_Systems:_An_Introduction_for_Scientists_and_Engineers)
4. Lynch, K. M., and Park, F. C. (2017). *Modern Robotics: Mechanics, Planning, and Control*. Cambridge University Press. [Book resources](https://modernrobotics.northwestern.edu/)

## Dynamics, system identification, and model-based control

5. Ljung, L. (1999). *System Identification: Theory for the User*, second edition. Prentice Hall.
6. Sutton, R. S. (1991). “Dyna, an integrated architecture for learning, planning, and reacting.” *ACM SIGART Bulletin*, 2(4). [DOI](https://doi.org/10.1145/122344.122377)
7. Camacho, E. F., and Bordons, C. (2007). *Model Predictive Control*, second edition. Springer. [DOI](https://doi.org/10.1007/978-0-85729-398-5)
8. Williams, G., Aldrich, A., and Theodorou, E. A. (2017). “Model Predictive Path Integral Control: From theory to parallel computation.” *Journal of Guidance, Control, and Dynamics*, 40(2). [DOI](https://doi.org/10.2514/1.G001921)
9. Chua, K., Calandra, R., McAllister, R., and Levine, S. (2018). “Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models.” *NeurIPS*. [Paper](https://papers.nips.cc/paper/7725-deep-reinforcement-learning-in-a-handful-of-trials-using-probabilistic-dynamics-models)
10. Hafner, D., Lillicrap, T., Ba, J., and Norouzi, M. (2020). “Dream to Control: Learning Behaviors by Latent Imagination.” *ICLR*. [Paper](https://openreview.net/forum?id=S1lOTC4tDS)

## Predictive representations and world models

11. LeCun, Y. (2022). “A Path Towards Autonomous Machine Intelligence.” [OpenReview](https://openreview.net/forum?id=BZ5a1r-kVsf)
12. Assran, M. et al. (2023). “Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture.” *CVPR*. [arXiv](https://arxiv.org/abs/2301.08243)
13. Bardes, A. et al. (2024). “V-JEPA: Latent Video Prediction for Visual Representation Learning.” [arXiv](https://arxiv.org/abs/2402.08446)
14. Ha, D., and Schmidhuber, J. (2018). “World Models.” [arXiv](https://arxiv.org/abs/1803.10122)
15. Schrittwieser, J. et al. (2020). “Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model.” *Nature*, 588. [DOI](https://doi.org/10.1038/s41586-020-03051-4)

## Temporal abstraction and hierarchy

16. Sutton, R. S., Precup, D., and Singh, S. (1999). “Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning.” *Artificial Intelligence*, 112(1–2). [DOI](https://doi.org/10.1016/S0004-3702(99)00052-1)
17. Dayan, P., and Hinton, G. E. (1993). “Feudal Reinforcement Learning.” *NeurIPS*. [Paper](https://papers.nips.cc/paper/714-feudal-reinforcement-learning)
18. Nachum, O. et al. (2018). “Data-Efficient Hierarchical Reinforcement Learning.” *NeurIPS*. [Paper](https://papers.nips.cc/paper/7591-data-efficient-hierarchical-reinforcement-learning)

## Reproducible machine learning

19. Pineau, J. et al. (2021). “Improving Reproducibility in Machine Learning Research.” *Journal of Machine Learning Research*, 22(164). [Paper](https://jmlr.org/papers/v22/20-303.html)
20. Gebru, T. et al. (2021). “Datasheets for Datasets.” *Communications of the ACM*, 64(12). [DOI](https://doi.org/10.1145/3458723)

## Suggested reading order

For the current stage:

1. MuJoCo documentation for simulator conventions.
2. *Feedback Systems* for feedback, stability, and controller intuition.
3. *Modern Robotics* for mechanics and control terminology.
4. The Dyna paper and an MPC reference for the bridge from prediction to planning.
5. I-JEPA, V-JEPA, and the autonomous-machine position paper before designing latent-prediction experiments.
6. The options framework before defining temporal abstractions.

When an experiment adopts an equation, architecture, metric, or protocol from a source, its README should cite that source directly and describe any deviation.
