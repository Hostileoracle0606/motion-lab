# References

This reading list favors primary papers, textbooks, and official documentation.
Inclusion indicates relevance to the roadmap, not that Motion Lab has reproduced
the work.

## Simulation and feedback control

1. Todorov, E., Erez, T., and Tassa, Y. (2012). “MuJoCo: A Physics Engine for Model-Based Control.” *IEEE/RSJ International Conference on Intelligent Robots and Systems*. [DOI](https://doi.org/10.1109/IROS.2012.6386109)
2. MuJoCo documentation. [Official documentation](https://mujoco.readthedocs.io/)
3. Åström, K. J., and Murray, R. M. (2021). *Feedback Systems: An Introduction for Scientists and Engineers*, second edition. [Open textbook](https://fbswiki.org/wiki/index.php/Feedback_Systems:_An_Introduction_for_Scientists_and_Engineers)
4. Lynch, K. M., and Park, F. C. (2017). *Modern Robotics: Mechanics, Planning, and Control*. Cambridge University Press. [Book resources](https://modernrobotics.northwestern.edu/)

## Dynamics, system identification, and model-based control

5. Ljung, L. (1999). *System Identification: Theory for the User*, second edition. Prentice Hall.
6. Sutton, R. S. (1991). “Dyna, an Integrated Architecture for Learning, Planning, and Reacting.” *ACM SIGART Bulletin*, 2(4). [DOI](https://doi.org/10.1145/122344.122377)
7. Camacho, E. F., and Bordons, C. (2007). *Model Predictive Control*, second edition. Springer. [DOI](https://doi.org/10.1007/978-0-85729-398-5)
8. Williams, G., Aldrich, A., and Theodorou, E. A. (2017). “Model Predictive Path Integral Control: From Theory to Parallel Computation.” *Journal of Guidance, Control, and Dynamics*, 40(2). [DOI](https://doi.org/10.2514/1.G001921)
9. Deisenroth, M. P., and Rasmussen, C. E. (2011). “PILCO: A Model-Based and Data-Efficient Approach to Policy Search.” *ICML*. [Paper](https://icml.cc/2011/papers/323_icmlpaper.pdf)
10. Chua, K., Calandra, R., McAllister, R., and Levine, S. (2018). “Deep Reinforcement Learning in a Handful of Trials Using Probabilistic Dynamics Models.” *NeurIPS*. [Paper](https://papers.nips.cc/paper_files/paper/2018/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html)
11. Hansen, N., Su, H., and Wang, X. (2024). “TD-MPC2: Scalable, Robust World Models for Continuous Control.” *ICLR*. [Paper](https://proceedings.iclr.cc/paper_files/paper/2024/hash/cf73d57b6dcda32b293df7c2d5341f49-Abstract-Conference.html)
12. Hafner, D., Lillicrap, T., Ba, J., and Norouzi, M. (2020). “Dream to Control: Learning Behaviors by Latent Imagination.” *ICLR*. [Paper](https://openreview.net/forum?id=S1lOTC4tDS)

## Predictive representations and world models

13. LeCun, Y. (2022). “A Path Towards Autonomous Machine Intelligence.” [OpenReview](https://openreview.net/forum?id=BZ5a1r-kVsf)
14. Assran, M. et al. (2023). “Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture.” *CVPR*. [arXiv](https://arxiv.org/abs/2301.08243)
15. Bardes, A. et al. (2024). “Revisiting Feature Prediction for Learning Visual Representations from Video.” [arXiv](https://arxiv.org/abs/2404.08471)
16. Hafner, D. et al. (2019). “Learning Latent Dynamics for Planning from Pixels.” *ICML*. [Paper](https://proceedings.mlr.press/v97/hafner19a.html)
17. Assran, M. et al. (2025). “V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning.” [arXiv](https://arxiv.org/abs/2506.09985)
18. Ha, D., and Schmidhuber, J. (2018). “World Models.” [arXiv](https://arxiv.org/abs/1803.10122)
19. Schrittwieser, J. et al. (2020). “Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model.” *Nature*, 588. [DOI](https://doi.org/10.1038/s41586-020-03051-4)

## Temporal abstraction and hierarchy

20. Sutton, R. S., Precup, D., and Singh, S. (1999). “Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning.” *Artificial Intelligence*, 112(1–2). [DOI](https://doi.org/10.1016/S0004-3702(99)00052-1)
21. Dayan, P., and Hinton, G. E. (1993). “Feudal Reinforcement Learning.” *NeurIPS*. [Paper](https://papers.nips.cc/paper/714-feudal-reinforcement-learning)
22. Nachum, O. et al. (2018). “Data-Efficient Hierarchical Reinforcement Learning.” *NeurIPS*. [Paper](https://papers.nips.cc/paper/7591-data-efficient-hierarchical-reinforcement-learning)

## Reproducible machine learning

23. Pineau, J. et al. (2021). “Improving Reproducibility in Machine Learning Research.” *Journal of Machine Learning Research*, 22(164). [Paper](https://jmlr.org/papers/v22/20-303.html)
24. Gebru, T. et al. (2021). “Datasheets for Datasets.” *Communications of the ACM*, 64(12). [DOI](https://doi.org/10.1145/3458723)

## Suggested reading order

For the current stage:

1. MuJoCo documentation for simulator conventions.
2. *Feedback Systems* for feedback, stability, and controller intuition.
3. *Modern Robotics* for mechanics and control terminology.
4. Dyna, PILCO, PETS, and TD-MPC2 for the progression from experience to learned dynamics and planning.
5. PlaNet, I-JEPA, V-JEPA, and V-JEPA 2 before designing latent-prediction experiments.
6. The options framework before defining temporal abstractions.

When an experiment adopts an equation, architecture, metric, or protocol from a
source, its README should cite that source directly and describe any deviation.
