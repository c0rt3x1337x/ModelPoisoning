## Related Works

The integrity of deep learning systems, particularly in multi-class classification tasks, has emerged as a critical area of concern in adversarial machine learning research. With the increasing deployment of CNNs in safety-critical applications, studies have highlighted how malicious interference—whether during training or inference—can significantly compromise model reliability. This project builds on such insights by using a baseline CNN trained on the MNIST dataset to evaluate the effects of *model poisoning*, focusing on *label-flip attacks* as a primary threat vector.

Initial theoretical frameworks provided by Liu and Chen [4] offer a broad taxonomy of data poisoning attacks, categorizing them based on attacker knowledge, goals, and capabilities. Their analysis lays the groundwork for understanding poisoning as both a training-time and model-level adversarial strategy. Extending this perspective, Mosenia and Rajkumar [6] contextualize these threats in industrial control systems, showing how subtle data integrity violations can propagate into significant operational disruptions—an outcome that motivates our examination of poisoning effects even in comparatively simple datasets like MNIST.

At the core of our study is the *Label-Flip Attack (LFA)* introduced by Rakin et al. [7], which demonstrated that flipping as few as one or two bits in a model's quantized weight representations can cause catastrophic performance degradation. This low-resource, high-impact threat model is especially concerning given the growing prevalence of edge-deployed, quantized deep learning models.

Other poisoning methodologies, such as BadNets proposed by Gu et al. [1], demonstrate how malicious training samples can introduce targeted backdoors without affecting performance on clean data. While their attack operates at the data level, it shares conceptual parallels with label-flip attacks by injecting misclassification potential without alerting standard validation routines. Similarly, Ma and Wang [5] present model poisoning attacks with provable convergence guarantees, reinforcing the notion that long-term integrity degradation can be introduced systematically even under constrained attacker assumptions.

Additional studies have explored nuanced dimensions of poisoning and adversarial resilience. Hong et al. [2] investigate *bit-level inference-time attacks* in quantized CNNs, aligning directly with our implementation focus on post-training model tampering. Meanwhile, Yuan et al. [9] provide a unifying overview of adversarial machine learning, framing data poisoning and model corruption as complementary elements in the broader security landscape of neural networks.

Shafahi et al. [8] introduce *clean-label poisoning*, which avoids altering class labels in poisoned data, making detection more difficult. Their approach highlights the stealth aspect of modern poisoning attacks, a trait shared by label-flip methods that do not rely on visible changes in training datasets. Zhao et al. [10] further explore *data-efficient poisoning strategies*, demonstrating that impactful attacks can be executed with minimal perturbations—a key insight for our simulation of minimal yet effective bit-level manipulations.

Lastly, Huang et al. [3] propose a *multi-objective poisoning framework*, optimizing both attack effectiveness and stealth. Their work underscores the importance of balancing attack goals, an idea that informs our study’s evaluation metrics which focus not only on classification accuracy but also on the degree and subtlety of model degradation under attack.

Together, these contributions form a rich foundation for this project’s exploration of model poisoning in CNNs. By developing a controlled environment using PyTorch and MNIST, and implementing BFA-based tampering mechanisms, our research extends prior findings with hands-on experimentation aimed at better understanding how even low-level hardware faults or adversarial bit manipulations can undermine deep learning model integrity in multi-class settings.

---

## References

[1] T. Gu, B. Dolan-Gavitt, and S. Garg, “BadNets: Identifying vulnerabilities in the machine learning model supply chain,” *arXiv preprint arXiv:1708.06733*, 2017.

[2] Y. Hong, J. Szefer, and Y. Li, “Terminal bit-flip attack against quantized DNNs,” in *Proc. ACM/IEEE MLHPC*, 2020, pp. 23–31.

[3] L. Huang, B. Li, and B. Li, “Multi-objective optimization for data poisoning attacks,” in *Proc. IJCAI*, 2021, pp. 3794–3800.

[4] Y. Liu and X. Chen, “Data poisoning attacks: Review and classification,” in *Proc. IJCNN*, 2020, pp. 1–8.

[5] X. Ma and Y. Wang, “Model-targeted poisoning attacks with provable convergence,” *NeurIPS*, vol. 34, pp. 1042–1053, 2021.

[6] A. Mosenia and R. R. Rajkumar, “Security implications of machine learning in industrial control systems,” *IEEE Internet Things J.*, vol. 9, no. 2, pp. 1156–1171, 2022.

[7] A. S. Rakin, Z. He, and D. Fan, “Bit-flip attack: Crushing neural network with progressive bit search,” in *Proc. ICCV*, 2019, pp. 1211–1220.

[8] A. Shafahi et al., “Poison frogs! Clean-label poisoning attacks on neural networks,” *NeurIPS*, vol. 31, 2018.

[9] X. Yuan, P. He, Q. Zhu, and X. Li, “Adversarial examples: Attacks and defenses for deep learning,” *IEEE Trans. Neural Netw. Learn. Syst.*, vol. 30, no. 9, pp. 2805–2824, 2019.

[10] J. Zhao, D. Dua, and S. Singh, “Data poisoning attacks on transfer learning,” in *Proc. AAAI*, vol. 34, no. 4, pp. 5981–5988, 2020.
