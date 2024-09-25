(chapter0.0)=
# Chapter 0: Gental Review on Key Concepts 

Before diving into the fascinating world of diffusion models, let's have the refresher 
of some key concepts in diffusion models, which would be useful later on. Notice that
here we just briefly touch the [**forward process**](chapter0.3) and [**reverse process**]
intuitively.  We'll have detailed explanation about them in [**Chapter 2**](chapter2.0).
Similarly, [**score function**](chapter0.4) as a commonly used objective function in 
diffusion models will be emphasized in following chapters.


(chapter0.1)=
## 1. **Probability Distributions**
Diffusion models are heavily based on probabilistic methods, making an understanding of probability distributions essential. The most important distribution is:

- **Gaussian (Normal) Distribution**: 
  - Used in the noise processes of diffusion models.
  - Defined by mean $\mu$ and variance $\sigma^2$, denoted as $ \mathcal{N}(\mu, \sigma^2) $.
  - Probability density function (PDF):
    ```{math}
    p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right).
    ```


(chapter0.2)=
## 2. **Markov Chains**
A **Markov chain** is a type of stochastic process that moves through a series of states, where the probability of transitioning to the next state depends only on the current state (and not on the history of states). This **"memoryless"** property is fundamental in modeling processes such as the noise injection in diffusion models.

### Key Properties:
- **Transition Probability**: 
  - Describes the probability of moving from one state to another.
  - If $X_t$ is the current state at time $t$, then the transition probability is:
    ```{math}
    P(X_{t+1} = x' | X_t = x) = P(x' | x).
    ```
- **Markov Property**: 
  - Future states depend <u>only on the present state</u>, not past states.
    ```{math}
    P(X_{t+1} | X_t, X_{t-1}, \dots, X_0) = P(X_{t+1} | X_t)
    ```

A **stationary distribution** is a probability distribution that remains constant in a Markov chain as it transitions between states. If a Markov chain reaches its stationary distribution, further transitions do not change the overall distribution of states.


(chapter0.3)=
## 3. **Diffusion Process**
A diffusion process describes the spread of particles (or probabilities) over time. The **forward process** introduces noise, while the **reverse process** denoises to recover a clear sample:

- **Forward Process**:
  ```{math}
  q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{\alpha_t} x_{t-1}, (1-\alpha_t)I)
  ```
  
```{figure} ../images/markov_chain.jpg
:height: 80px
:name: markov_chain_forward_process

With certain transition probability $q$, the **forward process** gradually add noise to the image $x_0$ since time $t=0$, and finally it becomes pure noise $x_T$ at time $T$.
```
  
- **Reverse Process**:
  ```{math}
  p(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(t))
  ```


(chapter0.4)=
## 4. **Score Matching**
In diffusion models, score matching helps recover the score function (gradient of log-probability):

- **Score Function**:
  ```{math}
  \nabla_x \log p(x)
  ```
  The score function is used to guide the **reverse process** back to the original data distribution.


(chapter0.5)=
## 5. **Stochastic Processes**
A **stochastic process** describes systems evolving over time under random influences. Important stochastic tools include:

### Wiener Process:
- **Wiener Process**, also known as **Brownian motion**, is a continuous-time stochastic process with the following properties:
  i. **Initial value**: $W(0) = 0$.
  ii. **Independent increments**: The increments $W(t) - W(s)$ are independent for $t > s$.
  iii. **Normally distributed increments**: For $t > s$, the increment $W(t) - W(s)$ is distributed as $\mathcal{N}(0, t - s)$.
  iv. **Continuous paths**: The function $W(t)$ is continuous in $t$, though it is non-differentiable at any point.
  
- The Wiener process is the core building block of **stochastic differential equations (SDEs)**, which are used to model noise in diffusion models.
The Wiener process $W(t)$ satisfies the stochastic differential equation:
  ```{math}
  dW(t) = \mathcal{N}(0, dt)
  ```
- This means that over a small time interval $dt$, the change in the process is normally distributed with mean zero and variance proportional to $dt$.

### Ito Calculus:
- **Ito Calculus** is a branch of stochastic calculus that extends traditional calculus to deal with the randomness introduced by processes like the Wiener process. It is specifically designed to handle stochastic differential equations (SDEs), which are central to diffusion models.
- **Ito's Lemma** is the stochastic calculus counterpart of the chain rule in classical calculus. If $f$ is a twice-differentiable function and $X_t$ is a stochastic process described by the SDE:
  ```{math}
  dX_t = \mu(X_t, t) dt + \sigma(X_t, t) dW_t
  ```
  
  Then Ito's Lemma gives the differential for $f(X_t, t)$:
  ```{math}
  df(X_t, t) = \frac{\partial f}{\partial X_t} dX_t + \frac{\partial f}{\partial t} dt + \frac{1}{2} \sigma^2(X_t, t) \frac{\partial^2 f}{\partial X_t^2} dt
  ```
  
- The **Ito integral** is defined differently from classical Riemann integrals due to the non-differentiability of the Wiener process. For a stochastic process $X(t)$ and a Wiener process $W(t)$, the Ito integral is written as:
  ```{math}
  \int_0^t X(s) dW(s)
  ```
  This integral represents the cumulative effect of random fluctuations up to time $t$, and it is used to solve SDEs.
- In the context of diffusion models, the **stochastic differential equations** describe the evolution of the data as it undergoes noise injection or denoising. These equations typically take the form:
  ```{math}
  dx_t = \mu(x_t, t) dt + \sigma(t) dW_t
  ```
  where:
     - $\mu(x_t, t)$ represents the deterministic drift term (direction of movement),
     - $\sigma(t)$ represents the volatility or noise intensity,
     - $dW_t$ is the Wiener process term that introduces randomness.
  In the **reverse process** of diffusion models, solving these SDEs using Ito calculus helps to recover the original data distribution by progressively removing noise.


(chapter0.5)=
## 6. **KL Divergence**
The **Kullback-Leibler (KL) Divergence** measures the difference between two probability distributions $P$ and $Q$:

```{math}
D_{KL}(P || Q) = \int P(x) \log \frac{P(x)}{Q(x)} \, dx
```

KL divergence is often used in diffusion models to minimize the difference between the true distribution and the model distribution.


(chapter0.7)=
## 7. **Variational Inference**

- **Variational Inference** (VI) is a technique used in probabilistic models to approximate complex, intractable probability distributions with simpler, tractable ones. In diffusion models, VI is crucial for learning the parameters of the reverse process, which transforms noisy data back into clean data. The goal is to approximate the true posterior distribution $p(x_{1:T} | x_0)$ (i.e., the reverse process) using a variational distribution $q(x_{1:T} | x_0)$, which is simpler to compute.

### Evidence Lower Bound (ELBO):
- The objective in variational inference is to maximize the **Evidence Lower Bound (ELBO)**, a quantity that provides a lower bound to the likelihood of the observed data. Maximizing the ELBO indirectly helps approximate the true posterior distribution.

- The ELBO is given by:
  ```{math}
  \mathcal{L} = \mathbb{E}_q \left[ \log p_\theta(x_0 | x_t) - D_{KL}(q(x_{1:T} | x_0) || p_\theta(x_{1:T})), \right]
  ```
  where:
  - $p_\theta(x_0 | x_t)$: The likelihood of the data given the noisy version at time $t$, parameterized by $\theta$.
  - $D_{KL}(q || p)$: The Kullback-Leibler (KL) divergence between the variational distribution $q(x_{1:T} | x_0)$ and the model distribution $p_\theta(x_{1:T})$, which measures how different these two distributions are.
  - $x_0$: The original (clean) data.
  - $x_t$: The noisy data at time step $t$.

### Breakdown of the ELBO:
i. **Reconstruction Term**: 
   - $\log p_\theta(x_0 | x_t)$ encourages the model to generate samples $x_0$ that closely resemble the original clean data given the noisy data $x_t$. It measures how well the model can reconstruct clean data from noisy observations.
   
ii. **KL Divergence Term**:
   - $D_{KL}(q(x_{1:T} | x_0) || p_\theta(x_{1:T}))$ penalizes the difference between the learned reverse process (the variational approximation) and the true posterior distribution. Minimizing this term helps align the learned process with the actual data-generating process.

- In diffusion models, the **forward process** gradually adds noise to data, leading to a tractable Gaussian distribution at the final step. The **reverse process**, however, is intractable. Using variational inference allows the model to **approximate the reverse process**, and by maximizing the ELBO, it becomes better at denoising, generating samples that are close to the original data distribution. This is critical for high-quality image or data generation in diffusion models.


(chapter0.8)=
## 8. **Monte Carlo Sampling**

- **Monte Carlo Sampling** is a numerical method used to estimate expectations of functions or integrals when direct computation is infeasible. In the context of diffusion models, Monte Carlo methods allow for the approximation of various quantities, such as expectations over complex distributions, by drawing random samples from these distributions.

### Estimating Expectations:
- In many cases, the exact expectation $\mathbb{E}[f(x)]$ under some probability distribution $p(x)$ is difficult to compute analytically. Instead, Monte Carlo methods estimate this expectation by drawing samples $x_1, x_2, \dots, x_N$ from the distribution $p(x)$, and using the empirical average:
  ```{math}
\mathbb{E}[f(x)] \approx \frac{1}{N} \sum_{i=1}^{N} f(x_i),
  ```
where:
  - $f(x)$: A function whose expectation you want to compute.
  - $N$: The total number of samples drawn.
  - $x_i$: Samples drawn from the distribution $p(x)$.

- In diffusion models, during training (**forward process**), random noise is added to the data at each time step $t$, typically sampled from a Gaussian distribution. Monte Carlo Sampling can be used to approximate how this noise impacts the data; Monte Carlo Sampling can be employed to approximate the **reverse process** by drawing samples from the variational posterior $q(x_{1:T} | x_0)$, especially when computing the expectation over possible latent variables or noisy data.


(chapter0.9)=
## 9. Remarks
Notice that all the terminologies above are brief and at a introductory level, and are by no means rigorous definitions. If you want to know more about them, please refer to the **Wikipedia page** of those concepts. Cheers!