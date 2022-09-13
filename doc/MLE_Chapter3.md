<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
  type="text/javascript">
</script>

# Chapter 3. Probability: Univariate Models  


## 3.2 The multivariate Gaussian (normal) distribution

#### Example: conditioning a 2d Gaussian

See ["Two properties of the Gaussian distribution"](https://fabiandablander.com/statistics/Two-Properties.html)

*Marginalizing means **ignoring**, and conditioning means **incorporating** information.*

Let us consider a 2d example. Two random variables $X_1$ and $X_2$ are introduced and each variable is coming from a univariate Gaussian distribution with mean 


**Convex combination**

For a vector ($x_1, x_2, ...., x_n$), the convex combination is:
$$ \alpha_1 x_1 + \alpha_2 x_2 + \alpha_3 x_3+ ... \alpha_n x_n $$ 
where  $\alpha_i \ge 0 $ **and** $\alpha_1+ \alpha_2+ \alpha_3+ ... +\alpha_n = 1$


## Mixture models

latent variables (from Latin: present participle of lateo (“lie hidden”), opposed to observable variables) are variables that are not directly observed but are rather inferred through a mathematical model from other variables that are observed (directly measured). Mathematical models that aim to explain observed variables in terms of latent variables are called latent variable models. 