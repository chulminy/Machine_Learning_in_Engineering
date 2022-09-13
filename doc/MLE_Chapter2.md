<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
  type="text/javascript">
</script>

# Chapter 2. Probability: Univariate Models  

## 2.1.1 What is probability? 

[Bayesian and frequentist reasoning in plain English](https://stats.stackexchange.com/questions/22/bayesian-and-frequentist-reasoning-in-plain-english)

<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Page Title</title>
    <style>
      /* The . with the boxed represents that it is a class */
      .boxed {
        background: lightgrey;
        color: black;
        border: 3px solid black;
        margin: 0px auto;
        width: auto;
        padding: 10px;
        border-radius: 10px;
      }
    </style>
  </head>
  <body>
  </body>
</html>

<div class="boxed">
Here is how I would explain the basic difference to my grandma:

I have misplaced my phone somewhere in the home. I can use the phone locator on the base of the instrument to locate the phone and when I press the phone locator the phone starts beeping.

Problem: Which area of my home should I search?

**Frequentist Reasoning**
I can hear the phone beeping. I also have a mental model which helps me identify the area from which the sound is coming. Therefore, upon hearing the beep, I infer the area of my home I must search to locate the phone.

**Bayesian Reasoning**
I can hear the phone beeping. Now, apart from a mental model which helps me identify the area from which the sound is coming from, I also know the locations where I have misplaced the phone in the past. So, I combine my inferences using the beeps and my prior information about the locations I have misplaced the phone in the past to identify an area I must search to locate the phone.
</div>


<br/>

Let's explain this analogy in equations. 

**Frequentist Reasoning**  
I can hear the phone beeping ($\mathcal{d_1}$). I also have a mental model $p(\mathcal{D}|\theta)$ which helps me identify the area from which the sound is coming. Therefore, upon hearing the beep, I infer the area of my home I must search to locate the phone.  We knew that a beeping sounds that you hear is likely comming from a certain location $\theta^*$ among many candidate locations:

$$ \theta^* = \underset{\theta}{\mathrm{argmax} \,}p(\mathcal{d_1}|\theta) $$

If you hear a different beeping sound, you will search a different location. You will find location that maximizes your likehood function 

Now, what is a prior distribution, $p(\theta)$? When you guess the location using the above strategy $p(\mathcal{D}|\theta)$, you do not consider where you often put your phone. You are only relying on beeping sound, which is your observation. However, the probabilty of the locations of your phone placement changes a lot depending on your "prior" knowledge (habit or experience in this analogy). 

**Bayesian Reasoning**  
I can hear the phone beeping ($\mathcal{d_1}$). Now, apart from a mental model $p(\mathcal{D}|\theta)$ which helps me identify the area from which the sound is coming from, I also know the locations where I have misplaced the phone in the past $p(\theta)$. So, I combine my inferences using the beeps and my prior information about the locations I have misplaced the phone in the past to identify an area I must search to locate the phone.

$$ p(\theta|\mathcal{D}) \propto p(\mathcal{D}|\theta)p(\theta)$$

As stated, the location that you first search for could be changed depending on your prior. For example, if you are 100% sure that you put your phone at $\theta_1$ (which means $p(\theta = \theta_1) = 1$) when you misplace your phone. Then, $\underset{\theta}{\mathrm{argmax}} \, p(\theta|\mathcal{D})$ becomes $\theta_1$ because $p(\theta)=0$ when $\theta \neq \theta_1$. Another example, you usually put the phone at $\theta_1$  or $\theta_2$ equally, you decide where you go first based on beep sound and evaluate which probability is higher between $p(\mathcal{d_1}|\theta_1)$ and $p(\mathcal{d_1}|\theta_2)$


Note that $p(\mathcal{D})$ is a probability of hearing a specific beeping sound. This is from marginalization of all beeping sounds from all possiable location $\theta$. There is no special meaning in this problem. 

One confusing concept indicates where our mental model guesses the probability of the phone location when we heard a certain beep sound (posterior) or the probability of a certain beep sound depending on the phone location (likelihood). Consider how to identify the location from a beep soud. We are using the direction and intensity of the sound to find the locations. The mental model mentioned here is just detecting a direction and distance of a beep sound and does not consider a house layout (as a prior knowledge). 


### 2.1.3.6 Conditional independence of events

What's the intuitive example?   
[Conditional Independence — The Backbone of Bayesian Networks](https://towardsdatascience.com/conditional-independence-the-backbone-of-bayesian-networks-85710f1b35b)

<div class="boxed">

Let’s say A is the height of a child and B is the number of words that the child knows. It seems when A is high, B is high too.

There is a single piece of information that will make A and B completely independent. What would that be?

**The child's age**

The height and the # of words known by the kid are NOT independent, but they are conditionally independent if you provide the kid’s age.

</div>

## 2.2 Random Variables

### 2.1.1 Discrete random variables

Suppose $X$ represents some unkown quantity of interest. If the value of X is unkown and/or could change, we call it a random variable (**rv**). The set of possible values, denoted $\chi$ is known as the **sample space** or **state space**. An **event** is a set of outcomes from a given sample space. 

For example, if $X$ represents the face of a dice that is rolled, so $\chi = \{1,2, ... 6\}$, the event of "seeing a 1" is denoted $X = 1$, the event of "seeing an odd number" is denoted $ X \in \{1,3,5\}$. 

**Probability mass function (pmf)**
$$ p(x) \triangleq \Pr (X \leq x) $$

where $X$ is a discrete random variable of which sample space is finite or countably infinite. 

See "Fig2_1 - Some Discrete Distributions.ipynb"



### 2.2.2 Continuous distribution function


## 3.2 The multivariate Gaussian (normal) distribution

#### Example: conditioning a 2d Gaussian

See ["Two properties of the Gaussian distribution (from Fabian Dablander)"](https://fabiandablander.com/statistics/Two-Properties.html)

*Marginalizing means **ignoring**, and conditioning means **incorporating** information.*

Let us consider a 2d example. Two random variables $X_1$ and $X_2$ are introduced and each variable is coming from a univariate Gaussian distribution with mean 


**Convex combination**

For a vector ($x_1, x_2, ...., x_n$), the convex combination is:
$$ \alpha_1 x_1 + \alpha_2 x_2 + \alpha_3 x_3+ ... \alpha_n x_n $$ 
where  $\alpha_i \ge 0 $ **and** $\alpha_1+ \alpha_2+ \alpha_3+ ... +\alpha_n = 1$


## Mixture models

latent variables (from Latin: present participle of lateo (“lie hidden”), opposed to observable variables) are variables that are not directly observed but are rather inferred through a mathematical model from other variables that are observed (directly measured). Mathematical models that aim to explain observed variables in terms of latent variables are called latent variable models. 