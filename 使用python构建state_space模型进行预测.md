State-Space模型和卡尔曼滤波器

考虑以下State-Space模型

$$
y_t=z\alpha_{t-1}+e_t
$$

$$
\alpha_t=c+w\alpha_{t-1}+u_t
$$

我们定义$\alpha_t$是状态变量（观测不到的数据），$y_t$是我们可以观测到的数据。

令人注意到的是$c$是一个常量，$t=1,2,\cdots,n$是我们假设观测到的$n$个时间点。假设$e_t$ 和 $u_t$ 是不相关的正态分布随机变量，即对任意$j$和$i$ ，$e_t\sim N(0,\sigma_{e}^2)$，$u_t\sim N(0,\sigma_{u}^2)$且$E(e_{t-j} u_{t-i})=0$ 。因为正态分布或者高斯分布的线性组合仍然是正态分布的，向量$$\left[\begin{array}{c}
 y_t\\\alpha_t 
\end{array}\right]$$仍然遵循二元正态部分。

现在我们定义$E(\alpha_t)=c+w E(\alpha_{t-1})$，上文中二元正态分布向量$$\left[\begin{array}{c}
 y_t\\\alpha_t 
\end{array}\right]$$均值为：
$$
\left[\begin{array}{c}
 z E(\alpha_{t-1})\\c+w E(\alpha_{t-1}) 
\end{array}\right]
$$
协方差矩阵是：
$$
\Sigma=\left[
 \begin{matrix}
   z^2E(\alpha_{t-1}-E[\alpha_{t-1}])^2+\sigma_e^2 & zwE(\alpha_{t-1}-E[\alpha_{t-1}])^2\\
   zwE(\alpha_{t-1}-E[\alpha_{t-1}])^2 & w^2E(\alpha_{t-1}-E[\alpha_{t-1}])^2+\sigma_e^2 \\
  \end{matrix} 
\right]
$$
我们可以将向量写成如下分布：
$$
\left[\begin{array}{c}
 y_t \\
 \\
 \alpha_t \\
\end{array}\right]Y_{t-1} \sim N\left(\left[\begin{array}{c}
 z a_{t-1} \\
 \\
 c+w a_{t-1} \\
\end{array}\right] ; \left[\begin{array}{cc}
 z^2p_{t-1}+\sigma_{e}^2&wzp_{t-1} \\
 \\
wzp_{t-1}&w^2p_{t-1}-\frac{(zw p_{t-1})^2}{z^2 p_{t-1}+\sigma_{e}^2}+\sigma_{u}^2 \\
\end{array}\right]\right)
$$
现在我们可以将讨论过的主要递归过程总结以下:
$$
p_t=w^2p_{t-1}-ezk_tp_{t-1}+\sigma^2_u 
$$

$$
k_t=\frac{zwp_{t-1}}{z^2p_{t-1}+\sigma^2_e}
$$

$$
\alpha_t=E(\alpha_t|y_t)=c+w\alpha_{t-1}+\frac{zwp_{t-1}}{z^2p_{t-1}+\sigma^2_e}(y_t-z\alpha_{t-1})=c+w\alpha_{t-1}+k_t(y_t-z\alpha_{t-1})
$$

$$
v_t=(y_t-z\alpha_{t-1})
$$

这些循环就是著名的卡尔曼滤波器

#### 卡尔曼滤波器的例子

考虑以下模型:

$$
y_t=\alpha_t+e_t
$$

$$
\alpha_t=9\alpha_{t-1}+u_t
$$

这里$\sigma_{e}^2=8$ ，$\sigma_{u}^2=4$ 

现在我们可以使用以下代码生成它：

```python
n = 100
import random

e = []
for _ in range(n):
    e.append(random.normalvariate(0, 0.8))
u = []
for _ in range(n):
    u.append(random.normalvariate(0, 0.4))

y = []
for i in range(n):
    y.append(0)
alpha = []
for i in range(n):
    alpha.append(0)  
y[0] = e[0]
alpha[0] = u[0]

for t in range(n-1):
    y[t+1] = alpha[t] + e[t+1]
    alpha[t+1] = 0.9 * alpha[t] + u[t+1]

t = []
for i in range(n):
    t.append(i)
import matplotlib.pyplot as plt
plt.plot(t,alpha) # line
plt.scatter(t, y) # point
plt.xlabel("t", fontsize=14)
plt.ylabel("y", fontsize=14)
plt.show()
```

![image-20200815050236150](C:\Users\ljs11\AppData\Roaming\Typora\typora-user-images\image-20200815050236150.png)

现在实现卡尔曼滤波器。

首先需要知道的是状态方程$\alpha_t$ 是一阶自回归过程，我们取初始值$a_1=0$和$p_1=\frac{\sigma_{u}^2}{1-0.81}=2.11$。假设我们知道噪音项的反差（一般不知道），我们可以使用以下代码：

```python
n = 100
sigmae = 0.8
sigmau = 0.4
w = 0.9
z = 1
a = []
p = []

for i in range(n):
    a.append(0)   
a[0] = 0

for i in range(n):
    p.append(0)   
p[0] = 2.11
    
k = []
for i in range(n):
    k.append(0) 
    
v = []
for i in range(n):
    v.append(0) 
    
for t in range(n-1):
    k[t+1] = (z*w*p[t])/((z**2)*p[t] + sigmae)
    p[t+1] = (w**2)*p[t] - w*z*k[t+1]*p[t] + sigmau
    v[t+1] = y[t+1] - z*a[t]
    a[t+1] = w*a[t] + k[t+1]*v[t+1]

t = []
for i in range(n):
    t.append(i)
import matplotlib.pyplot as plt
plt.plot(t,y,color='black',linewidth=1.0,linestyle='-') # line
plt.plot(t,alpha,color='blue',linewidth=1.0,linestyle='-.') # line
plt.plot(t,a,color='green',linewidth=1.0,linestyle='--') # line
plt.xlabel("t", fontsize=14)
plt.ylabel("y, alpha, a", fontsize=14)
plt.show()
```

![image-20200815050925307](C:\Users\ljs11\AppData\Roaming\Typora\typora-user-images\image-20200815050925307.png)

#### 练习: 

考虑以下模型:

$$
y_t=1.05*\alpha_{t-1}+e_t
$$

$$
\alpha_t=0.5+0.8\alpha_{t-1}+u_t
$$

我们提供了一个生成state-apace模型和卡尔曼滤波器的代码：

```python
def StateSpaceGen(param):# param:list[sigmae,sigmau,z,w,const]
    sigmae = param[0]
    sigmau = param[1]
    z = param[2]
    w = param[3]
    const = param[4]
    n = 100
    
    e = []
    for _ in range(n):
        e.append(random.normalvariate(0, sigmae))
    u = []
    for _ in range(n):
        u.append(random.normalvariate(0, sigmau))
    y = []
    for i in range(n):
        y.append(0)
        
    alpha = []
    for i in range(n):
        alpha.append(0)
    y[0] = e[0]
    alpha[0] = u[0]
    
    for t in range(n-1):
        y[t+1] = z*alpha[t] + e[t+1]
        alpha[t+1] = const + w*alpha[t] +u[t+1]
    
    return y,alpha

def KF(param): # param:list[sigmae,sigmau,z,w,const,y]
    sigmae = param[0]
    sigmau = param[1]
    z = param[2]
    w = param[3]
    const = param[4]
    y = param[5]
    a = []
    for i in range(n):
        a.append(0)
    p = []
    for i in range(n):
        p.append(0)
    a[0] = y[0]
    p[0] = 10000
    if w<1:
        a[0] = 0
        p[0] = sigmau/(1-w**2)
    k = []
    for i in range(n):
        k.append(0)
    v = []
    for i in range(n):
        v.append(0)
    for t in range(n-1):
        k[t+1] = (z*w*p[t])/((z**2)*p[t]+sigmae)
        p[t+1] = (w**2)*p[t] - w*z*k[t+1]*p[t] + sigmau
        v[t+1] = y[t+1] - z*a[t]
        a[t+1] = const + w*a[t] + k[t+1] +v[t+1]
    return a,v,k,p

t = []
for i in range(n):
    t.append(i)
# Let's see an example here:
random.seed(2)
result_state_space_gen = StateSpaceGen([0.5,0.1,1,0.8,0.3])
y = result_state_space_gen[0]
KF([0.5,0.1,1,0.8,0.3,y]) 
import matplotlib.pyplot as plt
plt.plot(t,y,color='black',linewidth=1.0,linestyle='-') # line
plt.plot(t,alpha,color='blue',linewidth=1.0,linestyle='-.') # line
plt.plot(t,a,color='green',linewidth=1.0,linestyle='--')
plt.xlabel("t", fontsize=14)
plt.ylabel("y, alpha, a", fontsize=14)
plt.show() 
```

![image-20200815051548957](C:\Users\ljs11\AppData\Roaming\Typora\typora-user-images\image-20200815051548957.png)



### 似然函数和模型估计

在假设我们知道噪音的方差的情况下，我们已经展示了卡尔曼滤波器的推导及其实现。那么问题来了: 为了让估计的$E(\alpha_t|Y_t)$尽可能接近 $E(\alpha_t)$，我们应该选择哪些参数？这是很重要的一点，因为在实践中我们不知道这些差异，因此我们需要做出估计。

首先，因为$y_t$ 是正态分布的线性组合，所以它也是正态分布，因此我们可以写出y的概率密度函数：
$$
Probability(y)=\frac{1}{\sqrt{(2\pi\sigma^2_y)}}exp(-\frac{1}{2}\frac{y-E(\alpha|Y)^2}{\sigma^2_y})
$$
其次，因为目标是预测$y_t$，我们寻求一组参数可以尽量将$a_t$构造为$y_t$。

现在如上文所示，已经知道$y$的方差是$\sigma_y^2=z^2 p_{t-1}+\sigma_{e}^2$，我们构造一个新变量$v_t=y_t-\alpha_{t-1}$，用来表示当我们用$\alpha_{t-1}$预测$y_t$时的误差。

可以将$y_t$的概率分布写作：

$$
Probability(y_t)=\frac{1}{\sqrt{(2\pi(z^2p_{t-1}+\sigma_e^2)}}exp(-\frac{1}{2}\frac{v_t^2}{z^2p_{t-1}+\sigma_e^2})
$$
假设观测的$y_t$是独立的，

Assuming that the observations $y_t$ are independent, 定义为似然函数的联合分布为：

$$
Likelihood=Prob(y_1)\times Prob(y_2)\times Prob(y_3)\cdots Prob(y_n)=\prod_{t=1}^{n}{Prob(y_t)}=(\prod_{t=1}^{n}{2\pi^{-\frac{1}{2}}(z^2p_{t-1}+\sigma^2_e)^{-\frac{1}{2}}})exp(-\frac{1}{2}\frac{v_t^2}{z^2p_{t-1}+\sigma_e^2}))
$$
对我们得到的可能性取对数:

$$
logL=-\frac{n}{2}log(2\pi)-\frac{1}{2}\sum log(z^2p_{t-1}+\sigma^2_e)-\frac{1}{2}\sum \frac{v_t^2}{(z^2p_{t-1}+\sigma^2_e)}
$$
因此，我们要选的使 $E(\alpha_t|Y_t)$尽可能接近 $y_t$ 的参数就是使得logL最大的参数

下面的模块供了一个示例:

```python
n = 100
su = 0.05
se = 0.5
u = []

for _ in range(n):
    e.append(random.normalvariate(0, se))
for _ in range(n):
    u.append(random.normalvariate(0, su))
z = 1
wreal = 0.86
const = 0.6
y = []
for i in range(n):
    y.append(0)
alpha = []
for i in range(n):
    alpha.append(0)
y[0] = const + e[0]
alpha[0] = const +u[0]
for t in range(n-1):
    y[t+1] = z*alpha[t] +e[t+1]
    alpha[t+1] = const +wreal*alpha[t] + u[t+1]
##### standard Kalman filter approach######
import math
a = []
for i in range(n):
    a.append(0)
p = []
for i in range(n):
    p.append(0)
a[0] = 0
p[0] = 10
k = []
for i in range(n):
    k.append(0)
v = []
for i in range(n):
    v.append(0)
def fu(mypa):
    w = abs(mypa[0])
    se = abs(mypa[1])
    su = abs(mypa[2])
    co = abs(mypa[3])
    z = 1
    likelihood = 0
    for t in range(n-1):
        k[t+1] = (z*w*p[t])/((z**2)*p[t] + se)
        p[t+1] = (w**2)*p[t] - w*z*k[t+1]*p[t] +su 
        v[t+1] = y[t+1] - z*a[t]
        a[t+1] = co + w*a[t] + k[t+1]*v[t+1]
        likelihood = likelihood+.5*(math.log(2*(math.pi)))+.5*(math.log((z**2)*p[t-1]+se)) +.5*((v[t]**2)/((z**2)*p[t-1]+se)) 

    #print(likelihood)
    return likelihood

import numpy as np
from scipy.optimize import minimize
results = minimize(fu,np.array([0.85,0.5,0.3,0.3]))
print("The results of the standard KF approach")
print(results)
print("the true parameters")
print("wreal:",wreal,"se:",se,"su:",su,"const:",const)
import matplotlib.pyplot as plt
t = []
for i in range(n):
    t.append(i)
plt.plot(t,y,color='red',linewidth=1.0,linestyle='-') # line
plt.xlabel("t", fontsize=14)
plt.ylabel("y, alpha, a", fontsize=14)
plt.show() 
```

![image-20200815060048636](C:\Users\ljs11\AppData\Roaming\Typora\typora-user-images\image-20200815060048636.png)



#### 初始化卡尔曼滤波器的递归

#### 集中对数可能性

下面的代码生成一个模型，并使用集中对数似然估计：:

```python
import math
n = 100
su = 0.1
se = 0.4
qreal = su/se
e = []
for _ in range(n):
    e.append(random.normalvariate(0, se))
u = []
for _ in range(n):
    u.append(random.normalvariate(0, su))
z = 1
wreal = 0.97
y = []
for i in range(n):
    y.append(0)
alpha = []
for i in range(n):
    alpha.append(0)
y[0] = e[0]
alpha[0] = u[0]
for t in range(n-1):
    y[t+1] = z*alpha[t] +e[t+1]
    alpha[t+1] = wreal*alpha[t] + u[t+1]
##### standard Kalman filter approach######
import math
a = []
for i in range(n):
    a.append(0)
p = []
for i in range(n):
    p.append(0)
a[0] = 0
p[0] = 10
k = []
for i in range(n):
    k.append(0)
v = []
for i in range(n):
    v.append(0)
def fu(mypa):
    w = abs(mypa[0])
    q = abs(mypa[1])
    z = 1
    likelihood = 0
    sigmae = 0
    for t in range(n-1):
        k[t+1] = (z*w*p[t])/((z**2)*p[t] + 1)
        p[t+1] = (w**2)*p[t] - w*z*k[t+1]*p[t] +q
        v[t+1] = y[t+1] - z*a[t]
        a[t+1] = w*a[t] + k[t+1]*v[t+1]
        sigmae = sigmae + (v[t+1]**2)/((z**2)*p[t]+1)
        likelihood = likelihood+0.5*(math.log(2*math.pi))+0.5+0.5*(math.log((z**2)*p[t]+1))
    
    return likelihood+0.5*n*(math.log(sigmae/n)) 

from scipy.optimize import minimize
import numpy as np
results = minimize(fu,np.array([0.85,0.5]))
print("The results of the standard KF approach")
print(results)
print("the true parameters")
print("wreal:",wreal,"qreal",qreal)
```

```
The results of the standard KF approach
      fun: 58.571144212506425
 hess_inv: array([[ 0.00562455, -0.0214509 ],
       [-0.0214509 ,  0.15927138]])
      jac: array([-1.14440918e-05,  1.90734863e-06])
     nfev: 336
      nit: 7
     njev: 81
   status: 2
  success: False
        x: array([0.89063022, 0.43716835])
the true parameters
wreal: 0.97 qreal 0.25
```

### 运行中的State-Space模型和卡尔曼滤波器

假设我们要生成以下状态空间模型：

$$
y_t=\alpha_{t-1}=e_t
$$

$$
\alpha_t=0.2+0.85\alpha_{t-1}+u_t
$$

此时 $e \sim Normal(\mu_e=0;\sigma_{e}^2=.1)$ ， $u \sim Normal(\mu_u=0;\sigma_{u}^2=.05)$ 且 $n=100$.

另外，假设我们要估计模型的参数。：

```python
n = 100
e = []
for _ in range(n):
    e.append(random.normalvariate(0, 0.1))
u = []
for _ in range(n):
    u.append(random.normalvariate(0, 0.05))
constant = 0.2
y = []
alpha = []
for i in range(n):
    y.append(0)
for i in range(n):
    alpha.append(0)
y[0] = e[0]
alpha[0] = u[0]
for t in range(n-1):
    y[t+1] = alpha[t] + e[t+1]
    alpha[t+1] = constant + 0.85*alpha[t] +u[t+1]

t = []
for i in range(n):
    t.append(i)
import matplotlib.pyplot as plt
plt.plot(t,alpha) # line
plt.scatter(t, y) # point
plt.xlabel("t", fontsize=14)
plt.ylabel("y", fontsize=14)
plt.show()
##### standard Kalman filter approach######
import math
a = []
for i in range(n):
    a.append(0)
p = []
for i in range(n):
    p.append(0)
a[0] = 0
p[0] = 1
k = []
for i in range(n):
    k.append(0)
v = []
for i in range(n):
    v.append(0)
z = 1
def fu(mypa):
    w = abs(mypa[0])
    q = abs(mypa[1])
    co = abs(mypa[2])
    
    likelihood = 0
    sigmae = 0
    for t in range(n-1):
        k[t+1] = (z*w*p[t])/((z**2)*p[t] + 1)
        p[t+1] = (w**2)*p[t] - w*z*k[t+1]*p[t] +q
        v[t+1] = y[t+1] - z*a[t]
        a[t+1] = co + w*a[t] + k[t+1]*v[t+1]
        sigmae = sigmae + (v[t+1]**2)/((z**2)*p[t]+1)
        likelihood = likelihood+0.5*(math.log(2*(math.pi)))+0.5+0.5*(math.log((z**2)*p[t]+1))
    
    return likelihood+0.5*n*(math.log(sigmae/n))

from scipy.optimize import minimize
results = minimize(fu,np.array([0.9,1,0.1]))
print(results.x)
v[0] = 0
w = abs(results.x[0])
q = abs(results.x[1])
co = abs(results.x[2])
likelihood = 0
sigmae = 0

for t in range(len(y)-1):
    k[t+1] = (z*w*p[t])/((z**2)*p[t] + 1)
    p[t+1] = (w**2)*p[t] - w*z*k[t+1]*p[t] +q
    v[t+1] = y[t+1] - z*a[t]
    a[t+1] = co + w*a[t] + k[t+1]*v[t+1]
    sigmae = sigmae + (v[t+1]**2)/((z**2)*p[t]+1)
    likelihood = likelihood+0.5*(math.log(2*(math.pi)))+0.5+0.5*(math.log((z**2)*p[t]+1))
likelihood = likelihood+0.5*n*(math.log(sigmae/n))
sigmae = sigmae/len(y)
sigmau = q*sigmae
print(co,w,z,sigmae,sigmau)
```

![image-20200815060627247](C:\Users\ljs11\AppData\Roaming\Typora\typora-user-images\image-20200815060627247.png)

```
[0.83008814 0.74292855 0.22696502]
co w z sigmae sigmau
0.2269650202309261 0.8300881355640696 1 0.005538350255899034 0.004114598511739415
```

#### 局部水平模型（或简单的指数平滑）

局部水平模型是最简单的State Space模型之一。 这个模型假设$w=z=1$ 且$constant=0$, 因此我们可以得到: 

$$
y_t=\alpha_{t-1}+e_t
$$

$$
\alpha_t=\alpha_{t-1}+u_t
$$

对于一个结果只有一个参数可以估计。 这是使集中对数似然函数最大化的唯一成分

例如，假设我们要生成以下State Space模型：

当 $e \sim Normal(\mu=0;\sigma_{e}^2=.5)$ ， $u \sim Normal(\mu=0;\sigma_{u}^2=.2)$ 且 $n=100$， q是0.4. 

```python
random.seed(153)
n = 100
e = []
for _ in range(n):
    e.append(random.normalvariate(0, 0.5))
u = []
for _ in range(n):
    u.append(random.normalvariate(0, 0.2))

y = []
alpha = []
for i in range(n):
    y.append(0)
for i in range(n):
    alpha.append(0)
y[0] = e[0]
alpha[0] = u[0]
for t in range(n-1):
    y[t+1] = alpha[t] + e[t+1]
    alpha[t+1] = alpha[t] +u[t+1]

t = []
for i in range(n):
    t.append(i)
import matplotlib.pyplot as plt
plt.plot(t,y,color='black',linewidth=1.0,linestyle='-') # line
plt.plot(t,alpha,color='black',linewidth=1.0,linestyle='-.') # line
plt.xlabel("t", fontsize=14)
plt.ylabel("y", fontsize=14)
plt.show()
```

![image-20200815061237175](C:\Users\ljs11\AppData\Roaming\Typora\typora-user-images\image-20200815061237175.png)



现在，我们可以使用KF递归来估算此模型。 下面的代码使用4次迭代估算模型：

```python
##### standard Kalman filter approach######
import math
a = []
for i in range(n):
    a.append(0)
p = []
for i in range(n):
    p.append(0)
a[0] = y[0]
p[0] = 10000
k = []
for i in range(n):
    k.append(0)
v = []
for i in range(n):
    v.append(0)

def fu(mypa):
    q = abs(mypa)
    z = 1
    w = 1  
    likelihood = 0
    sigmae = 0
    for t in range(n-1):
        k[t+1] = (z*w*p[t])/((z**2)*p[t] + 1)
        p[t+1] = (w**2)*p[t] - w*z*k[t+1]*p[t] +q
        v[t+1] = y[t+1] - z*a[t]
        a[t+1] = w*a[t] + k[t+1]*v[t+1]
        sigmae = sigmae + (v[t+1]**2)/((z**2)*p[t]+1)
        likelihood = likelihood+0.5*(math.log(2*(math.pi)))+0.5+0.5*(math.log((z**2)*p[t]+1))   
    return likelihood+0.5*n*(math.log(sigmae/n))

from scipy.optimize import minimize
results = minimize(fu,np.array([0.2]))
print(results.x) # [0.21282871]
# We now derive the estimates of the parameters (the two variances)
z = 1
w = 1 
q = results.x[0]
sigmae = 0
for t in range(n-1):
    k[t+1] = (z*w*p[t])/((z**2)*p[t] + 1)
    p[t+1] = (w**2)*p[t] - w*z*k[t+1]*p[t] +q
    v[t+1] = y[t+1] - z*a[t]
    a[t+1] = w*a[t] + k[t+1]*v[t+1]
    sigmae = sigmae + (v[t+1]**2)/((z**2)*p[t]+1)

#This is the variance of e
m = sigmae/(n-1)
#This is the variance of u
n = q*(sigmae/(n-1))
```

```
0.22781976522533484
0.048486587689492364
```



#### 存在漂移的局部水平模型: 

此模型是local level的简单变体，但由于它在与其他强大的模型的比较中效果非常好，因此在预测人员中非常受欢迎。

例如，假设我们要 $e \sim Normal(\mu=0;\sigma_{e}^2=.8)$ ，$u \sim Normal(\mu=0;\sigma_{u}^2=.1)$ 且$constant=.1$,  $n=100$. 这里 q 是 0.125。 这段代码产生带有漂移的局部水平模型：

```python
import numpy as np
n = 100
e = []
for _ in range(n):
    e.append(random.normalvariate(0, 0.8))
u = []
for _ in range(n):
    u.append(random.normalvariate(0, 0.1))

y = []
alpha = []
for i in range(n):
    y.append(0)
for i in range(n):
    alpha.append(0)
co = 0.1
y[0] = e[0]
alpha[0] = u[0]
for t in range(n-1):
    alpha[t+1] = co + alpha[t] +u[t+1]
    y[t+1] = alpha[t] + e[t+1]

t = []
for i in range(n):
    t.append(i)
import matplotlib.pyplot as plt
plt.plot(t,y,color='black',linewidth=1.0,linestyle='-') # line
plt.plot(t,alpha,color='black',linewidth=1.0,linestyle='-.') # line
plt.xlabel("t", fontsize=14)
plt.ylabel("y", fontsize=14)
plt.show()
```

![image-20200815061855288](C:\Users\ljs11\AppData\Roaming\Typora\typora-user-images\image-20200815061855288.png)

实线是我们观测的数据$y_t$，虚线是我们拟合的数据$\alpha_t$。



现在我们使用卡尔曼滤波器估计最大对数似然估计：

```python
##### Kalman filter ######
import math
a = []
for i in range(n):
    a.append(0)
p = []
for i in range(n):
    p.append(0)
a[0] = y[0]
p[0] = 10000
k = []
for i in range(n):
    k.append(0)
v = []
for i in range(n):
    v.append(0)
v[0] =0
def funcTheta(parameters):
    q = abs(parameters[0])
    co = abs(parameters[1])
    z = 1
    w = 1
    likelihood = 0
    sigmae = 0
    for t in range(n-1):
        k[t+1] = (z*w*p[t])/((z**2)*p[t] + 1)
        p[t+1] = (w**2)*p[t] - w*z*k[t+1]*p[t] +q
        v[t+1] = y[t+1] - z*a[t]
        a[t+1] = co + w*a[t] + k[t+1]*v[t+1]
        sigmae = sigmae + (v[t+1]**2)/((z**2)*p[t]+1)
        likelihood = likelihood+0.5*(math.log(2*(math.pi)))+0.5+0.5*(math.log((z**2)*p[t]+1))
    return likelihood + 0.5*n*(math.log(sigmae/n))

from scipy.optimize import minimize
results = minimize(funcTheta,np.array([0.6,0.2]))
print(results.x) # [-0.0293218  0.079177 ]
q = abs(results.x[0])
co = abs(results.x[1])
z = 1
w = 1
sigmae = 0
for t in range(n-1):
    k[t+1] = (z*w*p[t])/((z**2)*p[t]+1)
    p[t+1] = (w**2)*p[t] - w*z*k[t+1]*p[t]+q
    v[t+1] = y[t+1]-z*a[t]
    a[t+1] = co + w*a[t] + k[t+1]*v[t+1]
    sigmae = sigmae + (v[t+1]**2)/((z**2)*p[t]+1)

sigmae/(n-1)    
q*(sigmae/(n-1))
```

```
[-0.0293218  0.079177 ]
0.46398790406283147
0.013604961267647838
```

假设我们要创建一个生成时间序列模型的函数，以及一个使用稳态方法（例如Theta模型）估算模型并返回参数估算值的函数

```python
def generateTheta(n,sigmae,sigmau,co):
    e = []
    for _ in range(n):
        e.append(random.normalvariate(0, 0.8))
    u = []
    for _ in range(n):
	    u.append(random.normalvariate(0, 0.1))
    y = []
    alpha = []
    for i in range(n):
	    y.append(0)
    for i in range(n):
	    alpha.append(0)
    y[0] = e[0]
    alpha[0] = u[0]
    for t in range(n-1):
        alpha[t+1] = co + alpha[t] + u[t+1]
        y[t+1] = alpha[t] +e[t+1]
    return y
```

```python
def EstimateTheta(y):
    n = len(y)
    a = []
    for i in range(n):
        a.append(0)
    p = []
    for i in range(n):
        p.append(0)
    a[0] = 0
    p[0] = 0
    k = []
    for i in range(n):
	    k.append(0)
    v = []
    for i in range(n):
	    v.append(0)

    def fu(mypa):
        q = abs(mypa[0])
        co = abs(mypa[1])
        z = 1
        w = 1  
        likelihood = 0
        sigmae = 0
        for t in range(n-1):
            k[t+1] = (z*w*p[t])/((z**2)*p[t] + 1)
            p[t+1] = (w**2)*p[t] - w*z*k[t+1]*p[t] +q
            v[t+1] = y[t+1] - z*a[t]
            a[t+1] = co + w*a[t] + k[t+1]*v[t+1]
            sigmae = sigmae + (v[t+1]**2)/((z**2)*p[t]+1)
            likelihood = likelihood+0.5*(math.log(2*(math.pi)))+0.5+0.5*(math.log((z**2)*p[t]+1))
        return likelihood+0.5*n*(math.log(sigmae/n))
    
    results = minimize(fu,np.array([0.5,0.2]))
    print(results.x)

    v[0] = 0
    w = 1
    z = 1
    q = abs(results.x[0])
    co = abs(results.x[1])
    sigmae = 0
    for t in range(n-1):
        k[t+1] = (z*w*p[t])/((z**2)*p[t]+1)
        p[t+1] = (w**2)*p[t] - w*z*k[t+1]*p[t]+q
        v[t+1] = y[t+1]-z*a[t]
        a[t+1] = co + w*a[t] + k[t+1]*v[t+1]
        sigmae = sigmae + (v[t+1]**2)
    sigmae = sigmae/len(y)
    sigmau = q*sigmae
    thelist = [sigmae,sigmau,co,a,v]
    return thelist
EstimateTheta(generateTheta(100,0.6,0.2,1))
```

### 单一错误来源

在过去的二十年中，文献仅将注意力集中在只有一个错误源的State Space模型上。



#### 一种误差源的指数平滑

让我们看看错误的唯一根源是实践。 下面我们生成一个指数平滑模型：

```python
n = 100
e = []
for _ in range(n):
    e.append(random.normalvariate(0, 0.6))
gamma = 0.3
y = []
alpha = []
for i in range(n):
    y.append(0)
for i in range(n):
    alpha.append(0)
y[0] = e[0]
alpha[0] = e[0]
for t in range(n-1):
    y[t+1] = alpha[t] + e[t+1]
    alpha[t+1] = co + alpha[t] +gamma*e[t+1]
    
t = []
for i in range(n):
    t.append(i)
import matplotlib.pyplot as plt
plt.plot(t,y,color='black',linewidth=1.0,linestyle='-') # line
plt.plot(t,alpha,color='black',linewidth=1.0,linestyle='-.') # line
plt.xlabel("t", fontsize=14)
plt.ylabel("y", fontsize=14)
plt.show()

```

![image-20200815062449201](C:\Users\ljs11\AppData\Roaming\Typora\typora-user-images\image-20200815062449201.png)

```python
# We can now estimate this model with the two recursions.
a = []
for i in range(n):
    a.append(0)
a[0] = y[0]
e = []
for i in range(len(y)):
    e.append(0)
    
    
def square(num):
    return num*num


def fu(mypa):
    gamma = abs(mypa[0])
    co = abs(mypa[1])
    for t in range(n-1):
	    e[t+1] = y[t+1] - z*a[t]
	    a[t+1] = co + a[t] +gamma*e[t+1]
    return (sum(list(map(square,e))))/n

from scipy.optimize import minimize
results = minimize(fu,np.array([0.6,0.2]))
print(results.x) # 0.3196408
```

#### 有一个错误源的Theta方法

让我们看看实践中的单一错误源。 下面我生成一个指数平滑模型：

```python
import random
import numpy as np
# The Theta method with one source of error
n = 100
e = []
for _ in range(n):
    e.append(random.normalvariate(0, 0.4))
gamma = 0.1
con = 0.05
y = []
alpha = []
for i in range(n):
    y.append(0)
for i in range(n):
    alpha.append(0)
y[0] = e[0]
alpha[0] = e[0]
for t in range(n-1):
    y[t+1] = alpha[t] + e[t+1]
    alpha[t+1] = con + alpha[t] +gamma*e[t+1]
t= []
for i in range(n):
    t.append(i)
import matplotlib.pyplot as plt
plt.plot(t,y,color='black',linewidth=1.0,linestyle='-') # line
plt.plot(t,alpha,color='black',linewidth=1.0,linestyle='-.') # line
plt.xlabel("t", fontsize=14)
plt.ylabel("y", fontsize=14)
plt.show()  
```

![image-20200815062552304](C:\Users\ljs11\AppData\Roaming\Typora\typora-user-images\image-20200815062552304.png)

现在，我们对该模型迭代两次。

```python
a = []
for i in range(n):
    a.append(0)
a[0] = y[0]
e = []
for i in range(len(y)):
    e.append(0)
def square(num):
    return num*num
def fu(mypa):
    z = 1
    gamma = abs(mypa[0])
    co = abs(mypa[1])
    for t in range(n-1):
        e[t+1]  = y[t+1] - z*a[t]
        a[t+1] = co+a[t] +gamma*e[t+1]
    return (sum(list(map(square,e))))/n
 
from scipy.optimize import minimize
results = minimize(fu,np.array([0.2,0.1]))
print(results.x) # [ 0.05245195 -0.04390904]
```

这种估计可以获得噪声的方差，$\gamma$和常数





### 季节性

有时数据显示出动态行为，该行为往往会根据特定的时间频率不时重复。 在这种情况下，时间序列受表示时间序列动力学不可忽略特征的季节性成分的影响。 在本节中，我们介绍了两个简单的过程，可用来从季节组件中“清理”序列。 我们首先在下面介绍加法案例，然后是乘法案例。





#### 可加季节性

假设我们有一个每年观测4次的季度序列：

```python
y = [6,2,1,3,7,3,2,4]
n = 102
e = []
for _ in range(n):
    e.append(random.normalvariate(0, 0.5))
u = []
for _ in range(n):
    u.append(random.normalvariate(0, 0.1))
y = []
alpha = []
for i in range(n):
    y.append(0)
for i in range(n):
    alpha.append(0)
seasfactor = [5,-4,2,-3]
s = 4
seasonal = []
import math
for i in range(int(math.ceil(n/s))):
    seasonal.extend(seasfactor)
seasonal = seasonal[:102]
print(seasonal)
print(len(seasonal))

y[0] = e[0] + seasonal[0]
alpha[0] = u[0]
for t in range(n-1):
    y[t+1] = seasonal[t+1] +alpha[t] +e[t+1]
    alpha[t+1] = alpha[t] +u[t+1]

t = []
for i in range(n):
    t.append(i)
import matplotlib.pyplot as plt
plt.plot(t,y,color='black',linewidth=1.0,linestyle='-') # line
plt.plot(t,alpha,color='black',linewidth=1.0,linestyle='-.') # line
plt.xlabel("t", fontsize=14)
plt.ylabel("y", fontsize=14)
plt.show() 
```

![image-20200815063050385](C:\Users\ljs11\AppData\Roaming\Typora\typora-user-images\image-20200815063050385.png)

处理该系列的一种方法是通过使用移动平均法来去除季节性成分。

##### 取该序列的中心移动平均值，称为$CMA_t$

##### 从原始序列中减去$CMA$ ，即$residuals_t=y_t-CMA_t$

##### 对 $residuals_t$按季节求均值并获取季节因素

##### 用相应的季节性因素减去$ y_t $的元素


下一个代码运行一个示例，以使短系列反季节化。 该代码不是很智能，但是可以解决这个问题：

```python
y = [6,2,1,3,7,3,2,4]
import numpy as np
cma = []
for i in range(len(y)):
    cma.append(np.nan)  
cma[2] = (0.5*y[0]+y[1]+y[2]+y[3]+0.5*y[4])/4 
cma[3] = (0.5*y[1]+y[2]+y[3]+y[4]+0.5*y[5])/4 
cma[4] = (0.5*y[2]+y[3]+y[4]+y[5]+0.5*y[6])/4 
cma[5] = (0.5*y[3]+y[4]+y[5]+y[6]+0.5*y[7])/4
residuals = []
for i in range(len(y)):
    residuals.append(y[i] - cma[i])
print(y)
print(cma)
print(residuals)
```

```
y:         [6, 2, 1, 3, 7, 3, 2, 4]
cma:       [nan, nan, 3.125, 3.375, 3.625, 3.875, nan, nan]
residuals: [nan, nan, -2.125, -0.375, 3.375, -0.875, nan, nan]

```

```python
import numpy as np
factors = []
for i in range(4):
    current_r = []
    if residuals[0+i] == residuals[0+i]:
        current_r.append(residuals[0+i])
    if residuals[0+i+4] == residuals[0+i+4]:
        current_r.append(residuals[0+i+4])
    current_mean = np.mean(current_r)
    factors.append(current_mean)
# print(factors)
            
rep = 2
newseries = []
rep_factor = []
for i in range(rep):
    rep_factor.extend(factors)
for i in range(len(y)):
    newseries.append(y[i] - rep_factor[i])
# print(newseries)
t = []
for i in range(len(y)):
    t.append(i)
import matplotlib.pyplot as plt
plt.plot(t,y,color='black',linewidth=1.0,linestyle='-') # line
plt.plot(t,newseries,color='black',linewidth=1.0,linestyle='-.') # line
plt.xlabel("t", fontsize=14)
plt.ylabel("y and newseries", fontsize=14)
plt.show()   
```

![image-20200815063802057](C:\Users\ljs11\AppData\Roaming\Typora\typora-user-images\image-20200815063802057.png)

让我们生成一个序列，然后我们执行去除季节性的过程：

```
import random
n = 87
e = []
for _ in range(n):   
    e.append(random.normalvariate(0, 0.3))
u = []
for _ in range(n):
    u.append(random.normalvariate(0, 0.1))
y = []
for i in range(n):    
    y.append(0)

alpha = []
for i in range(n):
    alpha.append(0)

seasfactor = [5,-4,2,-3]
s = 4
import math
seasonal = []
rep = math.ceil(n/s)
for i in range(int(rep)):
    seasonal.extend(seasfactor)
seasonal = seasonal[:n]

y[0] = e[0] + seasonal[0]
alpha[0] = u[0]

for t in range(n-1):
    y[t+1] = seasonal[t+1] + alpha[t] +e[t+1]
    alpha[t+1] = alpha[t] +u[t+1]
w = []
for i in range(s+1):
    if i == 0 or i == s:
	    w.append(1/(2*s))
    else:
	    w.append(1/s)
        
cma = []
import numpy as np
for i in range(len(y)):
    cma.append(np.nan)

for g in range(len(y)-s):
    sum_ = 0
    for o in range(s):
	    sum_ = sum_ + w[o] * y[g+o]
    cma[int(g+s/2)] = sum_

residuals = []
for i in range(len(y)):
    residuals.append(y[i]-cma[i])

factors = []
for i in range(4):
    current_r = []
    if residuals[0+i] == residuals[0+i]:
        current_r.append(residuals[0+i])
    if residuals[0+i+4] == residuals[0+i+4]:
        current_r.append(residuals[0+i+4])
    current_mean = np.mean(current_r)
    factors.append(current_mean)
print(factors)

import numpy as np
mean_factors = np.mean(factors)
factors_new = []
for i in range(s):
    factors_new.append(factors[i] - mean_factors)

rep = math.ceil(n/s)
factors_rep = []
for i in range(int(rep)):
    factors_rep.extend(factors_new)
factors_rep = factors_rep[:n]
newseries = []
for i in range(n):
    newseries.append(y[i] - factors_rep[i])

t = []
for i in range(len(newseries)):
    t.append(i)
import matplotlib.pyplot as plt
plt.plot(t,newseries,color='black',linewidth=1.0,linestyle='-') # line
alpha_e = []
for i in range(len(alpha)):
    alpha_e.append(alpha[i] + e[i])
plt.plot(t,alpha_e,color='black',linewidth=1.0,linestyle='-.') # line
plt.xlabel("t", fontsize=14)
plt.ylabel("y,newseries and alpha+e", fontsize=14)
plt.show()

```

![image-20200815063847912](C:\Users\ljs11\AppData\Roaming\Typora\typora-user-images\image-20200815063847912.png)

```
factors    [4.7397614583434, -4.0486504015324245, 2.923388928056161, -3.383717811689974]
seasfactor [5, -4, 2, -3]
```

#### 乘法季节性

假设我们有一个季度序列，即每年观测4次的序列，如下所示：

```python
import random
n = 103
e = []
for _ in range(n):  
    e.append(random.normalvariate(0, 0.5))
u = []
for _ in range(n):   
    u.append(random.normalvariate(0, 0.4))
y = []
for i in range(n):   
    y.append(0)
alpha = []
for i in range(n):
    alpha.append(0)
  
seasfactor = [1.7,0.3,1.9,0.1]
import math
seasonal = []
rep = math.ceil(n/s)
for i in range(int(rep)):
    seasonal.extend(seasfactor)
seasonal = seasonal[:n]

y[0] = e[0]
alpha[0] = 5+u[0]

for t in range(n-1):
    y[t+1] = seasonal[t+1]*(alpha[t] +e[t+1])
    alpha[t+1] = alpha[t] +u[t+1]

import matplotlib.pyplot as plt
t = []
for i in range(len(y)):
    t.append(i)
plt.plot(t,y,color='black',linewidth=1.0,linestyle='-') # line
plt.plot(t,alpha,color='black',linewidth=1.0,linestyle='-.') # line
plt.xlabel("t", fontsize=14)
plt.ylabel("y and alpha", fontsize=14)
plt.show()
```

![image-20200815063929037](C:\Users\ljs11\AppData\Roaming\Typora\typora-user-images\image-20200815063929037.png)

季节性成分很明显，但与上面显示的有所不同。 在这种情况下，当序列增加时，它倾向于放大。 的确，水平乘以（未加）该因子。

此时，我们可以通过再次使用移动平均法，但以另一种方式来去除季节性成分。

##### 取该序列的中心移动平均值，称为$CMA_t$

##### 从原始序列中除以$CMA$，即$residuals_t=\frac{y_t}{CMA_t}$

##### 对 $residuals_t$按季节求均值并获取季节因素

##### 将$y_t$的元素除以相应的季节性因子

这是使用该技巧的代码（类似于上面的代码）：

```python
s= 4
n = len(y)
w = []
for i in range(s+1):
    if i == 0 or i == s:
	    w.append(1/(2*s))
    else:
	    w.append(1/s)
print(w)

cma = []
import numpy as np
for i in range(len(y)):
    cma.append(np.nan)

for g in range(len(y)-s):
    sum_ = 0
    for o in range(s):
	    sum_ = sum_ + w[o] * y[g+o]
    cma[int(g+s/2)] = sum_

residuals = []
for i in range(len(y)):
    residuals.append(y[i]/cma[i])

sfactors = []
for i in range(s):
    current_r = []
    if residuals[0+i] == residuals[0+i]:
        current_r.append(residuals[0+i])
    if residuals[0+i+s] == residuals[0+i+s]:
        current_r.append(residuals[0+i+s])
    current_mean = np.mean(current_r)
    sfactors.append(current_mean)
print(sfactors)

sum_sfactors = sum(sfactors)
for i in range(len(sfactors)):
    sfactors[i] = sfactors[i]*4/sum_sfactors

newseries = []
import math
rep = math.ceil(n/s)
for i in range(int(rep)):
    newseries.extend(sfactors)
newseries = newseries[:n]
    
new_newseries = []
for i in range(n):
    new_newseries.append(y[i]/newseries[i])
```

让我们看看它如何与上面生成的（乘性）季节性序列一起工作：

```python
n = 103
s = 4

e = []
for _ in range(n): 
    e.append(random.normalvariate(0, 0.5))
u = []
for _ in range(n): 
    u.append(random.normalvariate(0, 0.4))
y = []
for i in range(n):  
    y.append(0)
alpha = []
for i in range(n):
    alpha.append(0)
 
factor = [1.7,0.3,1.9,0.1]
import math
seasonal = []
rep = math.ceil(n/s)
for i in range(int(rep)):
    seasonal.extend(factor)
seasonal = seasonal[:n]

y[0] = e[0]
alpha[0] = 5+u[0]

for t in range(n-1):
    y[t+1] = seasonal[t]*(alpha[t] + e[t+1])
    alpha[t+1] = alpha[t] +u[t+1]

#Below I extract the seasonal component

w = []
for i in range(s+1):
    if i == 0 or i == s:
	    w.append(1/(2*s))
    else:
	    w.append(1/s)

cma = []
import numpy as np
for i in range(len(y)):
    cma.append(np.nan)


for g in range(len(y)-s):
    sum_ = 0
    for o in range(s):
	    sum_ = sum_ + w[o] * y[g+o]
    cma[int(g+s/2)] = sum_

residuals = []
for i in range(len(y)):
    residuals.append(y[i]/cma[i])

sfactors = []
for i in range(s):
    current_r = []
    if residuals[0+i] == residuals[0+i]:
        current_r.append(residuals[0+i])
    if residuals[0+i+s] == residuals[0+i+s]:
        current_r.append(residuals[0+i+s])
    current_mean = np.mean(current_r)
    sfactors.append(current_mean)
print(sfactors)

sum_sfactors = sum(sfactors)
for i in range(len(sfactors)):
    sfactors[i] = sfactors[i]*4/sum_sfactors

newseries = []
import math
rep = math.ceil(n/s)
for i in range(int(rep)):
    newseries.extend(sfactors)
newseries = newseries[:n]
    
new_newseries = []
for i in range(n):
    new_newseries.append(y[i]/newseries[i])

t = []
for i in range(len(newseries)):
    t.append(i)
import matplotlib.pyplot as plt

plt.plot(t,new_newseries,color='black',linewidth=1.0,linestyle='-') # line
alpha_e = []
for i in range(len(alpha)):
    alpha_e.append(alpha[i] + e[i])
plt.plot(t,alpha_e,color='black',linewidth=1.0,linestyle='-.') # line
plt.xlabel("t", fontsize=14)
plt.ylabel("y,newseries and alpha+e", fontsize=14)
plt.show()
print(factor)
print(sfactors)
```

![image-20200815064717866](C:\Users\ljs11\AppData\Roaming\Typora\typora-user-images\image-20200815064717866.png)

```
factor [1.7, 0.3, 1.9, 0.1]
sfactors [0.07490675990704394, 1.771608248820673, 0.22381438517333402, 1.929670606098949]

```

#### 季节性state-Space

处理季节性的另一种方法是通过考虑季节成分来修改状态空间模型。
例如，考虑具有季节性行为的SSOE框架中的Local Level模型，如下所示：
$$
y_t=\alpha_{t-s}+e_t
$$

$$
\alpha_t=\alpha_{t-s}+\gamma e_t
$$

其中s代表所考虑数据的频率（例如，每月，每季度，每周等）。 例如，假设我们观察到遵循特定动态（例如本地水平）的季度时间序列。 此外，我们还观察到，在每个特定季度，时间序列倾向于假定其值与去年同期相同。 我们可以将这个模型表示如下：
$$
y_t=\alpha_{t-4}+e_t
$$

$$
\alpha_t=\alpha_{t-4}+\gamma e_t
$$

例如，假设我们要生成以下季度Local Level模型：

此时 $e \sim Normal(\mu=0;\sigma_{e}^2=0.4)$ ， $gamma=0.3$ ， $n=100$.

```python
n = 100
e = []
for _ in range(n): 
    e.append(random.normalvariate(0, 0.4))
y = []
for i in range(n):   
    y.append(0)
alpha = []
for i in range(n):
    alpha.append(0)
s = 4
sfactor = []
for i in range(s):
    sfactor.append(np.random.uniform(0,1)*10)
y[0] = sfactor[0] + e[0]
y[1] = sfactor[1] + e[1]
y[2] = sfactor[2] + e[2]
y[3] = sfactor[3] + e[3]
alpha[0] = sfactor[0] +0.2*e[0]
alpha[1] = sfactor[1] +0.2*e[1]
alpha[2] = sfactor[2] +0.2*e[2]
alpha[3] = sfactor[3] +0.2*e[3]
for t in range(n-4):
    alpha[t+4]  = alpha[t+4-s] + 0.3*e[t+4]
    y[t+4] = alpha[t+4-s] +e[t+4]

t = []
for i in range(len(y)):
    t.append(i)
import matplotlib.pyplot as plt
plt.plot(t,y[:n],color='black',linewidth=1.0,linestyle='-') # line
plt.plot(t,alpha[:n],color='black',linewidth=1.0,linestyle='-.') # line
plt.xlabel("t", fontsize=14)
plt.ylabel("y and alpha", fontsize=14)
plt.show()
```

![image-20200815065238125](C:\Users\ljs11\AppData\Roaming\Typora\typora-user-images\image-20200815065238125.png)

好消息是，由于我们假设每个季度出现相同的因素，因此得出与上述相同的结果。 因此，我们可以估计模型如下：

```python
s = 4
state = []
for i in range(n):
    state.append(0)
e = []
for i in range(n):
    e.append(0)
state[:s] = y[:s]

def logLikConc(myparam):
    gamma = abs(myparam[0])
    for t in range(n-s):
        e[t+s] = y[t+s] - state[t]
        state[t+s] = state[t] + gamma*e[t+s]
    return gamma

sum = 0
for i in range(n-1):
    sum = sum + (e[i+1]**2)/(n-1)

from scipy.optimize import minimize
import numpy as np
myresults = minimize(logLikConc,np.array([0.4]))
print("this is the estimated gamma")
myresults.x[0]
```

假设我们要生成一个具有漂移的局部水平

```python
n = 100
e = []
for _ in range(n):   
    e.append(random.normalvariate(0, 0.4))
y = []
for i in range(n):    
    y.append(0)
alpha = []
for i in range(n):
    alpha.append(0)
s= 4
co = 0.3
sfactor = []
for i in range(s):
    sfactor.append(np.random.uniform(0,1)*10)
y[0] = sfactor[0] + e[0]
y[1] = sfactor[1] + e[1]
y[2] = sfactor[2] + e[2]
y[3] = sfactor[3] + e[3]
alpha[0] = sfactor[0] +e[0]
alpha[1] = sfactor[1] +e[1]
alpha[2] = sfactor[2] +e[2]
alpha[3] = sfactor[3] +e[3]
for t in range(n-s):
    alpha[t+4]  = co + alpha[t+s-s] + 0.1*e[t+s]
    y[t+4] = alpha[t+s-s] +e[t+s]

t = []
for i in range(n):
    t.append(i)
import matplotlib.pyplot as plt
plt.plot(t,y[:n],color='black',linewidth=1.0,linestyle='-') # line
plt.plot(t,alpha[:n],color='black',linewidth=1.0,linestyle='-.') # line
plt.xlabel("t", fontsize=14)
plt.ylabel("y and alpha", fontsize=14)
plt.show()
```

![image-20200815065352361](C:\Users\ljs11\AppData\Roaming\Typora\typora-user-images\image-20200815065352361.png)

现在，我们可以估算模型如下：

```python
s = 4
state = []
v = []
for i in range(n):
    state.append(0)
for i in range(n):
    v.append(0)
e = []
for i in range(n):
    e.append(0)
state[:s] = y[:s]

def LogLikConc(myparam):
    gamma = abs(myparam[0])
    co = abs(myparam[1])
    for t in range(n-s):
        v[t+s] = y[t+s] - state[t]
        state[t+s] = co + state[t] + gamma*e[t+s]
    return gamma

sum = 0
for i in range(n-1):
    sum = sum + (v[i+1]**2)/(n-1)

from scipy.optimize import minimize
import numpy as np
myresults = minimize(LogLikConc,np.array([0.2,0.2]))
print(myresults.x)
```



### **预测时间序列**

我们一直在讨论状态空间模型的估计，但没有讨论关键问题：我们如何预测时间序列？

状态空间模型最重要的用途之一是预测估计样本之外的时间序列。

状态变量起着至关重要的作用。 实际上，状态变量（未观察到的组件）是预测我们的数据向前迈进的关键。

例如，考虑以下Theta方法:

```python
import random
n = 105
e = []
for _ in range(n):   
    e.append(random.normalvariate(0, 0.5))
u = []
for _ in range(n):   
    u.append(random.normalvariate(0, 0.1))
y = []
for i in range(n):    
    y.append(0)
alpha = []
for i in range(n):
    alpha.append(0)
co = 0.06
y[0] = e[0]
alpha[0] = u[0]

for t in range(n-1):
    alpha[t+1] = co + alpha[t] + u[t+1]
    y[t+1] = alpha[t] + e[t+1]

t = []
for i in range(len(y)):
    t.append(i)
import matplotlib.pyplot as plt
plt.plot(t,y,color='black',linewidth=1.0,linestyle='-') # line
plt.xlabel("t", fontsize=14)
plt.ylabel("y", fontsize=14)
plt.show()
```

![image-20200815065720610](C:\Users\ljs11\AppData\Roaming\Typora\typora-user-images\image-20200815065720610.png)

假设我们知道前100个观察值，而我们不知道后5个观察值。 如果要预测它们，则需要首先估计参数，然后运行预测。

一个简单的代码如下：

```python
import math
n = 100
x = y[:n]
a = []
for i in range(n):   
    a.append(0)
p = []
for i in range(n):
    p.append(0)
a[0] = x[0]
p[0] = 10000
k = []
for i in range(n):   
    k.append(0)
v = []
for i in range(n):
    v.append(0)
v[0] = 0
def funcTheta(parameters):
    q = abs(parameters[0])
    co = abs(parameters[1])
    z = 1
    w = 1
    likelihood = 0
    sigmae = 0
    for t in range(n-1):
	    k[t+1] = (z*w*p[t])/((z**2)*p[t] + 1)
	    p[t+1] = (w**2)*p[t] - w*z*k[t+1]*p[t] + q
	    v[t+1] = x[t+1] - z*a[t]
	    a[t+1] = co +w*a[t] + k[t+1]*v[t+1]
	    sigmae = sigmae + (v[t+1]**2)/((z**2)*p[t+1] + 1)
	    likelihood = likelihood +0.5*math.log(2*math.pi) + 0.5 + 0.5*math.log((z**2)*p[t] + 1)
    return likelihood + 0.5*n*math.log(sigmae/n)

from scipy.optimize import minimize
import numpy as np
results = minimize(funcTheta,np.array([0.6,0.2]))
q = abs(results.x[0])
co = abs(results.x[1])
    
z = 1
w = 1
sigmae = 0
for t in range(n-1):
    k[t+1] = (z*w*p[t])/((z**2)*p[t] + 1)
    p[t+1] = (w**2)*p[t] - w*z*k[t+1]*p[t] + q
    v[t+1] = x[t+1] - z*a[t]
    a[t+1] = co +w*a[t] + k[t+1]*v[t+1]
    sigmae = sigmae + (v[t+1]**2)/((z**2)*p[t+1] + 1)

#This is the drift parameter
print(co) # 0.05351792116427493
#This is the variance of e
print(sigmae/(n-1)) # 0.2265097039923845
#This is the variance of u
print(q*(sigmae/(n-1))) # 0.007250851778974737

# Now we can forecast x 5-steps ahead as follows:
MyForecasts = []
for i in range(5): 
    MyForecasts.append(0)
#This is my one-step ahead for x: 
MyForecasts[0]=a[n-1] 
MyForecasts[1]=co+MyForecasts[0] 
MyForecasts[2]=co+MyForecasts[1] 
MyForecasts[3]=co+MyForecasts[2] 
MyForecasts[4]=co+MyForecasts[3] 

t = []
for i in range(len(y[100:105])):
    t.append(i)
import matplotlib.pyplot as plt
plt.plot(t,y[100:105],color='black',linewidth=1.0,linestyle='-') # line
plt.plot(t,MyForecasts,color='black',linewidth=1.0,linestyle='-.') # line
plt.xlabel("t", fontsize=14)
plt.ylabel("y and MyForecasts", fontsize=14)
plt.show()
```

![image-20200815065813176](C:\Users\ljs11\AppData\Roaming\Typora\typora-user-images\image-20200815065813176.png)

### 预测季节性序列

现在考虑受季节因素影响的系列的情况。 假设我们希望预测该系列的最后6个观测值（考虑到我们仅知道前100个观测值）：

```python
import random
n = 1293
h = 6
e = []
for _ in range(n):   
    e.append(random.normalvariate(0, 0.4))
u = []
for _ in range(n):   
    u.append(random.normalvariate(0, 0.1))
my = []
import numpy as np
for i in range(n):
    my.append(np.nan)
alpha = []
for i in range(n):
    alpha.append(np.nan)
con = 0.03

factor = [0.3,0.9,1.3,1.5]
import math
seasonal = []
rep = math.ceil(n/4)
for i in range(int(rep)):
    seasonal.extend(factor)
seasonal = seasonal[:len(my)]

my[0] = e[0]
alpha[0] = u[0]

for t in range(n-1):
    my[t+1] = seasonal[t+1]*(alpha[t] + e[t+1])
    alpha[t+1] = con + alpha[t] +u[t+1]

yy = my[299:396]
t = []
for i in range(len(yy)):
    t.append(i)
import matplotlib.pyplot as plt
plt.plot(t,yy,color='black',linewidth=1.0,linestyle='-') # line
plt.xlabel("t", fontsize=14)
plt.ylabel("yy", fontsize=14)
plt.show()
```

![image-20200815065851989](C:\Users\ljs11\AppData\Roaming\Typora\typora-user-images\image-20200815065851989.png)

对于这种类型的系列，预测过程如下：

（1）应用上面讨论的加性或乘性季节性调整过程

（2）用您希望的方法预测非季节性序列

（3）在预测中添加或乘以季节性因子 获得的值。

请记住，相加或相乘过程取决于我们观察到的季节性类型。 如果在系列水平增加时季节性成分被放大，我们将使用乘法程序。 另一方面，如果季节性成分是恒定的，并且在序列增加时没有增加，则使用加法。

在此示例中，我们假设我们知道前100个观察值，而剩下的6个则作为预测样本。

```python
y = yy[:len(yy) - h]
s = 4
n = len(y)
w = []
for i in range(s+1):
    if i == 0 or i == s:
	    w.append(1/(2*s))
    else:
	    w.append(1/s)

cma = []
import numpy as np
for i in range(len(y)):
    cma.append(np.nan)

for g in range(len(y)-s):
    sum_ = 0
    for o in range(s):
	    sum_ = sum_ + w[o] * y[g+o]
    cma[int(g+s/2)] = sum_

residuals = []
for i in range(len(y)):
    residuals.append(y[i]/cma[i])

sfactors = []
for i in range(s):
    current_r = []
    if residuals[0+i] == residuals[0+i]:
        current_r.append(residuals[0+i])
    if residuals[0+i+s] == residuals[0+i+s]:
        current_r.append(residuals[0+i+s])
    current_mean = np.mean(current_r)
    sfactors.append(current_mean)
print(sfactors)

sum_sfactors = sum(sfactors)
for i in range(len(sfactors)):
    sfactors[i] = sfactors[i]*s/sum_sfactors 
newseries = []
import math
rep = math.ceil(n/s)
for i in range(int(rep)):
    newseries.extend(sfactors)
newseries = newseries[:n]
    
new_newseries = []
for i in range(n):
    new_newseries.append(y[i]/newseries[i])
```

现在，我们可以预测新近淡化的序列，然后将预测值乘以因子以获得季节性预测。 下面的代码使窍门：

```python
a = []
for i in range(n):
    a.append(0)
p = []
for i in range(n):
    p.append(0)
a[0] = new_newseries[0]
p[0] = 10000

k = []
for i in range(n):
   k.append(0)
v = []
for i in range(n):
    v.append(0)

v[0] = 0
count = len(new_newseries)
def funcTheta(parameters):
    q = abs(parameters[0])
    co = abs(parameters[1])
    z = 1
    w = 1
    likelihood = 0
    sigmae = 0
    count = len(new_newseries)
    for t in range(count-1):
	    k[t+1] = (z*w*p[t])/((z**2)*p[t] + 1)
	    p[t+1] = (w**2)*p[t] - w*z*k[t+1]*p[t] + q
	    v[t+1] = new_newseries[t+1] - z*a[t]
	    a[t+1] = co +w*a[t] + k[t+1]*v[t+1]
	    sigmae = sigmae + (v[t+1]**2)/((z**2)*p[t] + 1)
	    likelihood = likelihood +0.5*math.log(2*math.pi) + 0.5 + 0.5*math.log((z**2)*p[t] + 1)
    return likelihood + 0.5*n*math.log(sigmae/count)

from scipy.optimize import minimize
import numpy as np
results = minimize(funcTheta,np.array([0.6,0.2]))
q = abs(results.x[0])
co = abs(results.x[1])
    
z = 1
w = 1
sigmae = 0
for t in range(count-1):
    k[t+1] = (z*w*p[t])/((z**2)*p[t] + 1)
    p[t+1] = (w**2)*p[t] - w*z*k[t+1]*p[t] + q
    v[t+1] = new_newseries[t+1] - z*a[t]
    a[t+1] = co +w*a[t] + k[t+1]*v[t+1]
    sigmae = sigmae + (v[t+1]**2)/((z**2)*p[t+1] + 1)

#This is the drift parameter
print(co)

#This is the variance of e
print(sigmae/(count-1))

#This is the variance of u
print(q*(sigmae/(count-1)))
```

```
0.04849822112420224
0.8609678254959053
1.9716381358265955e-09
```

现在我们可以预测x提前6步，然后将因子乘以样本外

```python
rep = math.ceil(n+h/s)
sfactnh = []
for i in range(int(rep)):
    sfactnh.extend(sfactors)
sfactnh = sfactnh[:(n+h)]
sfactout = sfactnh[(len(sfactnh)- h): (n+h)]

w = 1
z = 1

MyForecasts = []
for i in range(h):
   MyForecasts.append(0)

MyForecasts[0] = a[len(new_newseries)-1]
for o in range(h-1):
    MyForecasts[o+1] = co + MyForecasts[o]

SeasonalForecast = []
for i in range(len(sfactout)):
    SeasonalForecast.append(MyForecasts[i]*sfactout[i]) 

t = []
for i in range(len(SeasonalForecast)):
    t.append(i)
import matplotlib.pyplot as plt
plt.plot(t,yy[(len(yy)-h):len(yy)],color='black',linewidth=1.0,linestyle='-') # line
plt.plot(t,SeasonalForecast,color='red',linewidth=1.0,linestyle='-.') # line
plt.xlabel("t", fontsize=14)
plt.ylabel("black is y_t, red is the forecasts", fontsize=14)
plt.show()
```

![image-20200815070114781](C:\Users\ljs11\AppData\Roaming\Typora\typora-user-images\image-20200815070114781.png)





### 比较预测效果

假设我们打算在预测特定时间序列时比较两个模型。 一个重要的问题是确定一个度量标准，该度量标准允许我们确定哪个模型可以为该系列提供更好的预测。 预测文献早已讨论了预测评估指标。定义 $y_{t+1}\;y_{t+2}\;\cdots\;y_{t+h}$ 是一串我们不知道的未来真实数据 ， $\hat{y}_{t+1}\;\hat{y}_{t+2}\;\cdots\;\hat{y}_{t+h}$ 是模型预测的数据. 两种流行的评估指标是所谓的均值绝对比例误差（MASE）和均值绝对百分比误差（MAPE），其计算方法如下：

![image-20200815070420591](C:\Users\ljs11\AppData\Roaming\Typora\typora-user-images\image-20200815070420591.png)

让我们举个例子。 假设我们要预测意大利伦巴第大区（受COVID影响最严重的地区）的冠状病毒新病例数。 下面我提供了从互联网下载数据的代码。

```python
import math
# Let’s make an example. Suppose we want to forecast the number of newcases of Coronavirus (Covid19) in the Italian region Lombardy (the most aﬀected region by Covid). Below I provide the code to download data from internet.
# read_csv -> y
import pandas as pd
data = pd.read_csv('covid_italy.csv')
y = list(data['x'].values)

t = []
for i in range(len(y)):
    t.append(i)
import matplotlib.pyplot as plt
plt.plot(t,y,color='black',linewidth=1.0,linestyle='-') # line
plt.xlabel("t", fontsize=14)
plt.ylabel("New Covid19 cases in Italy", fontsize=14)
plt.show()
```

![image-20200815070531784](C:\Users\ljs11\AppData\Roaming\Typora\typora-user-images\image-20200815070531784.png)

在这里，我使用第一个观测值估计Theta方法，并保留最后5个观测值：

```
obs = len(y) - 5
x = y[:obs]

a = []
for i in range(obs):
   a.append(0)
p = []
for i in range(obs):
    p.append(0)

a[0] = x[0]
p[0] = 10000

k = []
for i in range(obs):
   k.append(0)
v = []
for i in range(obs):
    v.append(0)

def funcTheta(parameters):
    q = abs(parameters[0])
    co = abs(parameters[1])
    z = 1
    w = 1
    likelihood = 0
    sigmae = 0
    
    for t in range(obs-1):
        k[t+1] = (z*w*p[t])/((z**2)*p[t] + 1)
        p[t+1] = (w**2)*p[t] - w*z*k[t+1]*p[t] + q
        v[t+1] = x[t+1] - z*a[t]
        a[t+1] = co +w*a[t] + k[t+1]*v[t+1]
        sigmae = sigmae + (v[t+1]**2)/((z**2)*p[t] + 1)
        likelihood = likelihood +0.5*math.log(2*math.pi) + 0.5 + 0.5*math.log((z**2)*p[t] + 1)
    return likelihood + 0.5*math.log(sigmae/obs)

from scipy.optimize import minimize
import numpy as np
results = minimize(funcTheta,np.array([0.6,0.2]))
q = abs(results.x[0])
co = abs(results.x[1])
    
z = 1
w = 1
sigmae = 0
for t in range(obs-1):
    k[t+1] = (z*w*p[t])/((z**2)*p[t] + 1)
    p[t+1] = (w**2)*p[t] - w*z*k[t+1]*p[t] + q
    v[t+1] = x[t+1] - z*a[t]
    a[t+1] = co +w*a[t] + k[t+1]*v[t+1]
    sigmae = sigmae + (v[t+1]**2)/((z**2)*p[t+1] + 1)

# This is the drift parameter
print(co)

# This is the variance of e
print(sigmae/len(x)-1)

# This is the variance of u
q*(sigmae/(len(x)-1))

# Here I forecast:
MyForecasts = []
for i in range(5):
    MyForecasts.append(0)
#This is my one-step ahead for x: 
MyForecasts[0]=a[obs-1] 
MyForecasts[1]=co+MyForecasts[0] 
MyForecasts[2]=co+MyForecasts[1] 
MyForecasts[3]=co+MyForecasts[2] 
MyForecasts[4]=co+MyForecasts[3] 

t = []
for i in range(5):
    t.append(i)
import matplotlib.pyplot as plt
plt.plot(t,y[(len(y)-5):len(y)],color='black',linewidth=1.0,linestyle='-')
plt.plot(t,MyForecasts,color='red',linewidth=1.0,linestyle='-.') # line
plt.xlabel("t", fontsize=14)
plt.ylabel("black is y_t, red is the forecasts", fontsize=14)
plt.show()
```

![image-20200815070556633](C:\Users\ljs11\AppData\Roaming\Typora\typora-user-images\image-20200815070556633.png)

我现在可以按以下方式计算此方法的MASE：

```python
import numpy as np
middle = []
for i in range(len(MyForecasts)):
    middle.append(abs(y[(len(y)-5):len(y)][i]-MyForecasts[i]))
MASE = np.mean(middle)/np.mean(abs(np.diff(x)))
```

```python
v = [3,1,4,8,2] 
np.diff(v) # [ -2 3 4 -6]
```

sMAPE可以如下计算：

```python
middle = []
for i in range(len(MyForecasts)):
    middle.append(200*abs(y[(len(y)-5):len(y)][i]-MyForecasts[i]))
MAPE = np.mean(middle)/np.mean(abs(np.diff(x)))
```

现在假设我想使用SSOE假设将性能与简单的指数平滑（本地级别模型）进行比较。 我需要估计和预测系列：

```python
a = []
for i in range(obs):
    a.append(0)
a[0] = x[0]

def LogLikConc(myparam):
    gamma = abs(myparam)
    for t in range(obs-1):
        v[t+1] = x[t+1] - z*a[t]
        a[t+1] = co + w*a[t] + gamma*v[t+1]
    return gamma
        

sum_ = 0
for i in range(obs-1):
    sum_ = sum_ + (v[i+1]**2)

from scipy.optimize import minimize
import numpy as np
myresults = minimize(LogLikConc,np.array([0.2]))

w = 1
z = 1

a = []
for i in range(obs):
    a.append(0)
v = []
for i in range(obs):
    v.append(0)

a[0] = x[0]
gamma = myresults.x[0]
for t in range(obs-1):
    v[t+1] = x[t+1] - z*a[t]
    a[t+1] = a[t] + gamma*v[t+1]

LLForecasts = []
for i in range(5):
    LLForecasts.append(0)
#This is my one-step ahead for x: 
LLForecasts[0]=a[obs-1] 
LLForecasts[1]=LLForecasts[0] 
LLForecasts[2]=LLForecasts[1] 
LLForecasts[3]=LLForecasts[2] 
LLForecasts[4]=LLForecasts[3]
t = []
for i in range(5):
    t.append(i)
import matplotlib.pyplot as plt
plt.plot(t,y[(len(y)-5):len(y)],color='black',linewidth=1.0,linestyle='-') # line
plt.plot(t,LLForecasts,color='red',linewidth=1.0,linestyle='-.') # line
plt.xlabel("t", fontsize=14)
plt.ylabel("black is y_t, red is the forecasts", fontsize=14)
plt.show()
```

![image-20200815070718056](C:\Users\ljs11\AppData\Roaming\Typora\typora-user-images\image-20200815070718056.png)

现在，如果我们比较MASE的结果，:

```python
import numpy as np
middle = []
for i in range(len(MyForecasts)):
    middle.append(abs(y[(len(y)-5):len(y)][i]-MyForecasts[i]))
MASETheta = np.mean(middle)/np.mean(abs(np.diff(x)))
print(MASETheta)
middle = []
for i in range(len(MyForecasts)):
    middle.append(abs(y[(len(y)-5):len(y)][i]-LLForecasts[i]))
MASELL = np.mean(middle)/np.mean(abs(np.diff(x)))
print(MASELL)
```

```
0.18174890801989835
0.11916388756963092
```



###  Forecast competion in action

假设我们有一个数据集，我们想使用上面的MASE和MAPE比较不同模型的预测性能。

下面，我们提供用于使用以下方法预测$ h_{step}$step的代码：多种错误源版本和SSOE版本中的模型1。 两个版本和阻尼趋势模型中的Theta方法都相同（仅适用于SSOE）。

#### ForecastARkf

```python
def ForecastARkf(y,h):
    n = len(y)
    a = []
    for i in range(n):
	    a.append(0)
    p = []
    for i in range(n):
	    p.append(0)
    a[0] = y[0]
    p[0] = 10000
    k = []
    for i in range(n):
	    k.append(0)
    v = []
    for i in range(n):
        v.append(0)

    def fu(mypa):
        q = abs(mypa[0])
        co = mypa[1]
        w = 1-math.exp(-abs(mypa[2]))
        likelihood = 0
        sigmae = 0
        z = 0
        for t in range(n-1):
	        k[t+1] = (z*w*p[t])/((z**2)*p[t] + 1)
	        p[t+1] = (w**2)*p[t] - w*z*k[t+1]*p[t] + q
	        v[t+1] = y[t+1] - z*a[t]
	        a[t+1] = co + w*a[t] + k[t+1]*v[t+1]
	        sigmae = sigmae + ((v[t+1]**2)/((z**2)*p[t] +1))
	        likelihood = likelihood + 0.5*math.log(2*math.pi) + 0.5 + 0.5*math.log((z**2)*p[t] +1)
        return likelihood +0.5*n*math.log(sigmae/n)

    from scipy.optimize import minimize
    import numpy as np
    results = minimize(fu,np.array([0.2,1,2]))
    v[0] = 0
    z = 1
    q = abs(results.x[0])
    co = abs(results.x[1])
    w = abs(results.x[2])
    sigmae = 0

    for t in range(n-1):
	    k[t+1] = (z*w*p[t])/((z**2)*p[t] + 1)
	    p[t+1] = (w**2)*p[t] - w*z*k[t+1]*p[t] + q
	    v[t+1] = y[t+1] - z*a[t]
	    a[t+1] = co + w*a[t] + k[t+1]*v[t+1]
	    sigmae = sigmae + ((v[t+1]**2)/((z**2)*p[t] +1))

    Forec = []
    for i in range(h):
	    Forec.append(0)
    Forec[0] = a[len(y)-1]
    for i in range(h-1):
	    Forec[i+1] = co +w*Forec[i]
    return Forec
```

#### ForecastAR

```python
def ForecastAR(y,h):
    state = []
    for i in range(len(y)):
        state.append(0)
    v = []
    for i in range(len(y)):
        v.append(0)
    state[0] = y[0]
    def LogLikConc(myparam):
        w = 1 - math.exp(-abs(myparam[0]))
        gamma = abs(myparam[1])
        co = abs(myparam[2])
        for t in range(len(y)-1):
            v[t+1] = y[t+1] - state[t]
            state[t+1] = co + w*state[t] + gamma*v[t+1]
        sum_ = 0
        for i in range(len(y)-1):
            sum_ = sum_ + v[i+1]**2
        return w,gamma,co

    from scipy.optimize import minimize
    import numpy as np
    result = minimize(LogLikConc,np.array([2,0.2,1]))
	
    w = 1-math.exp(-abs(result.x[0]))
    gamma = abs(result.x[1])
    co = abs(result.x[2])
	
    for t in range(len(y)-1):
	    v[t+1] = y[t+1] - state[t]
	    state[t+1] = co +w*state[t] + gamma*v[t+1]

    Forec = []
    for i in range(h):
	    Forec.append(0)
    Forec[0] = state[len(y)-1]
    for i in range(h-1):
	    Forec[i+1] = co + w*Forec[i]
    return Forec
```

#### ForecastTheta

```python
def ForecastTheta(y,h):
    state = []
    for i in range(len(y)):
        state.append(0)
    v = []
    for i in range(len(y)):
        v.append(0)
    state[0] = y[0] 

    def LogLikConc(myparam):
        w = 1
        gamma = abs(myparam[1])
        co = abs(myparam[2])
        for t in range(len(y)-1):
            v[t+1] = y[t+1] - state[t]
            state[t+1] = co + w*state[t] + gamma*v[t+1]
        sum = 0
        for i in range(len(y)-1):
            sum = sum + v[i+1]**2
        return w,gamma,co

    from scipy.optimize import minimize
    import numpy as np
    result = minimize(LogLikConc,np.array([0.3,1]))

    w = 1
    gamma = abs(result.x[0])
    co = abs(result.x[1])
	
    for t in range(len(y)-1):
	    v[t+1] = y[t+1] - state[t]
	    state[t+1] = co +w*state[t] + gamma*v[t+1]

    Forec = []
    for i in range(h):
	    Forec.append(0)
    Forec[0] = state[len(y)-1]
    for i in range(h-1):
	    Forec[i+1] = co + w*Forec[i]
    return Forec
```

#### ForecastThetakf

```python
def ForecastThetakf(y,h):
    n = len(y)
    a = []
    for i in range(n):
	    a.append(0)
    p = []
    for i in range(n):
        p.append(0)
    a[0] = y[0]
    p[0] = 10000
    k = []
    for i in range(n):
        k.append(0)
    v = []
    for i in range(n):
        v.append(0)
    v[0] = 0

    def funcTheta(parameters):
        q = abs(parameters[0])
        co = abs(parameters[1])
        z = 1
        w = 1
        likelihood = 0
        sigmae = 0
        for i in range(n-1):
            k[t+1] = (z*w*p[t])/((z**2)*p[t] + 1)
            p[t+1] = (w**2)*p[t] - w*z*k[t+1]*p[t] + q
            v[t+1] = y[t+1] - z*a[t]
            a[t+1] = co + w*a[t] +k[t+1]*v[t+1]
            sigmae = sigmae + ((v[t+1]**2)/((z**2)*p[t] +1))
            likelihood = likelihood + 0.5*math.log(2*math.pi) + 0.5 +0.5*math.log(z**2*p[t]+1)
        return likelihood + 0.5*n*math.log(sigmae/n)
    
    results = minimize(funcTheta,np.array([0.3,1]))
    q = abs(results.x[0])
    co = abs(results.x[1])
    z = 1
    w = 1
    for t in range(n-1):
        k[t+1] = (z*w*p[t])/((z**2)*p[t] + 1)
        p[t+1] = (w**2)*p[t] - w*z*k[t+1]*p[t] + q
        v[t+1] = y[t+1] - z*a[t]
        a[t+1] = co + w*a[t] +k[t+1]*v[t+1]
    Forecast = []
    for i in range(h):
	    Forecast.append(0)
    Forecast[0] = a[n-1]
    for i in range(h-1):
	    Forecast[i+1] = co + Forecast[i]
    return Forecast
```

#### ForecastDamped

```python
def ForecastDamped(y,h):
    obs = len(y)
    damped_one = []
    for i in range(obs):
	    damped_one.append(0)
    damped = [damped_one,damped_one]
    damped[0][0] = y[0]
    damped[1][0] = 0
    
    inn = []
    for i in range(obs):
	    inn.append(0)

    def fmsoe(param):
        k1 = abs(param[0])
        k2 = abs(param[1])
        k3 = abs(param[2])
        for t in range(obs-1):
            inn[t+1] = y[t+1] - damped[0][t] - k3*damped[1][t]
            damped[0][t+1] = damped[0][t] + k3*damped[1][t] +k1*inn[t+1]
            damped[1][t+1] = k3*damped[1][t] + k2*inn[t+1]
        sum = 0
        for i in range(obs):
            sum = sum + inn[i]**2/obs
        return k1,k2,k3
    results = minimize(fmsoe,np.array([np.random.uniform(0,1),np.random.uniform(0,1),np.random.uniform(0,1)]))
    
    k1 = abs(results.x[0])
    k2 = abs(results.x[1])
    k3 = abs(results.x[2])
    if k3>1:
	    k3 = 1

    for t in range(obs-1):
        inn[t+1] = y[t+1] - damped[0][t] -k3*damped[1][t]
        damped[0][t+1] = damped[0][t] +k3*damped[1][t] + k1*inn[t+1]
        damped[1][t+1] = k3*damped[1][t] + k2*inn[t+1]
    
    Forecast = []
    for i in range(h):
        Forecast.append(0)
    Forecast[0] = damped[0][obs] + k3*damped[1][obs]
    for i in range(h-1):
        Forecast[i+1] = Forecast[i] +damped[1][obs]*(k3**i)
    return Forecast
```

