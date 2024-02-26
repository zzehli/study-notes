# Asymptotic Analysis
## Principles:
* ignore small inputs
* ignore multiplicative constants (since it is sensitive to details of implementation, hardware platform, et)
* in a polynomial, focus on the fastest-growing term
## Asymptotically similar and Asymptotically smaller
* $f(n) \approx g(n)$, the two are similar if $f(n) \over g(n)$ approaches constant c when n approaches infinity
* $f(n) \ll g(n)$, f(n) is smaller if $f(n) \over g(n)$ approaches 0 when n approaches infinity
## common patterns
* algos for sorting have running time that grow like $nlog(n)$
* relationship of common primitive functions
$$1 \ll log n \ll n \ll n log(n) \ll n^2$$

# big-O
Asymptotic analaysis is well suited for well-behaved functions. Since comparing program run time can be messy, we use a more relaxed relationship called big-O. This means, big-O relationship is a non-strict partial order like $\leq$ while $\ll$ is a strict partial order like <.
* The base of logs doesn't change the final big-O answer, since logs with different bases differ only by a constant multiplier, $log_2(n) = log_2(3)log_3(n)$. This is due to the log rule, change of bases.

When g(n) is $O(f(n))$ and f(n) is $O(g(n))$, then f(n) and g(n) are forced to remain close together as n goes to infinity. In this case we say that f(n) is $\Theta(g(n))$

> Computer scientists often say that $f(n)$ is $O(g(n))$ when they actually mean the stronger statement that $f(n)$ is $\Theta(g(n))$

For big-o analysis examples, such as merge sort, see 15-algorithms.pdf
# Algorithms
# Resource
* Chapter 14, 15 of Margaret Fleck's textbook: Building Blocks for Theoretical Computer Science
https://mfleck.cs.illinois.edu/building-blocks/index-sp2020.html