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
## Greedy Algorithm
## Search
### DFS
### BFS
### Binary Search
## Sort
# Data Structure
## Tree
### Binary Search Tree
# Leetcode
## 911 Online Election
* Initial thought:
    * in constructor, count the votes at each time frame; however, not sure which data struct to use since we do not know how many candidates are these in the first place
    * to find the lead, go to the closest time point and compare the votes to derive the lead
* Solution keys:
    * create a leads hash map to store the lead at each time point
    * use binary search to iterate through the leads
```
import java.util.Arrays;
class TopVotedCandidate {
    private int[] times;
    private Map<Integer, Integer> leads = new HashMap<>(); 
    public TopVotedCandidate(int[] persons, int[] times) {
        int lead = persons[0];
        this.times = times;
        Map<Integer, Integer> scores = new HashMap<>();
        for (int i = 0; i < persons.length; ++i) {
            //increment the score
            //persons[i] is the index of the person
            scores.put(persons[i], scores.getOrDefault(persons[i], 0)+1);
            //compare the current score with the lead's score, update the lead
            if (scores.get(lead) <= scores.get(persons[i])) lead = persons[i];
            leads.put(times[i], lead);
        }
        
    }
    
    public int q(int t) {
        //binary search to grab the time smaller then t and return the lead at the time
        int index = Arrays.binarySearch(times, t) < 0 ? -Arrays.binarySearch(times, t)-2 : Arrays.binarySearch(times, t);

        return leads.get(times[index]);
    }
}
 ```
## 53 Maximum subarray
* the basic idea is to keep track of current sum and maximum value separately, then update maximum by comparing maximum and current sum; these two values can be considered a sliding window
* there are two ways of track the current sum
    * none of the two comepare `currSum` with `currSum + nums[i]`, because this does not keep track of consecutive sum
    * one is to `max(nums[i], currSum + nums[i])`
```
class Solution {
    public int maxSubArray(int[] nums) {
        int currMax = nums[0];
        int max = nums[0];
        for (int i = 1; i < nums.length; ++i) {
            currMax = Math.max(nums[i], currMax + nums[i]);
            max = Math.max(currMax, max);
        }
        return max;
    }
}
```
    * the other is `max(currSum, 0)`, which is an implementation of the Kadane's algorithm: https://www.youtube.com/watch?v=umt7t1_X8Rc
```
class Solution {
    public int maxSubArray(int[] nums) {
        int currMax = nums[0];
        int sum = 0;
        for (int i = 0; i < nums.length; ++i) {
            sum += nums[i];
            currMax = Math.max(currMax, sum);
            sum = Math.max(sum, 0);
        }
        return currMax;
    }
}
```
## 128 Longest Consecutive Sequence
* Use HashSet to achieve O(1) lookup
* for each element, check if x-1 exists, if not check if x + 1, x + 2, .. exists, then update the best consequtive array length
# Resource
* Chapter 14, 15 of Margaret Fleck's textbook: Building Blocks for Theoretical Computer Science
https://mfleck.cs.illinois.edu/building-blocks/index-sp2020.html
* https://usaco.guide/general/resources-cp
* Competitive Programmer’s Handbook by Laaksonen: https://cses.fi/book/book.pdf
