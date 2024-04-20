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
## Backtracking (Erickson)
* recursion tree: The backtracking search for complete solutions is equivalent to a depth-first search of this tree.
* A backtracking algorithm tries to construct a solution to a computational problem incrementally, one small piece at a time. Whenever the algorithm needs to decide between multiple alternatives to the next component of the solution, it recursively evaluates every alternative and then chooses the best one.
* common characters for backtracking problems: 
    * goal structure is a sequence
    * each recursive step makes one decision, the the recursion requires a summary of the past decisions as its input; therefore in the middle of the recursion, the function try all possible next steps based on past decisions and decide wither these steps is acceptable
* use index instead of sub-array to represent the sequence/array
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
## backtracking
* leetcode question collection
    * https://leetcode.com/problems/letter-combinations-of-a-phone-number/solutions/780232/backtracking-python-problems-solutions-interview-prep 
    * https://leetcode.com/problems/permutations-ii/solutions/429532/general-backtracking-questions-solutions-in-python-for-reference
    * https://leetcode.com/problems/combination-sum/solutions/16502/a-general-approach-to-backtracking-questions-in-java-subsets-permutations-combination-sum-palindrome-partitioning/
* how to remove duplicate in permutations （combination instead of permutation)?
### 17** letter combinations of a phone number (backtracking)
* initial thought: how to code a permutation?
* this q can be done both recursive, an example of backtracking: https://youtu.be/gBC_Fd8EE8A?si=32mZVN18EOQuhP6O
* chapter 9 of Skiena
* complexity?
* the simplist solution is to iteratively build the permutation digit by digit (https://www.youtube.com/watch?v=7yyNwvzO240):
```
        _dict = {"2": "abc", "3": "def", "4": "ghi", "5": "jkl", "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz"}
        if not digits: 
            return []
        output = [""]
        for d in digits:
            # for each digit, look up letters
            letters = _dict[d]
            perm = []
            for l in letters:
                # for eaach letter, append that into the existing letter to the output
                for o in output:
                    newElem = o + l
                    perm.append(newElem)
            output = perm
        
        return output
```
### 79 Subset I (backtracking)
* Initially, unclear how to construct the recursion tree
* need to draw out recursion tree before deciding the base case and recursion case
### 39 combination sum (unlimited reuse) (backtracking)
* to resume the same item, pay attention to the index of the iteration to allow the same index to be called in recursion
### 40 combination sum (no reuse) (backtracking)
* sorting is necessary given the format of the output is sorted as well
* since the test cases contain duplicates, in each iteration, check if the current element is a duplicate of the prev elem
### 90 Subeset II (backtracking)
* remove duplicates
### 46 Permutations (backtracking)
* unlike the previous questions, this is a permutation, not a combination
### 47** Permutation II (backtracking)
* permutation but remove duplicate values
###
## Others
### 911 Online Election
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
### 53 Maximum subarray (two pointers)
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
### 128 Longest Consecutive Sequence (Hashmap)
* Use HashSet to achieve O(1) lookup
* for each element, check if x-1 exists, if not check if x + 1, x + 2, .. exists, then update the best consequtive array length
### 207** Course Schedule (Graph)
* Pointers: 1. construct an adjacency list to store prereq; 2. use dfs to search for cycle:
```
        adjList = defaultdict(list)
        for k, v in prerequisites:
            adjList[k].append(v)
        def cycle(courseNum, visitList):
            visitList.add(courseNum)                
            for course in adjList[courseNum]:
                if (course in visitList) or cycle (course, visitList):
                    return True
            visitList.remove(courseNum)
            return False
        
        for i in range(numCourses):
            visited = set()
            if cycle(i, visited):
                return False
        
        return True
```
* optimization: use a set to keep a record of the vertecies already visited.
```
class Solution(object):
    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        # use defaultdict to create a adjacency list, where key is course, value is the prereq
        # use dfs to go through the adjacency list, keep a record of visited courses, return false if run into a cycle
        adjList = defaultdict(list)
        for k, v in prerequisites:
            adjList[k].append(v)
        def cycle(courseNum, visitList, uniqueCourse):
            visitList.add(courseNum)                
            for course in adjList[courseNum]:
                if course in uniqueCourse:
                    continue
                elif (course in visitList) or cycle (course, visitList, uniqueCourse):
                    return True
            visitList.remove(courseNum)
            uniqueCourse.add(courseNum)
            return False
        
        courses = set() # notice the unique courses are set outside of the loop
        for i in range(numCourses):
            visited = set()
            if cycle(i, visited, courses):
                return False
        
        return True
```
* https://youtu.be/yPldqMtg-So?si=vi4WbPuqmeupZDWQ
* about recursion, variable changed at the bottom will affact the same variable that is cached in the recursion above it. eg(test case: [[0,1],[0,2],[1,3]] and print visitList)
### 2244 Min Rounds to Complete All Tasks (math)
* initial thoughts:
    * use a hash map to keep track of the number of tasks at each level
    * iterature over the hash map and calculate the number of rounds needed
* hint: determine the pattern for counting the rounds needed by enumeration
* the following solution is correct, but not efficient:
```
        taskDict = {}
        for i in tasks:
            if i in taskDict.keys():
                taskDict[i] += 1
            else:
                taskDict[i] = 1
        rounds = 0
        for j in taskDict.keys():
            if taskDict[j] == 1:
                return -1
            elif taskDict[j] % 3 == 0:
                rounds += taskDict[j] / 3
            else:
                rounds += ( taskDict[j] / 3 + 1)
        return rounds
```
* 
```
        taskDict = Counter(tasks).values()
        rounds = 0
        for j in taskDict:
            if j == 1:
                return -1
            else:
                rounds += ( (j + 2) // 3 )
        return rounds
```
* use efficient data structure andfaster
```
class Solution:
    def minimumRounds(self, tasks: List[int]) -> int:
        k = 0
        c = Counter(tasks).values()
        if 1 in c:
            return -1
        for j in c:
            k += j // 3 + bool(j % 3)
        return k
```
### 633 Sum of Square Numbers (two pointers, hashmap)
* set two pointers apart, squeeze the two points based on their sum
```
class Solution(object):
    def judgeSquareSum(self, c):
        """
        :type c: int
        :rtype: bool
        """
        left = 0
        right = math.floor(sqrt(c))
        while left <= right:
            if left * left + right * right == c:
                return True
            elif left * left + right * right < c:
                left += 1
            else:
                right -= 1
        return False
```
### 994 Rotting Orange (graph)
* initial thought:
    * breath first search
* to move to four directions, use (1, 0) (0, 1) (-1, 0) (0, -1) to represent a move
* besides the queue for bfs, needs another map to store the distance/time laps
### 14 longest common prefix (string)
* compare position i of all string at the same time, if one does not equal the others, return existing output
### 153 Find Minimum in Rotated Sorted Array （binary search)
* this question is a clever spin on binary search since the array is not sorted, 
however, the arrangement of the array is still very important
* Q: does the solution depend on the kind of array it is? Yes
* https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/solutions/158940/beat-100-very-simple-python-very-detailed-explanation

# Resource
* Chapter 14, 15 of Margaret Fleck's textbook: Building Blocks for Theoretical Computer Science
https://mfleck.cs.illinois.edu/building-blocks/index-sp2020.html
* USACO guide on Competitive Programming: https://usaco.guide/general/resources-cp
* Competitive Programmer’s Handbook by Laaksonen: https://cses.fi/book/book.pdf
* Jeff Erickson's Algorithm textbook
# Topics to be investigated:
* backtrack big O analysis
* recursion tree
* improve efficiency of backtracking methods