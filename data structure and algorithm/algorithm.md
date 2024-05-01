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
```
def search(self, nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: int
    """
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] > target:
            right = mid - 1
        else:
            left = mid + 1  
    return -1
```
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
* the simplist solution is to iteratively build the permutation digit by digit, not a backtracking solution (https://www.youtube.com/watch?v=7yyNwvzO240):
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
* simplist case of backtracking
    ```

            def backtrack(seq, path, ret):
                ret.append(path)
                for i in range(len(seq)):
                    backtrack(seq[i+1:], path + [seq[i]], ret)
            
            ret = []
            backtrack(nums, [], ret)
            return ret
    ```
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
## graph
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
### 994 Rotting Orange (Graph)
* initial thought:
    * breath first search
* to move to four directions, use (1, 0) (0, 1) (-1, 0) (0, -1) to represent a move
* besides the queue for bfs, needs another map to store the distance/time laps
## Stack
### 121 Best Time to Buy and Sell Stock (sliding window, Kadane's algorithm)
* initial thought: two pointers
* method 1, too slow:
```
        currMax = 0
        for fst in range(len(prices)):
            for snd in range(fst, len(prices)):
                if fst == snd:
                    continue
                if prices[snd] < prices[fst]:
                    continue
                if prices[snd] - prices[fst] > currMax:
                    currMax = prices[snd] - prices[fst]
        return currMax
```
* Kadane's algorithm, the goal is still to find the max subarray. Here, see elements of an array as differences of two consecutive elements: https://neetcode.io/courses/advanced-algorithms/0
```
currMax, arrayMax = 0, 0
        for i in range(1, len(prices)):
            currMax = max(currMax, 0)
            currMax += prices[i] - prices[i - 1]
            arrayMax = max(currMax, arrayMax)
        
        return arrayMax
```
### 169** Majority Element (moore's algorithm)
### 57 Insert Interval
* O(n) solution makes one pass through the array, compare the start/end of the new interval with the interval i, insert if no overlap, else create a new interval based on the overlapping condition
* don't seem to fit in existing categories of common solutions
### 56 Merge Interval
* the problem is similar to the one above, but the solution looks a lot simpler
* the main point of comparison is the 2nd element of the current array in the iteration with the last array in the resulting array. In 57, the resulting array isn't involved in the comparison
* when two array merge, only change the 2nd element of the array that is inserted in the previous iteration, no need to insert a new array
* https://leetcode.com/problems/merge-intervals/solutions/350272/python3-sort-o-nlog-n
### 15 3Sum (backtracking, two pointers)
* Initial thought: can be solved with backtracking, but would exceed the time limit
    ```
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        ret = []
        def sum(arr):
            res = 0
            for i in arr:
                res += i
            return res

        def backtrack(seq, elem):
            if len(elem) == 3 and sum(elem) == 0:
                ret.append(elem)
            else:
                for i in range(len(seq)):
                    if i > 0 and seq[i] == seq[i - 1]:
                        continue
                    backtrack(seq[i+1:], elem + [seq[i]])
        
        nums.sort()
        backtrack(nums, [])
        return ret
    ```
* two pointers solution with iteration:
    ```
    def threeSum(self, nums):
            """
            :type nums: List[int]
            :rtype: List[List[int]]
            """
            res = []
            nums.sort()
            for i in range(len(nums)):
                # skip duplicate values
                if i > 0 and nums[i] == nums[i - 1]:
                    continue
                l, r = i + 1, len(nums) - 1
                while l < r:
                    s = nums[i] + nums[l] + nums[r]
                    if s == 0:
                        res.append([nums[i], nums[l], nums[r]])
                        l = l + 1
                        while l < r and nums[l] == nums[l - 1]:
                            l = l + 1
                    elif s < 0:
                        l = l + 1
                    else:
                        r = r - 1
            return res
    ```
* https://leetcode.com/problems/3sum/solutions/736561/sum-megapost-python3-solution-with-a-detailed-explanation
### 238 Product of Array Except Self (prefix sum)
    ```
    def productExceptSelf(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
   
        n = len(nums)
        ret = [None] * n   
        prefix = [None] * n
        suffix = [None] * n
            prefix[0], suffix[n - 1] = 1, 1
        for i in range(1, n):
            prefix[i] = prefix[i - 1] * nums[i - 1]
        for i in range(n - 2, -1, -1):
            suffix[i] = suffix[i + 1] * nums[i + 1]
        for i in range(n):
            ret[i] = prefix[i] * suffix[i]
        return ret
    ```
### 75 Sort Colors (dutch partitioning problem, quicksort)
* topic: invest the relationship between divid-and-conquer solution used in quicksort and the iterative solution used in this problem
```
    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        # general idea, use three index, use middle (white) to iterate index thru, if the color of the middle index is not white, then
        # swap it with the index of its own color
        white, red, blue = 0, 0, len(nums) - 1
        while white <= blue:
            # if current （white) index is red swap current and red, since it is a mismatch
            if nums[white] == 0:
                nums[white], nums[red] = nums[red], nums[white]
                white += 1
                red += 1
            # if current (white) index is white, then proceed without changes
            elif nums[white] == 1:
                white += 1
            else:
                nums[white], nums[blue] = nums[blue], nums[white]
                blue -= 1
        return nums
```
### 11 Container With Most Water (two pointers, greedy)
* instead of thinking about finding the two biggest elements in the list, which is global, start from two ends of the list and remove the smaller element from consider. This move turns the question into a local comparison, a greedy algorithm approach
* topic: elaborate on the greedy algorithm lens
## String
### 8 String to Integer (atoi)
* a bit tedius question of string parsing
* use `ord(x) - ord('0')` to convert string to int
* use deterministic finite automaton to solve the problem: https://leetcode.com/problems/string-to-integer-atoi/solutions/798380/fast-and-simpler-dfa-approach-python-3/
### 3** Longest Substring Without Repeating Characters (sliding window)
* solution:
    ```
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        if len(s) == 0:
            return 0
        start, maxLen = 0, 0
        _dict = {}
        for i in range(len(s)):
            # if this letter appears before and it is in the current sbustring (sliding window), 
            # then need to update the starting point of the substring
            if s[i] in _dict and _dict[s[i]] >= start:
                start = _dict[s[i]] + 1
            else:
                maxLen = max(maxLen, i - start + 1)
            # the update of the dictionary and the update of the sliding window is two separated processes
            _dict[s[i]] = i
        return maxLen
    ```
* https://leetcode.com/problems/longest-substring-without-repeating-characters/solutions/347818/python3-sliding-window-o-n-with-explanation/
### 242 Valid Anagram (hashtable)
* leverage `defaultdict()` to create a hashtable initialized with default values
### 76** Minimum Window Substring (sliding window)
* initial thoughts: need a function to decide whether: 1, the current window meet the condition; 2, cannot meet the condition anymore; 2, not yet meet the condition
* the most important part of the question is to design a counter that keeps track of the status of the sliding window
* https://leetcode.com/problems/minimum-window-substring/solutions/26808/here-is-a-10-line-template-that-can-solve-most-substring-problems/
### 409** Longest Palindrom
* use bitwise operation to count the occurrences of odd letters
### 5** Longest Palindromic Substring (DP, two pointers)
* this is a classic problem that has many approaches: brute force, DP, two pointers, etc
* https://leetcode.com/problems/longest-palindromic-substring/solutions/650496/all-approaches-code-in-java-including-manacher-s-algorithm-explanation/
* https://en.wikipedia.org/wiki/Longest_palindromic_substring
* walk thru all substrings, not the since the var end doesn't exist when start is len(s) - 1, it will skip
    ```
    for start in range(len(s) - 1, -1, -1):
        for end in range(start + 1, len(s)):
    ```
* topic: dp table, top down, bottom up dp approaches
### 438** Find All Anagrams in a String
* Think about it: what is the best way to check if a string is an anagram of another?
* use dictionaries to conduct the comparison: https://leetcode.com/problems/find-all-anagrams-in-a-string/solutions/175381/sliding-window-logical-thinking/
    ```
        def findAnagrams(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: List[int]
        """
        ret = []
        s_len, p_len = len(s), len(p)
        dict_s = collections.defaultdict(int)
        dict_p = collections.defaultdict(int)
        for i in p:
            dict_p[i] += 1
        right = 0
        for left in range(s_len):
            while right < s_len and right - left + 1 <= p_len:
                dict_s[s[right]] += 1
                if dict_s == dict_p:
                    ret.append(left)
                right += 1
            dict_s[s[left]] -= 1
            if dict_s[s[left]] == 0:
                del dict_s[s[left]]
        return ret
    ```
* use a counter to keep track of the comparisons: https://leetcode.com/problems/find-all-anagrams-in-a-string/solutions/92007 sliding-window-algorithm-template-to-solve-all-the-leetcode-substring-search-problem/comments/857083
## Others
### 278** First Bad Version (binary search)
* This is a binary search problem in disguise. Investigate the suble differences in how the endpoints are defined as well as the loop condition. These would depend on the starting point
* In this question, although there is no target, the function call `isBadVersion` indicates where to find the diverging point where `True` starts. Then come up with appropriate left and right pointers so that one of them will end up at the diverging point.
* Solution:
    ```
        def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """
        left, right = 1, n
        while left <= right:
            mid = left + (right - left) // 2
            if isBadVersion(mid):
                right = mid - 1
            else:
                left = mid + 1
        return left
    ```
    Here `left` and `right` are based on the actual input, which starts at 1 and ends at `n`
### 54** Spiral Matrix (simulation)
* There is no pointers involved. Instead, create four loops and increment/decrement their boundaries after each run
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
### 53** Maximum subarray (two pointers, kadane's algorithm)
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
* use `Counter()`
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
### 14** longest common prefix (string)
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
* sliding window, two pointers
* prefix sum
* top down and bottom up DP: https://stackoverflow.com/questions/6164629/what-is-the-difference-between-bottom-up-and-top-down