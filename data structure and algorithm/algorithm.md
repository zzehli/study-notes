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
## String
### Trie
A trie is a rooted tree that maintains a set of strings. Each String in the set is stored as a chain of characters that starts at the root. If two strings have a common prefix, they also have a common chain in the tree. 

Each character can be represented as a node in the chain. The node holds two properties, one to represent if this character is the end of a word; another is linked to the rest of the string via a dict (so technically this is not a chain as in linked list)

Look up or adding a string in the trie takes O(n).
```
class TrieNode:
    def __init__(self):
        self.word = False
        self.children = {}

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for i in word:
            if i not in node.children:
                node.children[i] = TrieNode()
            node = node.children[i]
        node.word = True
    
    def search(self, word: str) -> bool:
        node = self.root
        for i in word:
            if i not in node.children:
                return False
            node = node.children[i]
        return node.word
        

    def startsWith(self, prefix: str) -> bool:
        node = self.root
        for i in prefix:
            if i not in node.children:
                return False
            node = node.children[i]
        return True

```
# Leetcode
## binary
### 67** Add Binary
* keep track of carry and sum
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
### 78 Subsets I (backtracking)
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
### 90 Subesets II (backtracking)
* remove duplicates
### 46** Permutations (backtracking)
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
* pay attention to how the visited list is updated
### 994 Rotting Orange (Graph)
* initial thought:
    * breath first search
* to move to four directions, use (1, 0) (0, 1) (-1, 0) (0, -1) to represent a move
* besides the queue for bfs, needs another map to store the distance/time laps
* Why dfs doesn't work? Because the algorithm traverses from multiple starting points, only bfs would ensure that the "flood fill" spread as intended (similar to 733. Flood Fill)
### 542 01 Matrix (Graph, Matrix, BFS, DP)
* Brute force:
    ```
    def updateMatrix(self, mat):
        """
        :type mat: List[List[int]]
        :rtype: List[List[int]]
        """
        # bfs
        def bfs(x,y):
            q = collections.deque()
            q.append((x, y, 0))
            while len(q) > 0:
                m, n, d = q.popleft()
                if mat[m][n] == 0:
                    return d
                for i, j in [(m+1, n), (m, n+1), (m-1, n), (m, n-1)]:
                    if 0 <= i < len(mat) and 0 <= j < len(mat[0]):
                        q.append((i,j,d+1))
            return -1

        output = [[0] * len(mat[0]) for _ in range(len(mat))]
        for i in range(len(mat)):
            for j in range(len(mat[0])):
                output[i][j] = bfs(i,j)
        return output
    ```
* Save memory by change matrix in-place
    ```
            q = collections.deque()
        # output = [[0] * len(mat[0]) for _ in range(len(mat))]
        for i in range(len(mat)):
            for j in range(len(mat[0])):
                if mat[i][j] == 0:
                    q.append((i,j))
                else:
                    mat[i][j] = -1
        while q:
            m, n = q.popleft()
            for i, j in [(m+1, n), (m, n+1), (m-1, n), (m, n-1)]:
                if 0 <= i < len(mat) and 0 <= j < len(mat[0]) and mat[i][j] == -1:
                    mat[i][j] = mat[m][n] + 1
                    q.append((i,j))              
        return mat
    ```
* investigate DP solution
### 133** Clone Graph (Graph, DFS/BFS)
* basic problem type, can be implemented in many ways
* needs a dictionary to hold all the nodes
### 200 Number of Islands
* initial thought: dfs on islands of 1s, traverse and turn 1 to 2
### 721** Accounts merge (Graph, DFS/BFS)
* construct the graph with email as key and a list of accounts as values
* because email identifies a person, use emails to connect accounts and merge them: eg. if A has email E1 and E1 is associated with [A, B], then A and B are the same person and A should have all emails for A and B.
* to merge the emails/accounts, iterature over the list of accounts, for each email Ei in an account i, traverse the graph to collect the all accounts associated with the email and traverse the account list, Li, since these accounts belong to the same person i.
### 127 Word Ladder (BFS)
* Initial thoughts: 
    * construct all posible word lists and choose the shortest
    * how to effectively see if two words differ by 1 letter en masse?
    * could use backtracking to construct words
* "So, all this problem is asking us to do, is find a shortest path from our start word to end word using only word inside our list." (https://leetcode.com/problems/word-ladder/solutions/1764371/a-very-highly-detailed-explanation)
* an elegant solution that uses `string.ascii_lowercase` and `yield`: https://leetcode.com/problems/word-ladder/solutions/509769/java-python-bfs-solution-clean-concise/
### 79 Word Search 
* Initial thoughts
    * brute force and keep track of visited cells
    * there could be multiple starting points -> need to use bfs
    * the difficulty is that each starting point need to keep track of their track separately -> flood fill with bfs is hard to fullfil the requirements
* For iterative solution to work, need to add the state of the stack to another var: https://leetcode.com/problems/word-search/solutions/131327/iterative-python-solution/
* dfs with recursion is the most straight forward solution
### 310** Minimum Height Trees
* Initial thought:
    * make an adjacency list out of the edges
    * construct all possible trees and compare their heights
    * construct a tree is a traversal
    * but how to calculate height?
* this q is unlike the other graph questions but it feels very practical
* the simplist solution treat the adjacency list as a tree and trim leaf nodes along the way until there are only 1 or 2 nodes left
* topic: topological sort
## Array
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
### 122** Best Time to Buy and Sell Stock II
* initial thought: this seems way more difficult that the previous one
* Q: explore the relationship between different solutions: https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/solutions/803206/python-js-java-go-c-o-n-by-dp-greedy-visualization-thinking-process
### 169** Majority Element (moore's algorithm)
### 57** Insert Interval
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
### 11** Container With Most Water (two pointers, greedy)
* instead of thinking about finding the two biggest elements in the list, which is global, start from two ends of the list and remove the smaller element from consideration. This move turns the question into a local comparison, a greedy algorithm approach
* topic: elaborate on the greedy algorithm approach
### 88 Merge Sorted Array** (in-place) (two pointers)
* move from the end of the word to the beginning
### 27 Remove Element (in-place)
* similar to 26
* make two pointers, one keep track of first pass, the other keep track of the modified version
### 26 Remove Duplicates in Sorted Array (in-place)
* similar to above
### 80 Remove Duplicates II
* can be solved in the same as above, only change 1 num
* https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/solutions/27987/short-and-simple-java-solution-easy-to-understand
### 189 Rotate Array
* in-place rotation thru reversal (a bit tricky)
### 28 Find the Index of the First Occurrence in a String
* design index so that the third index is not needed
* Q: explore KMT solution
### 55** Jump Game
* Initial throughts: this is a dynamic programming question: options at each step are fixed and can be memoized
* important lesson: how to prevent the loop from exiting early before trying all options with recursive calls?
    compare:
    ```
    for j in range(1, nums[start] + 1):
        if j > end:
            break
        return rec(j + start, end)
    print('here?')
    ```
    with
    ```
    for j in range(1, nums[start] + 1):
            if j > end:
                break
            if rec(j + start, end):
                return True
        return False  
    ```
    The second solution will try all possible iterations in the `range` call before exiting. Remember, the return statement cannot be inside the loop

## Stack
### 232** Implement Queue using Stacks (stack)
* use in and out arrays to keep track of ins and outs of the array
* need a sync function to keep the two functions in sync
### 150** Evaluate Reverse Polish Notation (stack)
* initial thoughts: no idea how to approach this
* the solution is quite simple, since the arithmetic rule is straight forward, despite of its seeming variations
* the stack will keep track of all the numbers (and result of operation), when you iterate thru the expression, the operation is always performed on the top most numbers in the stack
### 155** Min Stack (stack)
* since the requirement is O(1), iterate thru the stack is not possible
* one way to go about it is to record the new minimum value with every push and retrieve the current minium from the end of the stack/array
### 42** Trapping Rain Water (stack, two pointers)
* use two pointers (l and r) and two maximum values (lmax and rmax), if the current values are bigger than the max, update the max, otherwise add to the overall area
* Similar to 11 Container with Most Water
* topic: similarities with 84. Largest Rectangle in Histogram https://leetcode.com/problems/trapping-rain-water/solutions/17414/a-stack-based-solution-for-reference-inspired-by-histogram/
### 224 Basic Calculator (stack)
* https://leetcode.com/problems/basic-calculator/solutions/1456850/python-basic-calculator-i-ii-iii-easy-solution-detailed-explanation/
### 84** Largest Rectangle in Histogram
* topic: Monostack: https://leetcode.com/problems/largest-rectangle-in-histogram/solutions/995249/python-increasing-stack-explained/
* concise monostack: * https://leetcode.com/problems/largest-rectangle-in-histogram/solutions/28917/ac-python-clean-solution-using-stack-76ms/
    ```
    class Solution(object):
    def largestRectangleArea(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        heights.append(0)
        <!-- stack holds a list of index where the heights are higher than the current height -->
        stack = [-1]
        ret = 0
        for i in range(len(heights)):
            <!-- make sure the current height is high than the ones saved in the stack -->
            while heights[i] < heights[stack[-1]]:
                h = heights[stack.pop()]
                <!-- the width is from stack[-1] (which is always taller than height[i]) (left bound) to i - 1 (right bound)  -->
                w = i - stack[-1] - 1
                ret = max(ret, h * w)
            stack.append(i)
        # heights.pop()
        return ret
    ```

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
### 49** Group Anagrams (hashtable)
* use hashmap to keep track of word counts in words, use another dict to keep track of anagrams
* https://leetcode.com/problems/group-anagrams/solutions/4683832/beats-99-users-c-java-python-javascript-2-approaches-explained
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
* leetcode editorial: https://leetcode.com/problems/longest-palindromic-substring/editorial/
### 438** Find All Anagrams in a String (sliding window, hashtable)
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
* use a counter to keep track of the comparisons: https://leetcode.com/problems/find-all-anagrams-in-a-string/solutions/92007/sliding-window-algorithm-template-to-solve-all-the-leetcode-substring-search-problem/comments/857083
### 567** Permutation in String (sliding window)
* similar to 438: https://leetcode.com/problems/permutation-in-string/solutions/559278/java-python-sliding-window-clear-explanation-clean-concise
* notice how the hashtable is incremented/decremented as the window moves: https://leetcode.com/problems/permutation-in-string/solutions/1761953/python3-sliding-window-optimized-explained
### 2273 Find Resultant Array After removing anagrams (hashtable)
* use an array to count letters in an anagram and compare the array or use tuple for hashtable
## Trie

### 208 Implement Trie (Prefix Tree)
* See notes above
### 139** Word Break (String, Trie)
* initial throughts: with a Trie that contains the dictionary, a naitive search in the Trie based on the given string won't work since the words in the dict won't always match the start of the string
* The editorial: https://leetcode.com/problems/word-break/editorial/
    * Sol1: BFS solution is easy to understand
    * Q: time complexities for BFS
    * Sol2: top-down dynamic programming is easy to implement
    ```
        def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        @cache
        def dp(i: int) -> bool:
            if i < 0:
                return True
            for word in wordDict:
                if s[i - len(word) + 1: i+1] == word and dp(i - len(word)):
                    return True
            return False

        return dp(len(s) - 1)
    ```
## Linked List
### 146** LRU Cache
* Doubly linked list + hashmap
### 21** Merge Two Sorted Lists
* the recursive solution is more intuitive: create the return list by modifying the exisiting list
### 141** Linked List Cycle
* clever solution with a fast and a slow pointer and see if the fast can catch the slow pointer
## Binary Search
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
### 153 Find Minimum in Rotated Sorted Array (binary search)
* this question is a clever spin on binary search since the array is not sorted, 
however, the arrangement of the array is still very important
* Q: does the solution depend on the kind of array it is? Yes
* https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/solutions/158940/beat-100-very-simple-python-very-detailed-explanation
### 33 Search in Rotated Sorted Array (binary search)
* Similar to 153, but more thoughts needed to find the target, since knowing the rotation point is not enough.
* One solution is to list all possible senario and instruct how to choose in each:
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
            elif nums[right] >= nums[mid]:
                if target >= nums[mid] and nums[right] >= target:
                    left = mid + 1
                else:
                    right = mid - 1
            else:
                if target <= nums[mid] and nums[left] <= target:
                    right = mid - 1
                else:
                    left = mid + 1
        return -1
    ```
### 981 Time Based Key-Value Store (binary search)
* This can be solved by a speicial case of binary search that output the biggest value smaller than the targetf:
    ```
    def get(self, key, timestamp):
    values = self.__dict[key]
    if len(values) == 0:
        return ""
    left, right = 0, len(values)
    while left < right:
        mid = right + (left - right) // 2
        if values[mid][0] == timestamp:
            return values[mid][1]
        elif values[mid][0] < timestamp:
            left = mid + 1
        else:
            right = mid
    if right == 0:
        return ""
    else:
        return values[right - 1][1]
    ```
* topic: investigate the generalization of different binary search variation
### 1235** Maximum Profit in Job Scheduling (binary search, DP, backtracking)
* Topic: knapsack problem
* the brute force solution is thru backtracking
* Solution:
    ```
        def jobScheduling(self, startTime, endTime, profit):
        """
        :type startTime: List[int]
        :type endTime: List[int]
        :type profit: List[int]
        :rtype: int
        """
        jobs = sorted(zip(startTime, endTime, profit), key = lambda v: v[1])
        dp = [[0,0]]
        for s, e, p in jobs:
            # i is not the insertion point, it element prior to the insertion point
            # we compare the the potential profit (new_profit) by adding the existing profit dp[i][1] and current profit p with max profit so far dp[-1][1]
            i = bisect.bisect(dp, [s + 1]) - 1
            new_profit = dp[i][1] + p
            if new_profit > dp[-1][1]:
                # note that this dp is the resulting job combo
                # rather, it is a lookup table of each end time and its maximum profit
                dp.append([e, dp[i][1] + p])
        return dp[-1][1]
    ```
## B-Tree/Binary Search Tree
### 230** Kth Smallest Element in a BST (Binary Search Tree)
* traverse to the leaves of the tree, decrement k for on the way back until 0
### 102** Binary Tree Level Order Traversal (Binary Tree)
* use a queue to keep track of the number of nodes in a level; for each level, append each node's children to the queue, as well as put node's value to a temporary array, then append the array to the output at the end of the level iteration
```
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root: return []
        queue = deque([root])
        res = []

        while queue:
            currLv = []
            size = len(queue)
            for i in range(size):
                node = queue.popleft()
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
                currLv.append(node.val)
            res.append(currLv)
        return res
```
### 236 Lowest Common Ancestor of a Binary Tree (Binary Tree)
* similar to 235, which is for binary search tree, but harder, since b-tree is not ordered
* unlike 235, need to traverse first then see if a node has both p and q as its children
### 297** Serialize and Deserialize Binary Tree (Binary Tree)
### 543** Diameter of Binary Tree (Binary Tree)
* common b-tree problem
* `1 + max` to calculate the height of the tree
### 199 Binary Tree Right Side View (Binary Tree)
* similar to the level order traversal, keep track of levels. Traverse the whole tree but only insert the right most element
* similar to 102, 104
### 104 Maximum Depth of Binary Tree (Binary Tree)
* try to come up with the recursive solution
* can be solved by keep track the levels as in 102
### 105** Construct Binary Tree from Preorder and Inorder Traversal
* initial thoughts: no clue how to approach this
* the recursive solution split the inorder array in half, left children take the first half and the right children take the second half
## Heap
### 973 K Closest Point to Origin (heap)
* classic heap question: https://leetcode.com/problems/k-closest-points-to-origin/solutions/294389/easy-to-read-python-min-heap-solution-beat-99-python-solutions/
### 295 Find Median from Data Stream
* initial thought: store the number in a heap and use length to find the median -> won't work because heap is not sorted
* use two heaps and balance their length with each add
### 23** Merge k Sorted Lists (recursion, heap)
* two approaches, merge recursively or use heap
* recursion: https://leetcode.com/problems/merge-k-sorted-lists/solutions/10919/python-easy-to-understand-divide-and-conquer-solution/
* heap solution: https://leetcode.com/problems/merge-k-sorted-lists/solutions/465094/problems-with-python3-and-multiple-solutions/
### 621** Task Scheduler (greedy, heap)
* two solutions, greedy and heap
* topic, research greedy solution: https://leetcode.com/problems/task-scheduler/solutions/104500/java-o-n-time-o-1-space-1-pass-no-sorting-solution-with-detailed-explanation/ and https://www.youtube.com/watch?v=jUE-W5o6lMU
* heap, not time efficient: https://leetcode.com/problems/task-scheduler/solutions/130786/python-solution-with-detailed-explanation/
### 347** Top K Frequent Element (heap, bucket sort)
* turn a counter into a priority heap, attention to the maxHeap (as opposed to the default min heap) (https://leetcode.com/problems/top-k-frequent-elements/solutions/1502514/c-python-2-solutions-maxheap-bucket-sort-clean-concise)
```
        cnt = Counter(nums)
        maxHeap = [[-freq, num] for num, freq in cnt.items()]
        heapify(maxHeap)
```
* the most efficient solution is bucket sort
### 215** Kth Largest Element in an Array (heap, sorting)
* solution using heap: create an array, heap push into the array until the length of the array is k, heap pop extra element afterwards
* https://leetcode.com/problems/kth-largest-element-in-an-array/solutions/1349609/python-4-solutions-minheap-maxheap-quickselect-clean-concise
* multiple solutions, quick select is the fastest
```
    def findKthLargest(self, nums: List[int], k: int) -> int:
        return self.quick_select(nums, len(nums) - k)
    def quick_select(self, nums, j):
        pivot = random.choice(nums)
        left, mid, right = [], [], []
        for num in nums:
            if num < pivot:
                left.append(num)
            elif num > pivot:
                right.append(num)
            else:
                mid.append(num)
        
        if j < len(left):
            return self.quick_select(left, j)
        elif j < len(left) + len(mid):
            return pivot
        return self.quick_select(right, j - len(left) - len(mid))
``` 
### 506** Relative Ranks (heap)
* distinguish rank and position, create a map between them 
## Dynamic Programming
### 53** Maximum subarray (two pointers, dp, kadane's algorithm)
* the basic idea is to keep track of current sum and maximum value separately, then update maximum by comparing maximum and current sum; these two values can be considered a sliding window
* another important insight: don't judge the new value, instead decide whether the current sum is negative or not; the rationale is that even if the current value is negative, the current sum could still be positive thus useful for the future; don't look at the value in isolation
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
* dynamic programming perspective: no recursion needed, but identify subproblem and build the answer through bottom-up approach
### 62 Unique Paths (2-d dynamic programming)
* a classic top down approach use case
* many solutions: https://leetcode.com/problems/unique-paths/solutions/1581998/c-python-5-simple-solutions-w-explanation-optimization-from-brute-force-to-dp-to-math/
* recursion -> recursion with cache (bottom up solution)
```
    #same as recursion, add cache keyword to turn it into dp
    @cache
    def rec(i, j) -> int:
        if i > n - 1 or j > m - 1:
            return 0
        if i == n - 1 and j == m - 1:
            return 1
        return rec(i + 1, j) + rec(i, j + 1)
    return rec(0, 0)

    #avoid the cache keyword, use an array of array to store results
    def uniquePaths(self, m: int, n: int) -> int:
        # brute force would be a dfs/bfs solution
        cache = [ [None]*n for _ in range(m) ]
        def dp(c, i, j):
            if i > m - 1 or j > n - 1:
                return 0
            elif i == m - 1 and j == n - 1:
                return 1
            if c[i][j]:
                return c[i][j]
            c[i][j] = dp(c, i + 1, j) + dp(c, i, j + 1)
            return c[i][j]
        return dp(cache, 0, 0)

```
* Is dynamic programming just backtracking with cache? https://stackoverflow.com/a/22919483 (DAG, cahce with data struct instead of recursion)
### 322 Coin Change (DP)
* initial thoughts: 
    * isn't clear what the recursive step is since every case is unique in its coin combintion
    * could recursively find whether a given amount is possible given the coins
* bottom up dp approach:
    ```
        def coinChange(self, coins: List[int], amount: int) -> int:
            # tabular is an array that initialized with amount + 1 as the ct (fewest number of coins) at each index amount
            rs = [amount + 1] * (amount + 1)
            # when amount is 0, ct is 0
            rs[0] = 0
            # even tho rs has amount + 1 cells, the index of the array rs only goes to amount + 1: if amount is 2, we need an array of 3 with the biggest index 2
            for i in range(1, amount + 1):
                for c in coins:
                    if i >= c:
                        # don't need to know how to build the amount from scratch, simply build it by comparing i - c for all coins value c
                        rs[i] = min(rs[i], rs[i - c] + 1)
            
            # if the ct is amount + 1, it is impossible theoretically since the biggest ct would be amount, which is amount # of 1s
            if rs[amount] == amount + 1:
                return -1
            return rs[amount]
    ```
* top down approach: https://leetcode.com/problems/coin-change/solutions/2058537/python-easy-2-dp-approaches/
* Q: explore knapsack problem: https://www.geeksforgeeks.org/unbounded-knapsack-repetition-items-allowed/
### 70 Climbing Stairs (DP)
* easy problem, after investigation, realize this is somehow just a Fibonacci sequence
### 416 Partition Equal Subset Sum (DP)
* initial thought: backtracking seems to be a straight forward solution to this
* backtracking works, but most solutions suggest the knapsack problem as a more optimized solution
* backtrack with cache solution, notice how recursion is implemented:
```
    def canPartition(self, nums: List[int]) -> bool:
        target = 0
        for i in nums:
            target += i
        if target%2:
            return False
        target /= 2
        @cache
        def backtrack(currSum, index):
            if currSum > target or index >= len(nums):
                return False
            if currSum == target:
                return True
            return backtrack(currSum + nums[index], index + 1) or backtrack(currSum, index + 1)
        
        return backtrack(0, 0)
```
* Q: understand knapsack problem
### 118 Pascal's Triangle (DP)
* the edge cases are hard to figure out: 1st, 2nd rows and 1st and last element of each row are 1s (not sum of prev row)
* create a triangle and fill it with 1s: 
`rt = [[1] * i for i in range(1, numRows + 1)]`
### 746** Min Cost Climbing Stairs (1-d DP)
* This is a classic DP that can be solved by top down and bottom up
* backtracking solution (exceeds time limit):
```
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        if len(cost) == 0:
            return 0
        if len(cost) == 1:
            return 0
        minCost = 0
        minCost += min(cost[0] + self.minCostClimbingStairs(cost[1:]), cost[1] + self.minCostClimbingStairs(cost[2:]))
        return minCost
```
* add memoization to backtracking:
```
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        l = len(cost)
        stack = [None] *l
        def dp(cost, index):
            if index == l - 1 or index == l:
                return 0
            if stack[index]:
                return stack[index]
            minCost = 0
            minCost += min(cost[index] + dp(cost, index + 1), cost[index + 1] + dp(cost, index + 2))
            stack[index] = minCost
            return minCost
        return dp(cost, 0)
```
* Q: come back and add bottom up approach with iterative solution
### 198 House Robber (1-d DP)
* similar to the previous, can be solved by bottom-up approach in an array:
```
    def rob(self, nums: List[int]) -> int:
        dp = [None] * len(nums)
        if len(nums) == 1:
            return nums[0]
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        for i in range(2, len(nums)):
            dp[i] = max(dp[i - 1], nums[i] + dp[i - 2])
        return dp[-1]
```
## Others
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
* union find for graph problems