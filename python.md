* iterate through an array with index: for i in range(len(nums))
* define a function inside other:
    ```
    def funca(self, arg1):
        def funcb(self, arg2):
            self.funcb(var2)

        self.funcb(var1)
    ```
* don't modify in place in recursive calls
    ```
    path.append(nums[i])
    self.backtrack(nums[i:], target - nums[i], path, ret)
    ```
    This is a problem, according to chatGPT:
    Here, the `nums[i]` value is appended to the path list before making the recursive call. This means that the path list is modified in place, and this modified path is used in subsequent recursive calls. This can lead to issues because all recursive calls end up sharing the same path list.
    ```
    self.backtrack(nums[i:], target - nums[i], path + [nums[i]], ret)
    ```
    According to python [doc](https://docs.python.org/3/tutorial/controlflow.html#for):
    > Code that modifies a collection while iterating over that same collection can be tricky to get right. Instead, it is usually more straight-forward to loop over a copy of the collection or to create a new collection:
* `Counter()` build a dictionary object based on the given hashable object (array, dict) (https://docs.python.org/3/library/collections.html#collections.Counter)
    ```
    >>> l = collections.Counter(["fish", "cat", "fish"])
    >>> l["fish"]
    ```
* `enumerate()` returns an enumerate object (57 Insert Interval)
    ```
    >>> seasons = ['Spring', 'Summer', 'Fall', 'Winter']
    >>> list(enumerate(seasons))
    [(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
    ```
* `sort` array of arrays with lambda (https://docs.python.org/3/howto/sorting.html#key-functions):
    ```
    student_tuples = [
        ('john', 'A', 15),
        ('jane', 'B', 12),
        ('dave', 'B', 10),
    ]

    sorted(student_tuples, key=lambda student: student[2])   # sort by age
    [('dave', 'B', 10), ('jane', 'B', 12), ('john', 'A', 15)]
    ```
* use `bisect` for binary search
    * `bisect_left` returns the left most place of insertion, while `bisect` or `bisect_right` returns the right most place:
    ```
    >>> bisect.bisect_left([1,2,3], 2)
    1
    >>> bisect.bisect_right([1,2,3], 2)
    2
    ```
* create a 2d array (matrix): don't use `list * N`, instead use `[ [0]*M for _ in range(N) ]`
* create a dictionary of arrays: `defaultdict(list)`:
    ```
    s = [('yellow', 1), ('blue', 2), ('yellow', 3), ('blue', 4), ('red', 1)]
    d = defaultdict(list)
    for k, v in s:
        d[k].append(v)
    ```
* Use memoization to avoid repetition with `@cache`. see sol 2 of https://leetcode.com/problems/word-break/editorial/