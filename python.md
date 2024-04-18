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