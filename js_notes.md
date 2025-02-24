# Eloquent Javascript
## Ch 1.
- statement and expression
    * statement are sentences, they are followed by semicolons
- type coersion
	* special type coersion for logic operator: && preserve `false` on the left side, || preserves `false` on the right
	* comparison with `===` does not allow type conversion
	* type checking, check if the argument is a numer: !Number.isNaN(theNumber)
						check empty string: !yourName

## Ch 2.
- reference
	* `let` creates a reference to a value that can be changed later
	* `const` creates a immutable binding
Q: what is console.log?
	It is a output command that send output to console
- some keywords:
	* side effect vs. return a value
	* state
	* binding
	* block: {}

## Ch 3.
- wrapping a piece of program in a value
```
const square = function(x) {
	return x * x;
};

console.log(square(12));
```

- local and global
	*  `let` only define variables within the block
	* lexical scoping: each layer can access its outer scope's variable

- function binding behaves similar to a value
- function declarations are different: they are available to the entire scope, not just those below it
``` javascript
console.log("The future says:", future());
function future() {
	return "You'll never have flying cars";
}
```
- arrow function for smaller functions
```
let h = a => a % 3
```

``` javascript
const power = (base, exponent) => {
    let result = 1;
    for (let count = 0; count < exponent; count++) {
        result *= base;
    }
    return result;
};
```

- optional argument: function can accept more arg than it uses
``` javascript
function minus(a, b) {
	if (b === undefined) return -a;
	else return a - b;
}
```
- closure
- pure function: doesn't rely on non argument (such as global variables) to determine its return value; given the same arguments, always has the same output regardless of the context

## Ch.4 Objects and arrays
- array properties include `length`; elements in arrays are also considered as properties
- bracket notation in arrays to select properties
`value["John Doe"]` to access property if name of that property is not a proper binding name

- objects
	* empty property of an object will be `undefined`
	* number, string, boolean are immutable; they are not objects
	* objects are mutable
``` javascript
let day1 = {
   squirrel: false,
   events: ["work", "touched tree", "pizza", "running"]
};
```


- array operations: pop, push, shift, unshift, indexOf, slice, concat
- string operations: slice, indexOf, trim, padStart (take a string and make it to desired length with padding character specified), join, split,

- rest parameters
``` javascript
function max(...numbers) {
}
let input = [5, 1, 7];
console.log(max(...numbers));
```

- destructuring: quick access to array/object elements with alias
```
let {name} = {name: "Faraji", age:23};
```
- JSON serialization: JSON.stringify JSON.parse

## Ch.5
``` javascript
let labels = []l;
repeat(5, i => {
	labels.push('Unit ${i + 1}`)'
	});
console.log(labels);
```
- working with higher order functions in arrays with `forEach` and `filter`
- process data with map
``` javascript
function map(array, transform) {
	let mapped = [];
	for (let element of array) {
		mapped.push(transform(element));
		}
		return mapped;
}
let rtlScripts = SCERPTS.filter(s => s.direction == "rtl");
console.log(map(rtlScripts, s -> s.name);
```
- reduce (map and reduce)
``` javascript
function reduce(array, combine, start) {
	let current = start;
	for (let element of array) {
		current = combine(current, element);
	}
	return current;
}

console.log(reduce([1, 2, 3, 4], (a, b) => a + b, 0));
```
Another example
```
let arrays = [[1, 2, 3], [4, 5], [6]];

console.log(arrays.reduce((flat, current) => flat.concat(current)));
```
Another example
``` javascript
function dominantDirection(text) {
    let scripts = countBy(text, char => {
    let script = characterScript(char.codePointAt(0));
    return script ? script.direction : "none";
  }).filter(({direction}) => direction != "none");

//(a,b) destructureing for array elements
//count and name are element names in countBy() function
  return scripts.reduce((a, b) => a.count > b.count?  a : b, 0).name;
}
```
- `some` keyword for array returns true if given elements are in the array

## Ch.6
- methods
``` javascript
function speak(line) {
	console.log(`The ${this.type} rabbit says '${line}');
}
let whiteRabbit = {type: "white", speak};

//arrow function
function normalize() {
	console.log(this.coords.map(n => n / this.length));
}
```

``` javascript
//define method within object
let protoRabbit = {
	speak(line) {
		console.log(...);
	}
}
```

- prototype and constructor
`prototype` is a property of object just like other properties like length
``` javascript
let rabbit = Object.create(protoRabbit);
//here we create a instance of protoRabbit
//and define its properties with dot notation
rabbit.speak(...);
```
With a function more like a constructor
``` javascript
    function makeRabbit(type) {
        let rabbit = Object.create(protoRabbit);
        rabbit.type = type;
        return rabbit;
    }
```
Use `new` keyword to construct an object with empty Object.prototype to be defined
    * the new object's prototype will be found in the constructor

``` javascript
function Rabbit(type) {
     this.type = type;
     }

Rabbit.prototype.speak = function(line) {...};
//a new object with Rabbit.prototype is created by calling new
let weiredRabbit = new Rabbit("weird");
```
current iteration: use class keyword to define prototype

``` javascript
class Rabbit {
    constructor(type) {
        this.type = type;
    }
    speak(line) {
        consoele.log(...);
    }
}

let killerRabbit = new Rabbit("black");
```
-Override

``` javascript
Rabbit.prototype.teeth = "small";
console.log(killerRabbit.teeth);//small
killerRabbit.teeth = "long, sharp, and bloody";
console.log(Rabbit.prototype.teeth);//small
console.log(killerRabbit.teeth);//long sharp...
```
-Map
    * js properties are indistinguishable for `in`, technically,
    `toString` is property of `Object`, `ages` inherits it from `Object`, to
        avoid this behavior, use `Object.key(..)` or `hasOwnProperty(..)`e
``` javascript
    let ages = { Boris: 39, Liang: 22};
    console.log(`Boris is ${age["Boris"]}`);
    console.log{"toString" in ages}; \\true
```
    * Map class to store associated values, with `get`, `has` and `set` methods

``` javascript
let ages = new Map();
ages.set('Boris', 39);
ages.set('Boris', 22);

console.log(`Is Jack's age knnown?", ages.hash("Jack");
```

-Use Symbol to define potentially duplicate properties to avoid override

``` javascript
const toStringSumbol = Symbol("toString");
Array.prototype[toStringSymbol] = function() {...}
//the property created by Symbol is not the same as the default to String
console.log([1,2].toString());
console.log([1,2][toStringsymbol]());
```

-Iterator and iterable
    * defined by `Symbol.iterator`

``` javascript
let okIterator = "OK"[Symbol.iterator]();
//{value: "O", done: false}
console.log(okIterator.next());
console.log(...);
//third time, return {value: undefined, done:true}
console.log(...);
```

    * `iterator` method is stored as Symbol
    * the `itr` class for the iterable object need to
    provide `next` method that return the next `value` or `done`

``` javascript
//define a class that we want to iterate
class Matrix {
  constructor(width, height, element = (x, y) => undefined) {
    this.width = width;
    this.height = height;
    this.content = [];

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        this.content[y * width + x] = element(x, y);
      }
    }
  }

  get(x, y) {
    return this.content[y * this.width + x];
  }
  set(x, y, value) {
    this.content[y * this.width + x] = value;
  }
}

//implement iterator class with value and next
class MatrixIterator {
  constructor(matrix) {
    this.x = 0;
    this.y = 0;
    this.matrix = matrix;
  }

  next() {
    if (this.y == this.matrix.height) return {done: true};

    let value = {x: this.x,
                 y: this.y,
                 value: this.matrix.get(this.x, this.y)};
    this.x++;
    if (this.x == this.matrix.width) {
      this.x = 0;
      this.y++;
    }
    return {value, done: false};
  }
}

//set Matrix to be iterable, this could be done within the class above
Matrix.prototype[Symbol.iterator] = function() {
    return new MatrixIterator(this);
};

//use iterator with `of`
let matrix = new Matrix(2, 2, (x, y) => `value ${x}, ${y}`);
for (let {x, y, value} of matrix) {
    console.log(x, y, value);
    }
```
- inheritance
    * `extend` makes new prototype from old prototype

``` javascript
class SymmetricMatrix extends Matrix {
    constructor(size, element = (x, y) => undefined {
        super(size, size, (x, y) => {
            if (x < y) return element (y, x);
            else return element(x, y);
        });
    }

    set(x, y, value) {
        super.set(x, y, value);
        if (x != y) {
            super.set(y, x, value);
        }
    }
}
//set method is not invoked here
let matrix = new SymmetricMatrix(5, (x, y) => `${x},${y}`);
console.log(matrix.get(2,3));
```

- getter and setter are invoked automatically for respective properties

``` javascript
class Temperature {
    constructor(celcius) {
        this.celsius = celsius;
    }
    get fahrenheit() {
        return this.celsius * 1.8 + 32;
    }
    set fahrenheit(value) {
        this.celsius = (value - 32) / 1.8;
    }
//set up object and get and set fahrenheit
let temp = new Temperature(22);
console.log(temp.fahrenheit)
//set fahrenheit through setter, which convert to celsius;
temp.fahrenheit = 86;
//property assessed directly
console.log(temp.celsius);
```

- `instanceof` true if the object comes before is a child class of the object after
## Ch7
- persistent: data structures that do not change, `Object.freeze`
```javascript
let object = Object.freeze({value: 5});
object.value = 10;
console.log(object.value);//5
```
- generate random number of array length `Math.floor(Math.random() * array.length)`

## Ch 8
- "use strict"
- exception

``` javascript
function promptDirection(question) {
    //might result in error here
    let result = prompt(question);
    if (...) return "L";
    if (...) return "R";
    throw new Error("Invalid direction: " + result);
    }

    function look() {
        if (promptDirection("Which way?") == "L") {
            return "a house";
        } else {
            return "two angry bears";
        }

    }

    try {
        //call look(), might have an error from prompt()
        console.log("You see", look());
    } catch (error) {
        console.log("Something went wrong: " + error);
    }

```
- use fewer side effects, computes new values instead of changing existing data

- `try...finally` 
``` javascript
function transfer(from, amount) {
  if (accounts[from] < amount) return;
  let progress = 0;
  try {
    accounts[from] -= amount;
    progress = 1;
    accounts[getAccount()] += amount;
    progress = 2;
  } finally {
    if (progress == 1) {
      accounts[from] += amount;
    }
  }
}
```

- selective catch error
    * the `error` in js doesn't have specific types
    * use self defined `error` to catch specific ones:

```javascript
//define error from the Error
class InputError extends Error {}

function promptDirection(question) {
  let result = prompt(question);
  if (result.toLowerCase() == "left") return "L";
  if (result.toLowerCase() == "right") return "R";
  throw new InputError("Invalid direction: " + result);
}

for (;;) {
  try {
    let dir = promptDirection("Where?");
    console.log("You chose ", dir);
    break;
  } catch (e) {
    if (e instanceof InputError) {
      console.log("Not a valid direction. Try again.");
    } else {
      throw e;
    }
  }
}
```

- assertion
    * no specific syntax, just `throw` on speicific cases
```javascript
function firstElement(array) {
  if (array.length == 0) {
    throw new Error("firstElement called with []");
  }
  return array[0];
}
```

## Ch.9 Regex
- `RegExp` constructor with `new` or with enclosing forward slash `/.../`
```javascript
let re1 = new RegExp("abc");
let re2 = /abc/;
console.log(/abc/.test("abcde")) //true
```
- syntax
    * `/n` forward slash is treated along with character to mean a line break
    * `\+` special character in the search pattern use backslash to mark
    * `[0-9]` find a character matching any of the characters in the brackets
    * `\d` any digit, same as `[0-9]`
    * `\w` any word character/number
    * `\s` any whitespace (space, tab, newline, etc)
    * `\W` and `\S` opposites of `\w` and `\s`
    * `.` any character except newline
    * `[^]` except the following group
    * `+` one or more
    * `*` 0 or more
    * `?` optional 
    * `^` start with
    * `$` end with
    * `\b` word boundary
    * `()` capture group
    * `{}` number of repetition, could be range `{2, 4}`
- inside brackets, dot and plus doesn't have special meaning: `[\d.+]` single digit or a preiod or a plus sign

- use paren to group elements in order to match groups: `(hoo)+` matchines `hoo` one or more times

- case sensitive with `i`
```javascript
let cartoonCrying = /boo+(hoo+)/i;
```

- `exec` method for `RegExp`; `test` method from String
```javascript
let match = /\d+/.exec("one two 100");
console.log(match); //["100"]
console.log(match.index); // 8
console.log("one two 100".match(/\d+/))

let quotedText = /'([^']*)'/;
console.log(quotedText.exec("she said 'hello'"));
//second item in the array is the group match
// → ["'hello'", "hello"]

console.log(/bad(ly)?/.exec("bad"));
// → ["bad", undefined]
console.log(/(\d)+/.exec("123"));
//group match only return the last matching term
// → ["123", "3"]
```

-date
* current date and time with `new Date()`
* month starts at 0 but days start at 1
```javascript
console.log(new Date(2009, 11, 9));
// → Wed Dec 09 2009 00:00:00 GMT+0100 (CET)
```

```javascript
//extract Date from string
function getDate(string) {
    //'_' is the full match in /.../ and is ignored
  let [_, month, day, year] =
    /(\d{1,2})-(\d{1,2})-(\d{4})/.exec(string);
  return new Date(year, month - 1, day);
}
console.log(getDate("1-30-2003"));
// → Thu Jan 30 2003 00:00:00 GMT+0100 (CET)
```

- word boundary
```javascript
console.log(/\bcat\b/.test("concatenate"));
// → false
```

-replace method
```javascript
//global replacement with g
console.log("Borobudur".replace(/[ou]/g, "a"));

//parametrized parameter
console.log(
  "Liskov, Barbara\nMcCarthy, John\nWadler, Philip"
    .replace(/(\w+), (\w+)/g, "$2 $1"));
//   Barbara Liskov
//   John McCarthy
//   Philip Wadler
```

- greedy character (+, *, ? and {}), make them nongreedy with `?` after them
- dynamically creating regexp
```javascript
let name = "harry";
let text = "Harry is a suspicious character.";
let regexp = new RegExp("\\b(" + name + ")\\b", "gi");
console.log(text.replace(regexp, "_$1_"));
// → _Harry_ is a suspicious character.
```
- `search` method instead of `indexOf` 

## Ch.10 
- CommonJS with `require`
- Module's binding
- ES Modules
> An ES module’s interface is not a single value but a set of named bindings. The preceding module binds formatDate to a function. When you import from another module, you import the binding, not the value, which means an exporting module may change the value of the binding at any time, and the modules that import it will see its new value.

> When there is a binding named default, it is treated as the module’s main exported value. If you import a module like ordinal in the example, without braces around the binding name, you get its default binding. Such modules can still export other bindings under different names alongside their default export. 


## Ch.11
- Do the exercise at the end of the chapter
> So callbacks are not directly called by the code that scheduled them. If I call `setTimeout` from within a function, that function will have returned by the time the callback function is called. And when the callback returns, control does not go back to the function that scheduled it.
# Javascript in General
* Ways to compose parameters
use template literals
```
onst city = "Rome";
const price = "200";

const myNaiveUrl = `https://www.example.dev/?city=${city}&price=${price}`;
```

create search parameter
```
const params = new URLSearchParams({
  var1: "value",
  var2: "value2",
  arr: "foo",
});
console.log(params.toString());
//Prints "var1=value&var2=value2&arr=foo"
```

* two ways to register event handlers: https://developer.mozilla.org/en-US/docs/Web/Events/Event_handlers
  * onevent properties: event have properties that are prefixed by `on`
```
  const btn = document.querySelector('button');

function greet(event) {
  console.log('greet:', event)
}

btn.onclick = greet;
```

  * event listener specificies event expected and its callback functions, it has
  can add multiple handlers to a event
```
const btn = document.querySelector('button');

function greet(event) {
  console.log('greet:', event)
}

btn.addEventListener('click', greet);
```

* array function return object literals
```
setSearchRes(data['Search'].map(elem => ({Title: elem.Title, Year: elem.Year})));
```
* Every module can have two different types of export, named export and default export. You can have multiple named exports per module but only one default export.
# Typescript
* TypeScript can infer the type of an object or variable based on the initial values you provide, eg 
  ```
  let users = [
    { id: 1, name: 'John Doe', age: 30 },
    { id: 2, name: 'Jane Doe', age: 25 }
  ];
  ``` 
  (chatGPT) 

# Serverside Javascript
## Concepts
* Middleware: Middleware functions are functions that have access to the request object (req), the response object (res), and the next middleware function in the application's request-response cycle. The next middleware function is commonly denoted by a variable named `next`. [https://expressjs.com/en/guide/using-middleware.html]
  * Express is a routing and middleware web framework that has minimal functionality of its own: An Express application is essentially a series of middleware function calls.
  * > If you've ever used a library like Express or Koa, you might already be familiar with the idea of adding middleware to customize behavior. In these frameworks, middleware is some code you can put between the framework receiving a request, and the framework generating a response. For example, Express or Koa middleware may add CORS headers, logging, compression, and more. The best feature of middleware is that it's composable in a chain. You can use multiple independent third-party middleware in a single project. From Redux Doc [https://redux.js.org/tutorials/fundamentals/part-4-store]
