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
# React

## React, [Tic-Tac-Toe Game Tutorial](https://reactjs.org/tutorial/tutorial.html)
* use `slice()` to create a shallow copy of an array
* create an empty array of length 9: `Array(9).fill(null)`
* create a new object based on an existing one:
```
var player = {score: 1, name: 'Jeff'};

var newPlayer = Object.assign({}, player, {score: 2});
# or 
var newPlayer = {...player, score: 2};
```
* function component: use function to replace a component that only has has only `render()` and no states

## React, [EECS Tutorial](https://eecs485staff.github.io/p3-insta485-clientside/setup_react.html)
* Immutable objects in the components are stored in `this.props`, while mutables are in `this.state`
* Immutable objects in `props` are passed/accessible from parent to child components, 
while mutables stay in the instance; changes in mutable objects in `state` also triggers the DOM to change.

## React Fundamentals (https://reactjs.org/docs/components-and-props.html)
* `this.props` access input data, `this.state` maintains internal state/data within the component
* JSX are presented as html-like templated statements, 
    * JSX can include Javascript statements in curly braces
    * JSX are equivalent to Javascript expressions, thus can be placed inside JS functions
    * JSX is safe: user input is permitted in JSX
* native React has a single `root` element
* basic rendering mechanism:
```
const root = ReactDOM.createRoot(
  document.getElementById('root')
);
const element = <h1>Hello, world</h1>;
root.render(element);

```
First, pass the DOM element to `createRoot`, then pass the React element to `render`
* components are similar to JS functions, they accept `props` and return React elements.
* Two ways of describing a component
```
#function components
function Welcome(props) {
  return <h1>Hello, {props.name}</h1>;
}

#ES6 class
class Welcome extends React.Component {
  render() {
    return <h1>Hello, {this.props.name}</h1>
  }
}
```
* React element can use DOM tags or user-defined
* compose large component through smaller ones; components can refer to other components
* name props from the component's own point of view rather than its context
* __All React components must act like pure functions with respect to their props.__ Which means, do not modify props that are passed in
* changes in `state` trigger UI to update: state makes UI interactive
* `state` is private to the component
* to add states to a component
    * the components needs to be a class
    * add a constructor that initialize the state
    * use `setState` to update state, which might trigger UI update
* Q: What's `componentDidMount` and `componentWillUnmount` (https://reactjs.org/docs/state-and-lifecycle.html)
    * they are lifecycle methods on the component class; they manage the space taken up by components
    * `componentDidMount` runs after the component is rendered to the DOM
    * `componentWillUnmount` is called when the component is removed from DOM
* use camelCase
* handling events: what is `bind()` in JS: https://www.smashingmagazine.com/2014/01/
* template literal in jsx
```
<ListItemText primary = {`${item.Title} (${item.Year})`}/>
```

understanding-javascript-function-prototype-bind/
    * Generally, if you refer to a method without () after it, such as onClick={this.handleClick}, you should bind that method.
* lists: add keys to items in a list component; often in the `map()` call
* keys need to be passed under a different name
* controlled component: use `state` and event function to control data and event (submit, click) in HTML form, input, etc instead of letting html manage it
* essentially, controlled components lets you rewrite HTML forms and input tags in React
* Lifting State Up: if several components reflect the same changing data, we can lift the state up to their cloest common ancestor (component that needs it: calculator needs input)
* Because of the top-down data flow of react, lifting the state to higher up in the ladder means the what was originally a state now is a props from a higher level component. Thus, changes in `props` (state before lift) need to be handled by callbacks from the higher level component, where the `props` is a state. Thus, the higher level function needs to pass the event handler as part of the props to the lower level func

## React Hooks (https://reactjs.org/docs/hooks-intro.html)
* Hooks lets you use state and other React features without writing a class 
(hooks are used in funcion components, not in class)
* `useEffect` adds the ability to perform side effects from a function component.
It serves the same purpose as `componentDidUpdate`, `componentdidMount` and `componentWillUnmount`.
React runs the effects after every render
  * React will remember the function you passed (we'll refer to it as our "effect"), and call it later after performing the DOM updates. 
* Rules: Only call hooks at the top level, not inside loops, conditions or nested fun
         useReactHooks
* add new elements to array 
```
setMyArray(oldArray => [...oldArray, newElement]);
```

## React State
* state is some kind of data
* state is related to finite state machine: xstate
* in React, there are two kinds of state: client state and server state
* For client state: state in the client side can be complex to manage
  * `useState`
  * to avoid prop drilling: `useContext`, but it doesn't optimize for modifying state, just passing it down
  * simple use can alterantive, jotai
* On the server side: we need to fetch data from the server 
  * this can be done with simple fetch
  * React Query: modify fetched data and save the modification

## Redux
* This is the basic idea behind Redux: a single centralized place to contain the global state in your application, and specific patterns to follow when updating that state to make the code predictable. https://redux.js.org/tutorials/essentials/part-1-overview-concepts
* Redux expects that all state updates are done immutably
* Action, object with a `type` field, which takes the form of `domain/eventName`. Optionally, it has a `payload` field that contains additional info
* Reducer, function that takes in state and action and return a new state: `(state, action) => newState`
  * calculate new value only based on `state` and `action`, think of it as an event handler
  * make immutable update; a typical reducer will perform as such:
  ```
    * Check to see if the reducer cares about this action
        If so, make a copy of the state, update the copy with new values, and return it
    * Otherwise, return the existing state unchanged
  ```
  * no async logic
* Store, created by passing in a reducer
* State in store can be updated by dispatch, which takes in an action object:
```
store.dispatch({type: `counter/increment`})
console.log(store.getState())
```
* Selectors: extract specific pieces of information from a store state value

### [Example and Workflow](https://redux.js.org/tutorials/fundamentals/part-3-state-actions-reducers)
* Design State Value: To do Items (content text, id, color, completed), Filtering behavior(Active, Color)
* Design State Structure:
```
const todoAppState = {
todos: [
  { id: 0, text: 'Learn React', completed: true },
  { id: 1, text: 'Learn Redux', completed: false, color: 'purple' },
  { id: 2, text: 'Build something fun!', completed: false, color: 'blue' }
],
filters: {
  status: 'Active',
  colors: ['red', 'blue']
}
}
```
* Design Actions
```
{type: 'todos/todoAdded', payload: todoText}
{type: 'todos/todoToggled', payload: todoId}
{type: 'todos/colorSelected', payload: {todoId, color}}
{type: 'todos/todoDeleted', payload: todoId}
{type: 'todos/allCompleted'}
{type: 'todos/completedCleared'}
{type: 'filters/statusFilterChanged', payload: filterValue}
{type: 'filters/colorFilterChanged', payload: {color, changeType}}
```
* Simple Reducer (Notice the use of spread notation to compose state)
```
function nextTodoId(todos) {
  const maxId = todos.reduce((maxId, todo) => Math.max(todo.id, maxId), -1)
  return maxId + 1
}

// Use the initialState as a default value
export default function appReducer(state = initialState, action) {
  // The reducer normally looks at the action type field to decide what happens
  switch (action.type) {
	// Do something here based on the different types of actions
	case 'todos/todoAdded': {
	  // We need to return a new state object
	  return {
	// that has all the existing state data
	...state,
	// but has a new array for the `todos` field
	todos: [
	  // with all of the old todos
	  ...state.todos,
	  // and the new todo object
	  {
		// Use an auto-incrementing numeric ID for this example
		id: nextTodoId(state.todos),
		text: action.payload,
		completed: false
	  }
	]
	  }
	}
	default:
	  // If this reducer doesn't recognize the action type, or doesn't
	  // care about this specific action, return the existing state unchanged
	  return state
  }
}
```
* split reducer functions: `todosReducer` and `filtersReducer`
  * use slice files to organize app structures: a slice file contains reducer logic for part of the app state or feature
	*  The directory structure looks like this `features` -> `todos` -> `todosSlice.js`. The slice file looks like this:
	```
	const initialState = [
	  { id: 0, text: 'Learn React', completed: true },
	  { id: 1, text: 'Learn Redux', completed: false, color: 'purple' },
	  { id: 2, text: 'Build something fun!', completed: false, color: 'blue' }
	]

	function nextTodoId(todos) {
	  const maxId = todos.reduce((maxId, todo) => Math.max(todo.id, maxId), -1)
	  return maxId + 1
	}

	export default function todosReducer(state = initialState, action) {
	  switch (action.type) {
		default:
		  return state
	  }
	}
	```
* Combine reducers:
```
import todosReducer from './features/todos/todosSlice'
import filtersReducer from './features/filters/filtersSlice'

export default function rootReducer(state = {}, action) {
  // always return a new object for the root state
  return {
    // the value of `state.todos` is whatever the todos reducer returns
    todos: todosReducer(state.todos, action),
    // For both reducers, we only pass in their slice of the state
    filters: filtersReducer(state.filters, action)
  }
}
```
Alternatively, use the `combineReducers` function in redux.
```
import { combineReducers } from 'redux'

import todosReducer from './features/todos/todosSlice'
import filtersReducer from './features/filters/filtersSlice'

const rootReducer = combineReducers({
  // Define a top-level state field named `todos`, handled by `todosReducer`
  todos: todosReducer,
  filters: filtersReducer
})

export default rootReducer
```

### Redux Store
* store holds the current application state, access the state with `store.getState()`, update with `store.dispatch(action)`, register listener callbacks with `store.subscribe(listener)`
* every store has a single root reducer function: `const store = createStore(rootReducer, preloadedState)`, with a optional second argument `preloadedState` as the initial data
* Besides `rootReducer` and `preloadedState`, store can be customized with store enhancer, which allows the user to have customized dispatch, getState or subscribe functions
* Redux middleware lets you customize the dispatch behaviour. It is built on top of an type of enhancer called `applyMiddleware`
  * Importantly, middleware is intended to have side effects inside. 
  * Middleware form a piepline around `dispatch`. When `dispatch` is called, the first middleware is activated.
  * Things middleware can do: API calls, logging, modify an action, pause/stop an action
```
//createStore
let preloadedState
const persistedTodosString = localStorage.getItem('todos')

if (persistedTodosString) {
  preloadedState = {
    todos: JSON.parse(persistedTodosString)
  }
}
```
```
//store actions
import store from './store'

// Log the initial state
console.log('Initial state: ', store.getState())
// {todos: [....], filters: {status, colors}}

// Every time the state changes, log it
// Note that subscribe() returns a function for unregistering the listener
const unsubscribe = store.subscribe(() =>
  console.log('State after dispatch: ', store.getState())
)

// Now, dispatch some actions

store.dispatch({ type: 'todos/todoAdded', payload: 'Learn about actions' })

// Stop listening to state updates
unsubscribe()

// Dispatch one more action to see what happens

store.dispatch({ type: 'todos/todoAdded', payload: 'Try creating a store' })
```
### Redux and UI
* rendering content from the store with `useSelector`, which select appropriate element from the store AND listens to changes of the component it renders
```
const TodoList = () => {
  const todos = useSelector(selectTodos)

  // since `todos` is an array, we can loop over it
  const renderedListItems = todos.map(todo => {
    return <TodoListItem key={todo.id} todo={todo} />
  })

  return <ul className="todo-list">{renderedListItems}</ul>
}
```
* dispatch actions to change store
```
const Header = () => {
  const [text, setText] = useState('')
  const dispatch = useDispatch()

  const handleChange = e => setText(e.target.value)

  const handleKeyDown = e => {
    const trimmedText = e.target.value.trim()
    // If the user pressed the Enter key:
    if (e.key === 'Enter' && trimmedText) {
      // Dispatch the "todo added" action with this text
      dispatch({ type: 'todos/todoAdded', payload: trimmedText })
      // And clear out the text input
      setText('')
    }
  }
```
* use `Provider` to provide store context
```
ReactDOM.render(
  // Render a `<Provider>` around the entire `<App>`,
  // and pass the Redux store to as a prop
  <React.StrictMode>
    <Provider store={store}>
      <App />
    </Provider>
  </React.StrictMode>,
  document.getElementById('root')
)
```
## React Testing
### [Testing Implementation Details](https://kentcdodds.com/blog/testing-implementation-details)
* Testing implementation deails will result in two kinds of errors: false positive (test fails but the components still works, after implementation changes) and false negative (test passes but the component breaks, because it does not test whether the component works from a users perspective)
* Use React Testing Library to write implementation-free testing files
* Instead, we test from the perspective of two users: the end user, who will interact with the component and the developer who see the content of the props
> This is precisely what the React Testing Library test does. We give it our own React element of the Accordion component with our fake props, then we interact with the rendered output by querying the output for the contents that will be displayed to the user (or ensuring that it wont be displayed) and clicking the buttons that are rendered.

> Now consider the enzyme test. With enzyme, we access the state of openIndex. This is not something that either of our users care about directly. They don't know that's what it's called, they don't know whether the open index is stored as a single primitive value, or stored as an array, and frankly they don't care. They also don't know or care about the setOpenIndex method specifically. And yet, our test knows about both of these implementation details.

### [Never Use Shallow Rendering](https://kentcdodds.com/blog/why-i-never-use-shallow-rendering)



# Serverside Javascript
## Concepts
* Middleware: Middleware functions are functions that have access to the request object (req), the response object (res), and the next middleware function in the application's request-response cycle. The next middleware function is commonly denoted by a variable named `next`. [https://expressjs.com/en/guide/using-middleware.html]
  * Express is a routing and middleware web framework that has minimal functionality of its own: An Express application is essentially a series of middleware function calls.
  * > If you've ever used a library like Express or Koa, you might already be familiar with the idea of adding middleware to customize behavior. In these frameworks, middleware is some code you can put between the framework receiving a request, and the framework generating a response. For example, Express or Koa middleware may add CORS headers, logging, compression, and more. The best feature of middleware is that it's composable in a chain. You can use multiple independent third-party middleware in a single project. From Redux Doc [https://redux.js.org/tutorials/fundamentals/part-4-store]
