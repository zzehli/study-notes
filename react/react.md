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

### useMemo
* it lets you cache the result of a calculation between re-renders, so that the calculation does not run again when re-renders
* for example, when toggle theme, the content of the page does not change, in this case, we can use `useMemo` to cache the content

### useRef
* it lets you use a value that does not triggers re-render
* it lets you reference a value that’s not needed for rendering.
* this value is mutable with the `current` property
* for example, typing in an input form should not trigger re-render, therefore we can use a ref value to save the input box state

### useContext
* it lets you read and subscribe to context from your component

### useCallback
* it lets you cache a function definition between re-renders
* in javascript, functions are recreated every time it runs:
```
  useEffect(() => {
    const options = createOptions();
    const connection = createConnection(options);
    connection.connect();
    return () => connection.disconnect();
  }, [createOptions]);
```
where `createOptions` is:
```
  function createOptions() {
    return {
      serverUrl: 'https://localhost:1234',
      roomId: roomId
    };
  }
```
since `createOption` is inside useEffect, if `useEffect` runs, `createOption` is created, which triggers *another* `useEffect` rendering.

To fix this, wrape `createOption` with `useCallback`:
```
  const [message, setMessage] = useState('');

  const createOptions = useCallback(() => {
    return {
      serverUrl: 'https://localhost:1234',
      roomId: roomId
    };
  }, [roomId]); // ✅ Only changes when roomId changes

  useEffect(() => {
    const options = createOptions();
    const connection = createConnection(options);
    connection.connect();
    return () => connection.disconnect();
  }, [createOptions]);
```

### useReducer
* consolidate all the state update logic outside your component in a single function
* `const [state, dispatch] = useReducer(reducer, initialArg, init?)` where the `dispatch` function lets you update your state (`initialArg` is the initial val)

### resources
* see https://github.com/kentcdodds/react-hooks
* exercise https://epic-react-exercises.vercel.app/react/hooks/1

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
* why not to use `shallow` from enzyme