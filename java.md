# Java
## Reference books
* Core Java for the Impatient (Hortstmann, 3rd, rough cuts, easy to read)
* A Java Book (Hilfinger, 2018 CS61B from Berkeley, contain syntactical representation of the Java, which is hard to read)
* Effective Java (Bloch, 3rd, academic) 
## Knowledge
### Object-oriented programming (OOP)
* encapsulation: hide irrelevant information, expose API instead
    * example: get and set methods for classes, 
* inheritance: class hierarchy
* polymorphism: Pure polymorphism occurs when a single function can be applied to arguments of a variety of types. The other extreme occurs when we have a number of different functions all denoted by the same name - a situation known as overloading
    * example: interface that allow different implementation of the same methods
* abstraction: hide irrelevant information
    * example: ability to create complex classes out of simpler ones, and to create methods to abstract away complex structures
* Resource:
https://www3.ntu.edu.sg/home/ehchua/programming/java/J3a_OOPBasics.html
stanford cs108
* Data binding: how to look for correct implementation when a method is called; depending on when binding takes place (compile time or run time), there can be static binding and dynamic binding
https://www.cs.cornell.edu/courses/cs211/2004fa/
#### Composition over Inheritance
* https://www.reddit.com/r/AskProgramming/comments/lv7m7a/i_still_dont_understand_the_prefer_composition/
### Class
* `Static` methods are methods that do not operate on objects, eg `Math.pow()`
* `Static` variables have only one per class. Access by calling it directly, eg `System.out` (Hortstmann)
* Big three methods of `Object` class: `toString`, `equals` and `hashCode` (Hortstmann)
### Copy
* `clone` method makes a shallow copy, where instance variables are cloned from the original to the cloned object. If the variables are objects, they cloned object would share the same variable with the original.
* To prevent this behavior, implement a generic `clone` method to implement a deep copy. (See example in Hortsmann, 4.3.4)
* If one needs to use the default `clone` in Object, since `clone` is a protected class in `Object`, new objects would need to implement the `Cloneable` interface to raise the scope of `clone` to `protected` to `public`.
### Comparison
* `equals()` is a Object class method that compare if the two objects are equal.
* each instance of the class is inherently unique, since they occupy different location in memory (their references are equal).
* "logical equality" applies when you want to say that two objects are equal when they have the same value. This is true for `Integer` and `String`, where the `equals` method got overriden. (Bloch, Item 10)
* Otherwise, it is better not to override the `equals` method, since it is prone to errors. One also need to change the `hashCode` method to align with the `equals` behavior. 
### Pass-by-value and memory management
* Java is pass by value; when you pass an object to a method, the method obtains a copy of the object reference (Hilfinger, 157; Hortstmann "call by value"); example:
```
jshell> static void assign (int x){ x = 42; }
|  created method assign(int)

jshell> int y = 2;
y ==> 2

jshell> assign(y);

jshell> y
y ==> 2
```
* In Java, a variable holds a reference to an object, unlike C, which holds the actual object (Hortstmann)
* When an object is no longer referenced by any variables, it goes to garbage collection
### threading and cocurrency
* implements `Runnable`
```
jshell> class Fish implements Runnable {
   ...>     private int index;
   ...>     public Fish(int i) {
   ...>         index = i;
   ...>     }
   ...>     public void run() {
   ...>         System.out.println("fish index " + index);
   ...>         try{
   ...>         Thread.sleep(4000);
   ...>     } catch (Exception e){};
   ...>         System.out.println("fish index " + index + "finished");
   ...>     }}
jshell> for (int i = 0; i < 5; ++i) {
   ...>     Fish fish = new Fish(i);
   ...>     Thread net = new Thread(fish);
   ...>     net.start();
   ...> }
```
* extends `Thread`
```
jshell> class FishThread extends Thread {
   ...>     private int index;
   ...>     public FishThread(int i){
   ...>         index = i;
   ...>     }
   ...>     public void run() {
   ...>         System.out.println("1 from thread " + index);
   ...>         try{
   ...>         Thread.sleep(2000);}
   ...>         catch(Exception e){}
   ...>         System.out.println("2 from thread " + index);
   ...>     }}
jshell> for (int i = 0; i < 5; ++i) {
   ...>     Thread net = new FishThread(i);
   ...>     net.start();
   ...> }
```
* 
### common data structure
#### array
* array are reference types, which is a pointer to an array object: `Object A = new int[4]`
* initialization: `int[] A = new int[4]`, `int[] B = {1, 2, 3}`
* length with `arr.length`, which is a `final` field, rather than a method
* multidimentional array with double `[]`:
```
int[][] square = {
{ 16, 3, 2, 13 },
{ 5, 10, 11, 8 },
{ 9, 6, 7, 12 },
{ 4, 15, 14, 1}
};
```
#### List
* List is generic type where types of the elements of the list needs to be specified
* initialize and add elements happen separately:
```
ArrayList people = new ArrayList ();
while (/*( there are more people to add )*/) {
Person p = /*( the next person )*/;
people.add (p);
}
```
* length of a list with `size()`
* access and modify the list: `add()`, `get(index)`, `set(index, newElem)`, `remove(index OR obj)`, `contains()`, `indexOf()`
#### String
* String initiation concatnation (`"" + str1`) and `StringBuilder()`
* get the length with `length()`
* `String.valueOf()` convert primitive types to string; `Integer.parseInt()` converts str back to int
* `String.format()` output formatted string, see format print section below
* `substring()` access substrings with 0 based index
* String comparison with `equals()`, not `==`, this is because when applied to references, they compare the pointer values, whether they points to the same object (rather then the content); However, in strings, we want to compare content. (Hilfinger, 143)
* to append a large of strings, use `StringBuilder`
```
StringBuilder result = new StringBuilder ();
for (int i = 0; i < A.length; i += 1) {
    result.append(A[i])
}
```
* combine string with delimiter with `join`:
```
String names = String.join(", ", "Peter", "Paul",
"Mary");
```
#### Primative types
* Integer class or int? `Integer` is an object so its stances can be null. Use Integer when method take objects, like `List<T>`. Reference: https://stackoverflow.com/questions/10623682/using-int-vs-integer; see also 10.5 in Hilfinger where he talks about the distinction of primitive types and reference types and the need for wrapper classes
#### format print/string
examples: `System.out.printf("%-15s%03d\n", str, num);` and `String.format ("|%-5d|%5d|", 13, 13)`
* conversion-character
s formats strings.
d formats decimal integers.
f formats floating-point numbers.
t formats date/time values.
* %[flags][width][.precision]conversion-character
* Hortstmann, 1.6.2
#### HashMap
```
public class Main {
  public static void main(String[] args) {
    // Create a HashMap object called capitalCities
    HashMap<String, String> capitalCities = new HashMap<String, String>();

    // Add keys and values (Country, City)
    capitalCities.put("England", "London");
    capitalCities.put("Germany", "Berlin");
    capitalCities.put("Norway", "Oslo");
    capitalCities.put("USA", "Washington DC");
    capitalCities.get("England");
    System.out.println(capitalCities);
  }
}
```
## Coding Questions
* Q1
* 'abc' =>  ['ab', 'c_']
* 'abcdef' => ['ab', 'cd', 'ef']
```
public class StringSplit {
    public static String[] solution(String s) {
        //Write your code here
      if (s.length() % 2 != 0) {
        s += "_";
      }
      int outputLen = s.length()/2;
      String[] output = new String[outputLen];
      
      for (int i = 0; i < outputLen; ++i) {
        output[i] = s.substring(i *2 , i * 2+2);
      }
      return output;
    }
}
```
* Q2 convert to binary
```
public static String convertToBinary(int n) {
        if (n == 0) {
            return "0";
        }

        StringBuilder binary = new StringBuilder();
        while (n > 0) {
            int bit = n % 2;
            binary.insert(0, bit);
            n /= 2;
        }

        return binary.toString();
    }
```
* Q3 count the number of bits that are 1s
```
public static int countBits(int n) {
        int count = 0;
        while (n > 0) {
            count += n & 1;
            n >>= 1;
        }
        return count;
    }
```
iterate by performing right shift(>>=), check 1s of the least significant number by performing a bitwise AND (&).

## Loose Threads
How is Java event excution differ from JS?
https://developer.mozilla.org/en-US/docs/Web/JavaScript/Event_loop
