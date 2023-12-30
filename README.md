# Deepseek-coder-1.3b-instruct


Deepseek Coder is composed of a series of code language models, each trained from scratch on 2T tokens, with a composition of 87% code and 13% natural language in both English and Chinese. We provide various sizes of the code model, ranging from 1B to 33B versions. Each model is pre-trained on project-level code corpus by employing a window size of 16K and a extra fill-in-the-blank task, to support project-level code completion and infilling. For coding capabilities, Deepseek Coder achieves state-of-the-art performance among open-source code models on multiple programming languages and various benchmarks.
[[Huggingface]](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-instruct)



## Setup

Download Huggingface model and save weights

```sh
python weights.py
```

Using custom path

```sh
python weights.py --path <your path here>
```


## Generate

To generate text with the default prompt:

```sh
python model.py
```

```
Default prompt: write a react native snippet to display the text 'hello world'
```
Should give the output:

```
import React from 'react';
import { Text, View, StyleSheet } from 'react-native';

const App = () => {
  return (
    <View style={styles.container}>
      <Text style={styles.text}>Hello, world</Text>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  text: {
    fontSize: 20,
  },
});

export default App;

This is a simple App component in React Native. It uses the Text component from 'react-native' to display the string 'Hello, world'. The Text component is styled with a font size of 20 using a StyleSheet.

The App component is wrapped in a View component with a style of container, which gives the component a flex of 1, aligning it on both the horizontal and vertical axes. The justifyContent and alignItems properties are set to 'center', which centers the Text within the container.

The Text component is styled with a font size of 20 using the text style defined in the styles object.

Remember to import the necessary components from 'react-native' and declare your App component within the component section of your code. This code should work in a React Native environment.
```

To use your own prompt:

```sh
python model.py --prompt <your prompt here> --max-tokens <max_tokens_to_generate> --device 
<device to run inference on>
```

## Examples

### Example 1
````
write a quick sort algorithm in python.
````

````python
import random
from typing import List

def quick_sort(arr: List[int]) -> List[int]:
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[0]
        less_than_pivot = [x for x in arr[1:] if x <= pivot]
        greater_than_pivot = [x for x in arr[1:] if x > pivot]
        return quick_sort(less_than_pivot) + [pivot] + quick_sort(greater_than_pivot)

numbers = [random.randint(0, 100) for _ in range(100)]
print(quick_sort(numbers))
````
### Example 2

````
write a python function the returns the minimum integer in a rotated sorted array..
````

````python
#function
def solve(arr):
    n = len(arr)
    if n == 1:
        return arr[0]
    else:
        for i in range(n):
            min_index = i
            for j in range(i + 1, n):
                if arr[j] < arr[min_index]:
                    min_index = j
            arr[i], arr[min_index] = arr[min_index], arr[i]
        return arr[0]

#test
arr = [64, 34, 25, 12, 22, 11, 90]
print("Original array:", arr)
print("Sorted array:", solve(arr))


#output
'''
Original array: [64, 34, 25, 12, 22, 11, 90]
Sorted array: [11, 12, 22, 25, 34, 64, 90]


In the above function, the outer loop runs for n-1 times where n is the length of the given 
array. The inner loop is responsible for finding the minimum element in the unsorted part of the array. After finding the minimum element, we swap it with the first element of the unsorted part of the array. We keep doing this until the entire array is sorted.

This function will work for any array, not just the one mentioned in the problem.

The time complexity of the above implementation is O(n^2) as we are using nested loops.

The space complexity is O(1) as we are not using any extra space for sorting.'''
````

To see a list of options run:

```sh
python model.py --help
```
