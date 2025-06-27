# Code execution sandboxes
## Pyodide 
* is a port of Cpython in WASM. It allows you to run python code in the browser. The problem with this approach is Pyodide doesn't support all the pip packages. Instead, it uses `micropip`, which impose restrictions on package choices
## containers
* Proper sandbox environment include various container solutions: e2b, etc
* codeact agent use jupyter kernel to execute code: https://github.com/xingyaoww/code-act/blob/d607f56c9cfe9e8632ebaf65dcaf2b4b7fe1c6f8/scripts/chat/code_execution/jupyter.py
* many uses microvms. E2B uses `firecracker` 