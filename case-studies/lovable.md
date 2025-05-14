# GPT-Engineer, aka Lovable (https://github.com/AntonOsika/gpt-engineer/tree/main)
## Generate from scratch
* The cli calls the `main` function to set up a CliAgent: https://github.com/AntonOsika/gpt-engineer/blob/main/gpt_engineer/applications/cli/main.py#L281
* the `main` function calls agent `init` function: https://github.com/AntonOsika/gpt-engineer/blob/main/gpt_engineer/applications/cli/main.py#L542-L546
* Generate code and an `entrypoint`, which is a script to run the code: https://github.com/AntonOsika/gpt-engineer/blob/main/gpt_engineer/applications/cli/cli_agent.py#L152
* setup system prompt: https://github.com/AntonOsika/gpt-engineer/blob/main/gpt_engineer/core/default/steps.py#L75
    * multi-parts prompt: https://github.com/AntonOsika/gpt-engineer/tree/main/gpt_engineer/preprompts
* process the llm output:  [`chat_to_files_dict`](https://github.com/AntonOsika/gpt-engineer/blob/main/gpt_engineer/core/chat_to_files.py#L38)
    * 
## Work on existing project
* The cli calls the `main` function to set up a CliAgent
* `FileSelector.ask_for_files` prompt user to select files they want to work on (not LLM calls): https://github.com/AntonOsika/gpt-engineer/blob/main/gpt_engineer/applications/cli/file_selector.py#L79
* runs the `handle_improve_mode` function, which (after several wrapper) runs the `improve_fn` with an agent : https://github.com/AntonOsika/gpt-engineer/blob/main/gpt_engineer/core/default/steps.py#L271
* the improve function gets the same system prompt as the generating from scratch: https://github.com/AntonOsika/gpt-engineer/blob/main/gpt_engineer/core/default/steps.py#L97 and https://github.com/AntonOsika/gpt-engineer/blob/main/gpt_engineer/preprompts/improve
    * note that the improved code will have git syntax, which will be presented to the user to confirm changes
* https://deepwiki.com/search/walk-me-through-how-improve-mo_407d582e-4c3b-4f6a-9469-77868db06a28
