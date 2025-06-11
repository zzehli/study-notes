* codeact agent: https://github.com/All-Hands-AI/OpenHands/blob/d6d5499416fe8a40b8b36353350c26e7d974d682/openhands/agenthub/codeact_agent/codeact_agent.py
    * does use function calling for bash calls, not naive parsing, eg.
    ```
    A: Sure! Let me first check the current directory:
    <function=execute_bash>
    <parameter=command>
    pwd && ls
    </parameter>
    </function>
    ```
    * there could be thoughts as well as function call, the two are not contradictory
* parsing this action:
```
            if tool_call.function.name == create_cmd_run_tool()['function']['name']:
                if 'command' not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "command" in tool call {tool_call.function.name}'
                    )
                # convert is_input to boolean
                is_input = arguments.get('is_input', 'false') == 'true'
                action = CmdRunAction(command=arguments['command'], is_input=is_input)
```
see https://github.com/All-Hands-AI/OpenHands/blob/0abc6f27/openhands/agenthub/codeact_agent/function_calling.py#L85C11-L101
* prompt: https://github.com/All-Hands-AI/OpenHands/blob/6171395ef9194c0d3757ca6ad0c65997c95b47eb/openhands/agenthub/codeact_agent/prompts/in_context_learning_example.j2
* reference: https://deepwiki.com/search/how-does-codeagent-call-tools_5c6cdf61-d2c9-4b7e-8abb-ba05e49d0968