* `AgentOutput` class: https://github.com/browser-use/browser-use/blob/0e4d4ca1908969f9cc67fff7c6337248edd91106/browser_use/agent/message_manager/service.py#L230-L260
* essentially there is only one tool, which is the `AgentOutput`, which looks like this:
```
example_tool_call = AIMessage(
			content='',
			tool_calls=[
				{
					'name': 'AgentOutput',
					'args': {
						'current_state': {
							'evaluation_previous_goal': """
							Success - I successfully clicked on the 'Apple' link from the Google Search results page, 
							which directed me to the 'Apple' company homepage. This is a good start toward finding 
							the best place to buy a new iPhone as the Apple website often list iPhones for sale.
						""".strip(),
							'memory': """
							I searched for 'iPhone retailers' on Google. From the Google Search results page, 
							I used the 'click_element_by_index' tool to click on element at index [45] labeled 'Best Buy' but calling 
							the tool did not direct me to a new page. I then used the 'click_element_by_index' tool to click 
							on element at index [82] labeled 'Apple' which redirected me to the 'Apple' company homepage. 
							Currently at step 3/15.
						""".strip(),
							'next_goal': """
							Looking at reported structure of the current page, I can see the item '[127]<h3 iPhone/>' 
							in the content. I think this button will lead to more information and potentially prices 
							for iPhones. I'll click on the link at index [127] using the 'click_element_by_index' 
							tool and hope to see prices on the next page.
						""".strip(),
						},
						'action': [{'click_element_by_index': {'index': 127}}],
					},
					'id': str(self.state.tool_id),
					'type': 'tool_call',
				},
			],
		)
```
notice that `actions` itself is only part of the function output
* Model output is then parsed into `AgentOutput` by `Agent.get_next_action()`: https://github.com/browser-use/browser-use/blob/0e4d4ca1908969f9cc67fff7c6337248edd91106/browser_use/agent/service.py#L1087