# Bolt: https://github.com/stackblitz/bolt.new
## Generate from scratch
* `Chat.client`: send it chat message to `/api/chat`, get back streaming text, parse it
* `/api/chat`: send chat messages to llm and return streaming text, no process
* llm output is parsed on client side: when parsing the message (perform actions and modify artifacts), the client calls `useMessageParser`, all action and artifacts gets passed to workbench
* the specific parsing logic is implemented here: `app/lib/runtime/message-parser`
* `store/workbench` serves as a place to run actions on artifacts with `runtime/action-runner'`
* the actions are performed in the `webcontainer`
## Work on existing project
* `send_message` is where actions take place, it performs a diff and append file modification to chat history: https://deepwiki.com/search/walk-me-through-how-improve-mo_407d582e-4c3b-4f6a-9469-77868db06a28
