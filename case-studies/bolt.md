
* `api.chat`: send chat messages to llm and return streaming text, no process
* `Chat.client`: send it chat message to `api.chat`, get back streaming text, parse it
* server activities on `api.chat`
    *  when parsing the message (perform actions and modify artifacts), the client calls `useMessageParser`, all action and artifacts gets passed to workbench
    * the specific parsing is implemented here: `app/lib/runtime/message-parser`
    * `store/workbench` serves as a place to run actions on artifacts with `runtime/action-runner'`
    * the actions are performed in the `webcontainer`