# Dyad: https://github.com/dyad-sh/dyad
* `chat_stream_handlers` receives chat stream, which gets added to `fullResponse`, which is then passed to `processFullResponseActions` (`response_processor`)
* `response_processor` parse the response and handle all the actions