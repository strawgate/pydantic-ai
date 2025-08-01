
# Toolsets

A toolset represents a collection of [tools](tools.md) that can be registered with an agent in one go. They can be reused by different agents, swapped out at runtime or during testing, and composed in order to dynamically filter which tools are available, modify tool definitions, or change tool execution behavior. A toolset can contain locally defined functions, depend on an external service to provide them, or implement custom logic to list available tools and handle them being called.

Toolsets are used (among many other things) to define [MCP servers](mcp/client.md) available to an agent. Pydantic AI includes many kinds of toolsets which are described below, and you can define a [custom toolset](#building-a-custom-toolset) by inheriting from the [`AbstractToolset`][pydantic_ai.toolsets.AbstractToolset] class.

The toolsets that will be available during an agent run can be specified in three different ways:

* at agent construction time, via the [`toolsets`][pydantic_ai.Agent.__init__] keyword argument to `Agent`
* at agent run time, via the `toolsets` keyword argument to [`agent.run()`][pydantic_ai.Agent.run], [`agent.run_sync()`][pydantic_ai.Agent.run_sync], [`agent.run_stream()`][pydantic_ai.Agent.run_stream], or [`agent.iter()`][pydantic_ai.Agent.iter]. These toolsets will be additional to those provided to the `Agent` constructor
* as a contextual override, via the `toolsets` keyword argument to the [`agent.override()`][pydantic_ai.Agent.iter] context manager. These toolsets will replace those provided at agent construction or run time during the life of the context manager

```python {title="toolsets.py"}
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset


def agent_tool():
    return "I'm registered directly on the agent"


def extra_tool():
    return "I'm passed as an extra tool for a specific run"


def override_tool():
    return "I override all other tools"


agent_toolset = FunctionToolset(tools=[agent_tool]) # (1)!
extra_toolset = FunctionToolset(tools=[extra_tool])
override_toolset = FunctionToolset(tools=[override_tool])

test_model = TestModel() # (2)!
agent = Agent(test_model, toolsets=[agent_toolset])

result = agent.run_sync('What tools are available?')
print([t.name for t in test_model.last_model_request_parameters.function_tools])
#> ['agent_tool']

result = agent.run_sync('What tools are available?', toolsets=[extra_toolset])
print([t.name for t in test_model.last_model_request_parameters.function_tools])
#> ['agent_tool', 'extra_tool']

with agent.override(toolsets=[override_toolset]):
    result = agent.run_sync('What tools are available?', toolsets=[extra_toolset]) # (3)!
    print([t.name for t in test_model.last_model_request_parameters.function_tools])
    #> ['override_tool']
```

1. The [`FunctionToolset`][pydantic_ai.toolsets.FunctionToolset] will be explained in detail in the next section.
2. We're using [`TestModel`][pydantic_ai.models.test.TestModel] here because it makes it easy to see which tools were available on each run.
3. This `extra_toolset` will be ignored because we're inside an override context.

_(This example is complete, it can be run "as is")_

## Function Toolset

As the name suggests, a [`FunctionToolset`][pydantic_ai.toolsets.FunctionToolset] makes locally defined functions available as tools.

Functions can be added as tools in three different ways:

* via the [`@toolset.tool`][pydantic_ai.toolsets.FunctionToolset.tool] decorator
* via the [`tools`][pydantic_ai.toolsets.FunctionToolset.__init__] keyword argument to the constructor which can take either plain functions, or instances of [`Tool`][pydantic_ai.tools.Tool]
* via the [`toolset.add_function()`][pydantic_ai.toolsets.FunctionToolset.add_function] and [`toolset.add_tool()`][pydantic_ai.toolsets.FunctionToolset.add_tool] methods which can take a plain function or an instance of [`Tool`][pydantic_ai.tools.Tool] respectively

Functions registered in any of these ways can define an initial `ctx: RunContext` argument in order to receive the agent [context][pydantic_ai.tools.RunContext]. The `add_function()` and `add_tool()` methods can also be used from a tool function to dynamically register new tools during a run to be available in future run steps.

```python {title="function_toolset.py"}
from datetime import datetime

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset


def temperature_celsius(city: str) -> float:
    return 21.0


def temperature_fahrenheit(city: str) -> float:
    return 69.8


weather_toolset = FunctionToolset(tools=[temperature_celsius, temperature_fahrenheit])


@weather_toolset.tool
def conditions(ctx: RunContext, city: str) -> str:
    if ctx.run_step % 2 == 0:
        return "It's sunny"
    else:
        return "It's raining"


datetime_toolset = FunctionToolset()
datetime_toolset.add_function(lambda: datetime.now(), name='now')

test_model = TestModel() # (1)!
agent = Agent(test_model)

result = agent.run_sync('What tools are available?', toolsets=[weather_toolset])
print([t.name for t in test_model.last_model_request_parameters.function_tools])
#> ['temperature_celsius', 'temperature_fahrenheit', 'conditions']

result = agent.run_sync('What tools are available?', toolsets=[datetime_toolset])
print([t.name for t in test_model.last_model_request_parameters.function_tools])
#> ['now']
```

1. We're using [`TestModel`][pydantic_ai.models.test.TestModel] here because it makes it easy to see which tools were available on each run.

_(This example is complete, it can be run "as is")_

## Toolset Composition

Toolsets can be composed to dynamically filter which tools are available, modify tool definitions, or change tool execution behavior. Multiple toolsets can also be combined into one.

### Combining Toolsets

[`CombinedToolset`][pydantic_ai.toolsets.CombinedToolset] takes a list of toolsets and lets them be used as one.

```python {title="combined_toolset.py" requires="function_toolset.py"}
from function_toolset import weather_toolset, datetime_toolset

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import CombinedToolset


combined_toolset = CombinedToolset([weather_toolset, datetime_toolset])

test_model = TestModel() # (1)!
agent = Agent(test_model, toolsets=[combined_toolset])
result = agent.run_sync('What tools are available?')
print([t.name for t in test_model.last_model_request_parameters.function_tools])
#> ['temperature_celsius', 'temperature_fahrenheit', 'conditions', 'now']
```

1. We're using [`TestModel`][pydantic_ai.models.test.TestModel] here because it makes it easy to see which tools were available on each run.

_(This example is complete, it can be run "as is")_

### Filtering Tools

[`FilteredToolset`][pydantic_ai.toolsets.FilteredToolset] wraps a toolset and filters available tools ahead of each step of the run based on a user-defined function that is passed the agent [run context][pydantic_ai.tools.RunContext] and each tool's [`ToolDefinition`][pydantic_ai.tools.ToolDefinition] and returns a boolean to indicate whether or not a given tool should be available.

To easily chain different modifications, you can also call [`filtered()`][pydantic_ai.toolsets.AbstractToolset.filtered] on any toolset instead of directly constructing a `FilteredToolset`.

```python {title="filtered_toolset.py" requires="function_toolset.py,combined_toolset.py"}
from combined_toolset import combined_toolset

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

filtered_toolset = combined_toolset.filtered(lambda ctx, tool_def: 'fahrenheit' not in tool_def.name)

test_model = TestModel() # (1)!
agent = Agent(test_model, toolsets=[filtered_toolset])
result = agent.run_sync('What tools are available?')
print([t.name for t in test_model.last_model_request_parameters.function_tools])
#> ['weather_temperature_celsius', 'weather_conditions', 'datetime_now']
```

1. We're using [`TestModel`][pydantic_ai.models.test.TestModel] here because it makes it easy to see which tools were available on each run.

_(This example is complete, it can be run "as is")_

### Prefixing Tool Names

[`PrefixedToolset`][pydantic_ai.toolsets.PrefixedToolset] wraps a toolset and adds a prefix to each tool name to prevent tool name conflicts between different toolsets.

To easily chain different modifications, you can also call [`prefixed()`][pydantic_ai.toolsets.AbstractToolset.prefixed] on any toolset instead of directly constructing a `PrefixedToolset`.

```python {title="combined_toolset.py" requires="function_toolset.py"}
from function_toolset import weather_toolset, datetime_toolset

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import CombinedToolset


combined_toolset = CombinedToolset(
    [
        weather_toolset.prefixed('weather'),
        datetime_toolset.prefixed('datetime')
    ]
)

test_model = TestModel() # (1)!
agent = Agent(test_model, toolsets=[combined_toolset])
result = agent.run_sync('What tools are available?')
print([t.name for t in test_model.last_model_request_parameters.function_tools])
"""
[
    'weather_temperature_celsius',
    'weather_temperature_fahrenheit',
    'weather_conditions',
    'datetime_now',
]
"""
```

1. We're using [`TestModel`][pydantic_ai.models.test.TestModel] here because it makes it easy to see which tools were available on each run.

_(This example is complete, it can be run "as is")_

### Renaming Tools

[`RenamedToolset`][pydantic_ai.toolsets.RenamedToolset] wraps a toolset and lets you rename tools using a dictionary mapping new names to original names. This is useful when the names provided by a toolset are ambiguous or would conflict with tools defined by other toolsets, but [prefixing them](#prefixing-tool-names) creates a name that is unnecessarily long or could be confusing to the model.

To easily chain different modifications, you can also call [`renamed()`][pydantic_ai.toolsets.AbstractToolset.renamed] on any toolset instead of directly constructing a `RenamedToolset`.

```python {title="renamed_toolset.py" requires="function_toolset.py,combined_toolset.py"}
from combined_toolset import combined_toolset

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel


renamed_toolset = combined_toolset.renamed(
    {
        'current_time': 'datetime_now',
        'temperature_celsius': 'weather_temperature_celsius',
        'temperature_fahrenheit': 'weather_temperature_fahrenheit'
    }
)

test_model = TestModel() # (1)!
agent = Agent(test_model, toolsets=[renamed_toolset])
result = agent.run_sync('What tools are available?')
print([t.name for t in test_model.last_model_request_parameters.function_tools])
"""
['temperature_celsius', 'temperature_fahrenheit', 'weather_conditions', 'current_time']
"""
```

1. We're using [`TestModel`][pydantic_ai.models.test.TestModel] here because it makes it easy to see which tools were available on each run.

_(This example is complete, it can be run "as is")_

### Preparing Tool Definitions

[`PreparedToolset`][pydantic_ai.toolsets.PreparedToolset] lets you modify the entire list of available tools ahead of each step of the agent run using a user-defined function that takes the  agent [run context][pydantic_ai.tools.RunContext] and a list of [`ToolDefinition`s][pydantic_ai.tools.ToolDefinition] and returns a list of modified `ToolDefinition`s.

This is the toolset-specific equivalent of the [`prepare_tools`](tools.md#prepare-tools) argument to `Agent` that prepares all tool definitions registered to an agent across toolsets.

Note that it is not possible to add or rename tools using `PreparedToolset`. Instead, you can use [`FunctionToolset.add_function()`](#function-toolset) or [`RenamedToolset`](#renaming-tools).

To easily chain different modifications, you can also call [`prepared()`][pydantic_ai.toolsets.AbstractToolset.prepared] on any toolset instead of directly constructing a `PreparedToolset`.

```python {title="prepared_toolset.py" requires="function_toolset.py,combined_toolset.py,renamed_toolset.py"}
from dataclasses import replace
from typing import Union

from renamed_toolset import renamed_toolset

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition

descriptions = {
    'temperature_celsius': 'Get the temperature in degrees Celsius',
    'temperature_fahrenheit': 'Get the temperature in degrees Fahrenheit',
    'weather_conditions': 'Get the current weather conditions',
    'current_time': 'Get the current time',
}

async def add_descriptions(ctx: RunContext, tool_defs: list[ToolDefinition]) -> Union[list[ToolDefinition], None]:
    return [
        replace(tool_def, description=description)
        if (description := descriptions.get(tool_def.name, None))
        else tool_def
        for tool_def
        in tool_defs
    ]

prepared_toolset = renamed_toolset.prepared(add_descriptions)

test_model = TestModel() # (1)!
agent = Agent(test_model, toolsets=[prepared_toolset])
result = agent.run_sync('What tools are available?')
print(test_model.last_model_request_parameters.function_tools)
"""
[
    ToolDefinition(
        name='temperature_celsius',
        parameters_json_schema={
            'additionalProperties': False,
            'properties': {'city': {'type': 'string'}},
            'required': ['city'],
            'type': 'object',
        },
        description='Get the temperature in degrees Celsius',
    ),
    ToolDefinition(
        name='temperature_fahrenheit',
        parameters_json_schema={
            'additionalProperties': False,
            'properties': {'city': {'type': 'string'}},
            'required': ['city'],
            'type': 'object',
        },
        description='Get the temperature in degrees Fahrenheit',
    ),
    ToolDefinition(
        name='weather_conditions',
        parameters_json_schema={
            'additionalProperties': False,
            'properties': {'city': {'type': 'string'}},
            'required': ['city'],
            'type': 'object',
        },
        description='Get the current weather conditions',
    ),
    ToolDefinition(
        name='current_time',
        parameters_json_schema={
            'additionalProperties': False,
            'properties': {},
            'type': 'object',
        },
        description='Get the current time',
    ),
]
"""
```

1. We're using [`TestModel`][pydantic_ai.models.test.TestModel] here because it makes it easy to see which tools were available on each run.

### Wrapping a Toolset

[`WrapperToolset`][pydantic_ai.toolsets.WrapperToolset] wraps another toolset and delegates all responsibility to it.

It is is a no-op by default, but enables some useful abilities:

#### Changing Tool Execution

You can subclass `WrapperToolset` to change the wrapped toolset's tool execution behavior by overriding the [`call_tool()`][pydantic_ai.toolsets.AbstractToolset.call_tool] method.

```python {title="logging_toolset.py" requires="function_toolset.py,combined_toolset.py,renamed_toolset.py,prepared_toolset.py"}
import asyncio
from typing_extensions import Any

from prepared_toolset import prepared_toolset

from pydantic_ai.agent import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import WrapperToolset, ToolsetTool

LOG = []

class LoggingToolset(WrapperToolset):
    async def call_tool(self, name: str, tool_args: dict[str, Any], ctx: RunContext, tool: ToolsetTool) -> Any:
        LOG.append(f'Calling tool {name!r} with args: {tool_args!r}')
        try:
            await asyncio.sleep(0.1 * len(LOG)) # (1)!

            result = await super().call_tool(name, tool_args, ctx, tool)
            LOG.append(f'Finished calling tool {name!r} with result: {result!r}')
        except Exception as e:
            LOG.append(f'Error calling tool {name!r}: {e}')
            raise e
        else:
            return result


logging_toolset = LoggingToolset(prepared_toolset)

agent = Agent(TestModel(), toolsets=[logging_toolset]) # (2)!
result = agent.run_sync('Call all the tools')
print(LOG)
"""
[
    "Calling tool 'temperature_celsius' with args: {'city': 'a'}",
    "Calling tool 'temperature_fahrenheit' with args: {'city': 'a'}",
    "Calling tool 'weather_conditions' with args: {'city': 'a'}",
    "Calling tool 'current_time' with args: {}",
    "Finished calling tool 'temperature_celsius' with result: 21.0",
    "Finished calling tool 'temperature_fahrenheit' with result: 69.8",
    'Finished calling tool \'weather_conditions\' with result: "It\'s raining"',
    "Finished calling tool 'current_time' with result: datetime.datetime(...)",
]
"""
```

1. All docs examples are tested in CI and their their output is verified, so we need `LOG` to always have the same order whenever this code is run. Since the tools could finish in any order, we sleep an increasing amount of time based on which number tool call we are to ensure that they finish (and log) in the same order they were called in.
2. We use [`TestModel`][pydantic_ai.models.test.TestModel] here as it will automatically call each tool.

_(This example is complete, it can be run "as is")_

#### Modifying Toolsets During a Run

You can change the `WrapperToolset`'s `wrapped` property during an agent run to swap out one toolset for another starting at the next run step.

To add or remove available toolsets, you can wrap a [`CombinedToolset`](#combining-toolsets) and replace it during the run with one that can include fewer, more, or entirely different toolsets.

```python {title="wrapper_toolset.py" requires="function_toolset.py"}
from function_toolset import weather_toolset, datetime_toolset

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import WrapperToolset

togglable_toolset = WrapperToolset(weather_toolset)

test_model = TestModel() # (1)!
agent = Agent(
    test_model,
    deps_type=WrapperToolset # (2)!
)

@agent.tool
def toggle(ctx: RunContext[WrapperToolset]):
    if ctx.deps.wrapped == weather_toolset:
        ctx.deps.wrapped = datetime_toolset
    else:
        ctx.deps.wrapped = weather_toolset

result = agent.run_sync('Toggle the toolset', deps=togglable_toolset, toolsets=[togglable_toolset])
print([t.name for t in test_model.last_model_request_parameters.function_tools]) # (3)!
#> ['toggle', 'now']

result = agent.run_sync('Toggle the toolset', deps=togglable_toolset, toolsets=[togglable_toolset])
print([t.name for t in test_model.last_model_request_parameters.function_tools])
#> ['toggle', 'temperature_celsius', 'temperature_fahrenheit', 'conditions']
```

1. We're using [`TestModel`][pydantic_ai.models.test.TestModel] here because it makes it easy to see which tools were available on each run.
2. We're using the agent's dependencies to give the `toggle` tool access to the `togglable_toolset` via the `RunContext` argument.
3. This shows the available tools _after_ the `toggle` tool was executed, as the "last model request" was the one that returned the `toggle` tool result to the model.

## Building a Custom Toolset

To define a fully custom toolset with its own logic to list available tools and handle them being called, you can subclass [`AbstractToolset`][pydantic_ai.toolsets.AbstractToolset] and implement the [`get_tools()`][pydantic_ai.toolsets.AbstractToolset.get_tools] and [`call_tool()`][pydantic_ai.toolsets.AbstractToolset.call_tool] methods.

If you want to reuse a network connection or session across tool listings and calls during an agent run step, you can implement [`__aenter__()`][pydantic_ai.toolsets.AbstractToolset.__aenter__] and [`__aexit__()`][pydantic_ai.toolsets.AbstractToolset.__aexit__], which will be called when the agent that uses the toolset is itself entered using the [`async with agent`][pydantic_ai.Agent.__aenter__] context manager.

### Deferred Toolset

A deferred tool is one whose result will be produced outside of the Pydantic AI agent run in which it was called, because it depends on an upstream service (or user) or could take longer to generate than it's reasonable to keep the agent process running.

Deferred tools enable various use cases:

- Support client-side tools implemented by a web or app frontend
- Implement a Human-in-the-Loop flow where the user needs to explicitly provide an "answer" before the run can continue
- Pass slow tasks off to a background worker or external service that will send a (webhook) notification when the result is ready and the agent run can be continued.

When the model calls a deferred tool, the agent run ends with a [`DeferredToolCalls`][pydantic_ai.output.DeferredToolCalls] object containing the deferred tool call names and arguments, which are expected to be returned to the service that will (eventually) produce the result(s). Once all the results are ready, a new Pydantic AI agent run can then be started with the original run's message history plus new [`ToolReturnPart`s][pydantic_ai.messages.ToolReturnPart] (or [`RetryPromptPart`s][pydantic_ai.messages.RetryPromptPart] in case of failure) corresponding to each deferred call, after which the run will continue.

To enable an agent to call deferred tools, you create a [`DeferredToolset`][pydantic_ai.toolsets.DeferredToolset], pass it a list of [`ToolDefinition`s][pydantic_ai.tools.ToolDefinition], and provide it to the agent using one of the methods described above. Additionally, you need to add `DeferredToolCalls` to the `Agent`'s [`output_type`](output.md#structured-output) so that the possible types of the agent run output are correctly inferred. Finally, you should handle the possible `DeferredToolCalls` output by passing it to the service that will produce the results.

If your agent can also be used in a context where no deferred tools are available, you will not want to include `DeferredToolCalls` in the `output_type` passed to the `Agent` constructor as you'd have to deal with that type everywhere you use the agent. Instead, you can pass the `toolsets` and `output_type` keyword arguments when you run the agent using [`agent.run()`][pydantic_ai.Agent.run], [`agent.run_sync()`][pydantic_ai.Agent.run_sync], [`agent.run_stream()`][pydantic_ai.Agent.run_stream], or [`agent.iter()`][pydantic_ai.Agent.iter]. Note that while `toolsets` provided at this stage are additional to the toolsets provided to the constructor, the `output_type` overrides the one specified at construction time (for type inference reasons), so you'll need to include the original output types explicitly.

To demonstrate, let us first define a simple agent _without_ deferred tools:

```python {title="deferred_toolset_agent.py"}
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.toolsets.function import FunctionToolset

toolset = FunctionToolset()


@toolset.tool
def get_default_language():
    return 'en-US'


@toolset.tool
def get_user_name():
    return 'David'


class PersonalizedGreeting(BaseModel):
    greeting: str
    language_code: str


agent = Agent('openai:gpt-4o', toolsets=[toolset], output_type=PersonalizedGreeting)

result = agent.run_sync('Greet the user in a personalized way')
print(repr(result.output))
#> PersonalizedGreeting(greeting='Hello, David!', language_code='en-US')
```

Next, let's define a function that represents a hypothetical "run agent" API endpoint that can be called by the frontend and takes a list of messages to send to the model plus a list of frontend tool definitions. This is where `DeferredToolset` and `DeferredToolCalls` come in:

```python {title="deferred_toolset_api.py" requires="deferred_toolset_agent.py"}
from deferred_toolset_agent import agent, PersonalizedGreeting

from typing import Union

from pydantic_ai.output import DeferredToolCalls
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import DeferredToolset
from pydantic_ai.messages import ModelMessage

def run_agent(
    messages: list[ModelMessage] = [], frontend_tools: list[ToolDefinition] = {}
) -> tuple[Union[PersonalizedGreeting, DeferredToolCalls], list[ModelMessage]]:
    deferred_toolset = DeferredToolset(frontend_tools)
    result = agent.run_sync(
        toolsets=[deferred_toolset], # (1)!
        output_type=[agent.output_type, DeferredToolCalls], # (2)!
        message_history=messages, # (3)!
    )
    return result.output, result.new_messages()
```

1. As mentioned above, these `toolsets` are additional to those provided to the `Agent` constructor
2. As mentioned above, this `output_type` overrides the one provided to the `Agent` constructor, so we have to make sure to not lose it
3. We don't include an `user_prompt` keyword argument as we expect the frontend to provide it via `messages`

Now, imagine that the code below is implemented on the frontend, and `run_agent` stands in for an API call to the backend that runs the agent. This is where we actually execute the deferred tool calls and start a new run with the new result included:

```python {title="deferred_tools.py" requires="deferred_toolset_agent.py,deferred_toolset_api.py"}
from deferred_toolset_api import run_agent

from pydantic_ai.messages import ModelMessage, ModelRequest, RetryPromptPart, ToolReturnPart, UserPromptPart
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.output import DeferredToolCalls

frontend_tool_definitions = [
    ToolDefinition(
        name='get_preferred_language',
        parameters_json_schema={'type': 'object', 'properties': {'default_language': {'type': 'string'}}},
        description="Get the user's preferred language from their browser",
    )
]

def get_preferred_language(default_language: str) -> str:
    return 'es-MX' # (1)!

frontend_tool_functions = {'get_preferred_language': get_preferred_language}

messages: list[ModelMessage] = [
    ModelRequest(
        parts=[
            UserPromptPart(content='Greet the user in a personalized way')
        ]
    )
]

final_output = None
while True:
    output, new_messages = run_agent(messages, frontend_tool_definitions)
    messages += new_messages

    if not isinstance(output, DeferredToolCalls):
        final_output = output
        break

    print(output.tool_calls)
    """
    [
        ToolCallPart(
            tool_name='get_preferred_language',
            args={'default_language': 'en-US'},
            tool_call_id='pyd_ai_tool_call_id',
        )
    ]
    """
    for tool_call in output.tool_calls:
        if function := frontend_tool_functions.get(tool_call.tool_name):
            part = ToolReturnPart(
                tool_name=tool_call.tool_name,
                content=function(**tool_call.args_as_dict()),
                tool_call_id=tool_call.tool_call_id,
            )
        else:
            part = RetryPromptPart(
                tool_name=tool_call.tool_name,
                content=f'Unknown tool {tool_call.tool_name!r}',
                tool_call_id=tool_call.tool_call_id,
            )
        messages.append(ModelRequest(parts=[part]))

print(repr(final_output))
"""
PersonalizedGreeting(greeting='Hola, David! Espero que tengas un gran día!', language_code='es-MX')
"""
```

1. Imagine that this returns the frontend [`navigator.language`](https://developer.mozilla.org/en-US/docs/Web/API/Navigator/language).

_(This example is complete, it can be run "as is")_

## Third-Party Toolsets

### MCP Servers

See the [MCP Client](./mcp/client.md) documentation for how to use MCP servers with Pydantic AI.

### LangChain Tools {#langchain-tools}

If you'd like to use tools or a [toolkit](https://python.langchain.com/docs/concepts/tools/#toolkits) from LangChain's [community tool library](https://python.langchain.com/docs/integrations/tools/) with Pydantic AI, you can use the [`LangChainToolset`][pydantic_ai.ext.langchain.LangChainToolset] which takes a list of LangChain tools. Note that Pydantic AI will not validate the arguments in this case -- it's up to the model to provide arguments matching the schema specified by the LangChain tool, and up to the LangChain tool to raise an error if the arguments are invalid.

You will need to install the `langchain-community` package and any others required by the tools in question.

```python {test="skip"}
from langchain_community.agent_toolkits import SlackToolkit

from pydantic_ai import Agent
from pydantic_ai.ext.langchain import LangChainToolset


toolkit = SlackToolkit()
toolset = LangChainToolset(toolkit.get_tools())

agent = Agent('openai:gpt-4o', toolsets=[toolset])
# ...
```

### ACI.dev Tools {#aci-tools}

If you'd like to use tools from the [ACI.dev tool library](https://www.aci.dev/tools) with Pydantic AI, you can use the [`ACIToolset`][pydantic_ai.ext.aci.ACIToolset] [toolset](toolsets.md) which takes a list of ACI tool names as well as the `linked_account_owner_id`. Note that Pydantic AI will not validate the arguments in this case -- it's up to the model to provide arguments matching the schema specified by the ACI tool, and up to the ACI tool to raise an error if the arguments are invalid.

You will need to install the `aci-sdk` package, set your ACI API key in the `ACI_API_KEY` environment variable, and pass your ACI "linked account owner ID" to the function.

```python {test="skip"}
import os

from pydantic_ai import Agent
from pydantic_ai.ext.aci import ACIToolset


toolset = ACIToolset(
    [
        'OPEN_WEATHER_MAP__CURRENT_WEATHER',
        'OPEN_WEATHER_MAP__FORECAST',
    ],
    linked_account_owner_id=os.getenv('LINKED_ACCOUNT_OWNER_ID'),
)

agent = Agent('openai:gpt-4o', toolsets=[toolset])
```
