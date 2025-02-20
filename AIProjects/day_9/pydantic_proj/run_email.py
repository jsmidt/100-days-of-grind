from __future__ import annotations as _annotations

from dataclasses import dataclass, field
from pprint import pprint


from pydantic import BaseModel, EmailStr

from pydantic_ai import Agent
from pydantic_ai.format_as_xml import format_as_xml
from pydantic_ai.messages import ModelMessage, SystemPromptPart, UserPromptPart, ToolCallPart, ToolReturnPart
from pydantic_graph import BaseNode, End, Graph, GraphRunContext
import asyncio




@dataclass
class C:
    mylist: list[int] = field(default_factory=list)

c = C()

print (c.mylist)


@dataclass
class User:
    name: str
    email: EmailStr
    interests: list[str]


@dataclass
class Email:
    subject: str
    body: str


@dataclass
class State:
    user: User
    write_agent_messages: list[ModelMessage] = field(default_factory=list)


email_writer_agent = Agent(
    'google-gla:gemini-2.0-flash-001',
    result_type=Email,
    system_prompt='Write a welcome email to our tech blog.',
)


@dataclass
class WriteEmail(BaseNode[State]):
    email_feedback: str | None = None

    async def run(self, ctx: GraphRunContext[State]) -> Feedback:
        if self.email_feedback:
            prompt = (
                f'Rewrite the email for the user:\n'
                f'{format_as_xml(ctx.state.user)}\n'
                f'Feedback: {self.email_feedback}'
            )
        else:
            prompt = (
                f'Write a welcome email for the user:\n'
                f'{format_as_xml(ctx.state.user)}'
            )
        print (' * prompt:', prompt)

        result = await email_writer_agent.run(
            prompt,
            message_history=ctx.state.write_agent_messages,
        )
        ctx.state.write_agent_messages += result.all_messages()

        print ('\n *** All messages ***\n')
        for m in result.all_messages():
            print ()
            
            for p in m.parts:
                
                if (isinstance(p,SystemPromptPart)):
                    print (" * SystemPromptPart:", p.content)
                elif (isinstance(p,UserPromptPart)):
                    print (" * UserPromptPart:", p.content)
                elif (isinstance(p,ToolCallPart)):
                    print (f" * ToolCallPart: called {p.tool_name} with args {p.args}")
                elif (isinstance(p,ToolReturnPart)):
                    print (f" * ToolReturnPart: called {p.tool_name} with content {p.content}")
                else:
                    print (' - Not anything special:',type(p))

        print ('\n *** New messages ***\n')
        for m in result.new_messages():
            print ()
            for p in m.parts:
                
                if (isinstance(p,SystemPromptPart)):
                    print (" * SystemPromptPart:", p.content)
                elif (isinstance(p,UserPromptPart)):
                    print (" * UserPromptPart:", p.content)
                elif (isinstance(p,ToolCallPart)):
                    print (f" * ToolCallPart: called {p.tool_name} with args {p.args}")
                elif (isinstance(p,ToolReturnPart)):
                    print (f" * ToolReturnPart: called {p.tool_name} with content {p.content}")
                else:
                    print (' - Not anything special:',type(p))

        print ('********:',result.data)
        return Feedback(result.data)


class EmailRequiresWrite(BaseModel):
    feedback: str


class EmailOk(BaseModel):
    pass


feedback_agent = Agent[None, EmailRequiresWrite | EmailOk](
    'google-gla:gemini-2.0-flash-001',
    result_type=EmailRequiresWrite | EmailOk,  # type: ignore
    system_prompt=(
        'Review the email and provide feedback, email must reference the users specific interests.'
    ),
)


@dataclass
class Feedback(BaseNode[State, None, Email]):
    email: Email

    async def run(
        self,
        ctx: GraphRunContext[State],
    ) -> WriteEmail | End[Email]:
        # Run it on your complex object

        prompt = format_as_xml({'user': ctx.state.user, 'email': self.email})
        result = await feedback_agent.run(prompt, message_history=ctx.state.write_agent_messages,)
        print ('\n *** All messages ***\n')
        for m in result.all_messages():
            print ()
            
            for p in m.parts:
                
                if (isinstance(p,SystemPromptPart)):
                    print (" * SystemPromptPart:", p.content)
                elif (isinstance(p,UserPromptPart)):
                    print (" * UserPromptPart:", p.content)
                elif (isinstance(p,ToolCallPart)):
                    print (f" * ToolCallPart: called {p.tool_name} with args {p.args}")
                elif (isinstance(p,ToolReturnPart)):
                    print (f" * ToolReturnPart: called {p.tool_name} with content {p.content}")
                else:
                    print (' - Not anything special:',type(p))

        print ('\n *** New messages ***\n')
        for m in result.new_messages():
            print ()
            for p in m.parts:
                
                if (isinstance(p,SystemPromptPart)):
                    print (" * SystemPromptPart:", p.content)
                elif (isinstance(p,UserPromptPart)):
                    print (" * UserPromptPart:", p.content)
                elif (isinstance(p,ToolCallPart)):
                    print (f" * ToolCallPart: called {p.tool_name} with args {p.args}")
                elif (isinstance(p,ToolReturnPart)):
                    print (f" * ToolReturnPart: called {p.tool_name} with content {p.content}")
                else:
                    print (' - Not anything special:',type(p))
        if isinstance(result.data, EmailRequiresWrite):
            return WriteEmail(email_feedback=result.data.feedback)
        else:
            return End(self.email)


async def main():
    user = User(
        name='John Doe',
        email='john.joe@example.com',
        interests=['Haskel', 'Lisp', 'Fortran'],
    )
    state = State(user)
    feedback_graph = Graph(nodes=(WriteEmail, Feedback))
    feedback_graph.mermaid_image(start_node=WriteEmail,image_type='png')
    feedback_graph.mermaid_save('mygraph.png')
    email, _ = await feedback_graph.run(WriteEmail(), state=state)
    print(email)
    """
    Email(
        subject='Welcome to our tech blog!',
        body='Hello John, Welcome to our tech blog! ...',
    )
    """




asyncio.run(main())