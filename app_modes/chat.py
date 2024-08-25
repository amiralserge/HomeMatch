import abc
import base64
import logging
from io import BufferedReader
from typing import Any, Dict, Tuple, Union

import gradio as gr
import PIL
import PIL.Image
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.documents.base import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from typing_extensions import Self

from config import CONFIG
from models.listings import Listing
from service_layer.services import get_listing_by_id, get_relevant_listings

from . import register_app_mode

_logger = logging.getLogger(__name__)


class ChatState:
    is_terminal: bool = False

    @abc.abstractmethod
    def run(self, history: ChatMessageHistory, user_input: Any) -> str:
        raise NotImplementedError()

    @abc.abstractmethod
    def next(self) -> Self | None:
        raise NotImplementedError()


class AbtractInputQuestionState(ChatState):
    def __init__(self, question: str) -> None:
        self.question = question

    def run(self, history: ChatMessageHistory, user_input: Any):
        print(f"{self.question} : {user_input}")
        if not user_input:
            return None
        self._process_input(history, user_input)

    @abc.abstractmethod
    def _process_input(self, history: ChatMessageHistory):
        raise NotImplementedError()


class TextInputQuestion(AbtractInputQuestionState):
    def __init__(self, question: str) -> None:
        super().__init__(question)

    def _process_input(self, history: ChatMessageHistory, user_input: Any) -> None:
        history.add_ai_message(self.question)
        history.add_user_message(
            user_input["text"] if isinstance(user_input, dict) else user_input
        )


class FileInputQuestionState(AbtractInputQuestionState):
    def __init__(self, question: str, allowed_extensions) -> None:
        super().__init__(question)
        self._allowed_extentions = allowed_extensions
        self.input_files = []

    def _process_input(self, history: ChatMessageHistory, user_input: Any) -> None:
        files = user_input.get("files") or [] if isinstance(user_input, dict) else None
        if not files:
            _logger.debug(f"input ignored for {self.question}")
            return
        self.input_inputs = list(
            map(self._read_file, [f["path"] for f in files if self._is_valid(f)])
        )

    def _read_file(self, file_path) -> BufferedReader:
        return open(file_path, "rb")

    def _is_valid(self, file_input: dict) -> bool:
        file_ext = (file_input.get("mime_type") or "").split("/")[-1]
        return file_ext in self._allowed_extentions


class ImageInputQuestionState(FileInputQuestionState):
    def __init__(self, question: str) -> None:
        super().__init__(
            question,
            allowed_extensions=[
                "jpeg",
                "jpg",
            ],
        )

    def _read_file(self, file_path) -> PIL.Image.Image:
        return PIL.Image.open(file_path)


class UserPrefsInputState(ChatState):
    _substates = [
        TextInputQuestion("How big do you want your house to be?"),
        TextInputQuestion(
            "What are 3 most important things for you in choosing this property?"
        ),
        TextInputQuestion("Which amenities would you like?"),
        TextInputQuestion("Which transportation options are important to you?"),
        TextInputQuestion("How urban do you want your neighborhood to be?"),
        ImageInputQuestionState(
            "Please upload a picture of the living room that looks close enough to your expectation (Optional)",
        ),
    ]

    _listing_rendering_template = """
<div style="padding: 10px;">
    <div style="margin: 10px;">
        <img src="{image_uri}"/>
    </div>
    <div style="margin: 10px;">
        <p>{description}</p>
    </div>
    <ul>
        <li>Neighborhood: {neighborhood}</li>
        <li>Price: ${price}</li>
        <li>Bedrooms: {bedrooms}</li>
        <li>Bathrooms: {bathrooms}</li>
        <li>House Size: {house_size} sqft</li>
        <li>Neighborhood Description: {neighborhood_description}</li>
    </ul>
</div>
    """

    _llm_query = """
Given the real estate listings in the CONTEXT section,
For each real estate listing,
generate a personalized descriptions based on the human preferences in the QUESTIONS ANWERS SUMMARY section
    and the description of the listing.
The descriptions should be unique, appealing, and tailored to the preferences provided,
 emphasizing aspects of the property that align with those preferences.
RETURN INSTRUCTIONS: a json object array. The attributes are id(the listing id), and description(your personalized description)"""

    _llm_query_prompt_template = """
{query}
---QUESTIONS ANWERS SUMMARY
{questions_and_answers}
---QUESTIONS ANSWERS SUMMARY END
CONTEXT: {context}
    """

    def __init__(self) -> None:
        super().__init__()
        self._substates_number = len(self._substates)
        self.current_state_index = -1

    def run(self, history: ChatMessageHistory, user_input: Dict) -> Any:
        # if we are at the first run, ingore the input and just return the question
        if self.current_state_index == -1:
            self.current_state_index = 0
            return self._substates[self.current_state_index].question
        # otherwise run the treatment for the answer(the user_input)
        self._substates[self.current_state_index].run(history, user_input)
        # increment state index
        self.current_state_index += 1
        if self.current_state_index >= self._substates_number:
            return [
                self._llm(history),
                "I hope this was helpful to you. Please feel free to retry!",
            ]
        return self._substates[self.current_state_index].question

    def _llm(self, history: ChatMessageHistory) -> Any:
        text_input, image = self._extract_user_input(history)
        relevant_listings, response = self._query_llm(history, text_input, image)
        return self._process_llm_response(response, relevant_listings)

    def _query_llm(self, history, text_input, image):
        relevant_listings = get_relevant_listings(
            text=text_input, image=image, columns=["id", "listing_summary"]
        )
        llm_chain = self._build_llm_query_chain(history)
        response = llm_chain.invoke(
            {
                "input_documents": relevant_listings,
                "query": self._llm_query,
            }
        ).get("output_text")
        return relevant_listings, response

    def _extract_user_input(self, history) -> Tuple[str | None, PIL.Image.Image | None]:
        human_messages = filter(lambda m: type(m) is HumanMessage, history.messages)
        text = "\n".join(map(lambda m: m.content, human_messages))
        image_states = list(
            filter(lambda s: type(s) is ImageInputQuestionState, self._substates)
        )
        images = [s.input_files[0] for s in image_states if s.input_files]
        return text, images[0] if images else None

    def _build_llm_query_chain(
        self, history: ChatMessageHistory
    ) -> BaseCombineDocumentsChain:
        conversational_memory = ConversationBufferMemory(
            chat_memory=history,
            memory_key="questions_and_answers",
            input_key="query",
        )
        prompt = PromptTemplate(
            template=self._llm_query_prompt_template,
            input_variables=["query", "context", "questions_and_answers"],
        )
        return load_qa_chain(
            llm=ChatOpenAI(
                name=CONFIG.llm_model, max_tokens=CONFIG.max_tokens, temperature=1
            ),
            prompt=prompt,
            chain_type="stuff",
            memory=conversational_memory,
        )

    def _process_llm_response(
        self, llm_response: str, submitted_listings: list[Document]
    ) -> gr.HTML:
        try:
            descriptions_data = JsonOutputParser().parse(llm_response)
        except Exception as e:
            _logger.exception(e)
            return "Sorry ! Unfortunately, I was unable to process your request due to some technical issue; I may need to get fixed up!"

        res = []
        listing_fields = list(
            set(Listing.field_names()) - {"vector", "image_vector", "description"}
        )
        for listing, description_data in zip(submitted_listings, descriptions_data):
            listing_info = dict(
                get_listing_by_id(
                    id=listing.metadata.get("id"), columns=listing_fields
                )[0].metadata
            )
            image_bytes = listing_info.pop("image")
            res.append(
                self._listing_rendering_template.format(
                    description=description_data["description"],
                    image_uri=f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}",
                    **listing_info,
                )
            )
        return gr.HTML(
            "<br/><hr/><br/>".join(res)
            + "<br/><hr/><br/>Type In Anything or Nothing to Restart"
        )

    def next(self) -> Union[Any, Self]:
        if self.current_state_index >= self._substates_number:
            self.current_state_index = -1
            return RestartState()
        return self


class RestartState(TextInputQuestion):
    is_terminal = True

    def __init__(self) -> None:
        super().__init__(question="Press enter to restart or type in `No` to stop")
        self._answer = None

    def run(self, history: ChatMessageHistory, user_input: Any):
        if user_input["text"].lower() == "no":
            return "Goodbye! You are welcome any time :)"

    def next(self) -> Self | None:
        if self._answer and self._answer.lower() == "no":
            return UserPrefsInputState()


class ChatStateMachine:
    def __init__(self, history: ChatMessageHistory) -> None:
        self.initial_state = UserPrefsInputState
        self.current_state: ChatState = self.initial_state()
        self.chat_history: ChatMessageHistory = history

    def run(self, input) -> str:
        res = self.current_state.run(self.chat_history, input)
        self.current_state = self.current_state.next()
        return res

    def reset(self) -> None:
        self.current_state = self.initial_state()
        self.chat_history.clear()

    @property
    def is_current_state_terminal(self) -> bool:
        if self.current_state:
            return self.current_state.__class__.is_terminal
        return False


def run():

    chat_state_machine = ChatStateMachine(history=ChatMessageHistory())

    def submit_message(history, message):
        for x in message["files"]:
            history.append(((x,), None))

        if message["text"] is not None:
            history.append([message["text"], None])

        res = chat_state_machine.run(message)
        for msg in list(res) if isinstance(res, str) else res:
            history.append((None, msg))

        is_terminal_state = chat_state_machine.is_current_state_terminal
        restart_btn = gr.Button(visible=is_terminal_state)
        chat_input = gr.MultimodalTextbox(
            value=None,
            interactive=not is_terminal_state,
            visible=not is_terminal_state,
        )
        return history, chat_input, restart_btn

    def reset_chat(chatbot: gr.Chatbot, chat_input: gr.MultimodalTextbox):
        chat_state_machine.reset()
        return (
            chat_state_machine.run(None),
            gr.MultimodalTextbox(visible=True, interactive=True),
            gr.Button(visible=False),
        )

    with gr.Blocks(fill_height=True) as app:
        gr.Markdown(
            "<h1 style='text-align: center; margin-bottom: 1rem'>Real Estate Assistant</h1>"
        )
        gr.Markdown("description")
        chatbot = gr.Chatbot(
            elem_id="chatbot",
            bubble_full_width=False,
            likeable=False,
            scale=1,
        )
        with gr.Blocks(fill_width=True):
            restart_btn = gr.Button("Restart", visible=False)

        chat_input = gr.MultimodalTextbox(
            interactive=True,
            visible=True,
            placeholder="Enter message or upload file...",
            show_label=False,
        )
        restart_btn.click(
            reset_chat,
            inputs=[chatbot, chat_input],
            outputs=[chatbot, chat_input, restart_btn],
        )

        chat_input.submit(
            submit_message,
            inputs=[chatbot, chat_input],
            outputs=[chatbot, chat_input, restart_btn],
            scroll_to_output=True,
        )

        app.unload(fn=lambda: chat_state_machine.reset())

    app.launch()


register_app_mode(run, "chat")
