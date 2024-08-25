from unittest import mock

from app_modes.chat import ChatStateMachine, RestartState, UserPrefsInputState


@mock.patch.object(UserPrefsInputState, "_llm")
def test_chat_machine(mock_llm):

    def _llm_side_effect(*args, **kwargs):
        return "llm return"

    mock_llm.side_effect = _llm_side_effect

    state_machine = ChatStateMachine(history=mock.MagicMock())

    assert type(state_machine.current_state) is UserPrefsInputState
    assert state_machine.run(input=None) == "How big do you want your house to be?"
    assert type(state_machine.current_state) is UserPrefsInputState

    assert (
        state_machine.run(input=None)
        == "What are 3 most important things for you in choosing this property?"
    )
    assert type(state_machine.current_state) is UserPrefsInputState

    assert state_machine.run(input=None) == "Which amenities would you like?"
    assert type(state_machine.current_state) is UserPrefsInputState

    assert (
        state_machine.run(input=None)
        == "Which transportation options are important to you?"
    )
    assert type(state_machine.current_state) is UserPrefsInputState

    assert (
        state_machine.run(input=None)
        == "How urban do you want your neighborhood to be?"
    )
    assert type(state_machine.current_state) is UserPrefsInputState

    assert (
        state_machine.run(input=None)
        == "Please upload a picture of the living room that looks close enough to your expectation (Optional)"
    )
    assert type(state_machine.current_state) is UserPrefsInputState

    assert state_machine.run(input=None) == [
        "llm return",
        "I hope this was helpful to you. Please feel free to retry!",
    ]
    assert type(state_machine.current_state) is RestartState
