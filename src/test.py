#ref https://github.com/AI-Yash/st-chat

import streamlit as st
from streamlit_chat import message


class Chat_UI:
    def __init__(self, is_debugging=False, user_data=None, bot_data=None):
        # 과거 대화 내역 가져오기
        st.session_state.setdefault(
            'past', 
            []
        )
        st.session_state.setdefault(
            'generated', 
            []
        )
        self.is_debugging = is_debugging

    def __on_input_change(self):
        user_input = st.session_state.user_input
        st.session_state.past.append(user_input) #user메시지 올라감
        st.session_state["user_input"] = ""
        st.session_state.generated.append("test") #bot메시지 올라감

    def __on_btn_click(self):
        del st.session_state.past[:]
        del st.session_state.generated[:]

    def __saving_chat(self):
        pass

    def get_bot_answer(self, data):
        st.session_state.generated.append(data)


    def running_chat(self):
        chat_placeholder = st.empty()

        with chat_placeholder.container():    
            for i in range(len(st.session_state['generated'])):                
                message(st.session_state['past'][i], is_user=True, key=f"{i}_user")
                message(st.session_state['generated'][i],  key=f"{i}")

            if self.is_debugging:
                st.button("Clear message", on_click=self.__on_btn_click)

        with st.container():
            st.text_input("User Input:", on_change=self.__on_input_change, key="user_input")


if __name__ == "__main__":
    chat = Chat_UI(is_debugging=False)
    chat.running_chat()
