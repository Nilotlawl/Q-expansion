import streamlit as st
from src.model import resolve_coreferences, refine_query, classify_topic

def main_app():
    st.title("Chat-Based Query expansion and Topic classification")
    
    if "dialogue_history" not in st.session_state:
        st.session_state.dialogue_history = []
        
    user_input = st.text_input("Enter your message: ")
    
    if st.button("Send"):
        if user_input:
            st.session_state.dialogue_history.append(user_input)
            
            st.write("### Dialogue History:")
            for msg in st.session_state.dialogue_history:
                st.write("- ", msg)
            
            # Process the dialogue only if there are at least 2 messages
            if len(st.session_state.dialogue_history) >= 2:
                resolved_query = resolve_coreferences(st.session_state.dialogue_history)
                expanded_query = refine_query(resolved_query)
                topic = classify_topic(expanded_query)
                
                st.write("**Resolved Query:**", resolved_query)
                st.write("**Expanded Query:**", expanded_query)
                st.write("**Assigned Topic:**", topic)
                
        else:
            st.warning("Please enter a message.")
            
if __name__ == "__main__":
    main_app()