# src/main.py

from model import resolve_coreferences, refine_query, classify_topic

def main():
    # Example dialogue session
    dialogue_history = [
        "Who is the PM of India?",
        "What are his duties?"
    ]
    
    print("Dialogue History:")
    for msg in dialogue_history:
        print("User:", msg)
    
    # Step 1: Coreference Resolution
    resolved_query = resolve_coreferences(dialogue_history)
    print("\nResolved Query:")
    print(resolved_query)
    
    # Step 2: Query Expansion using T5
    expanded_query = refine_query(resolved_query)
    print("\nExpanded Query:")
    print(expanded_query)
    
    # Step 3: Topic Classification
    topic = classify_topic(expanded_query)
    print("\nAssigned Topic:")
    print(topic)

if __name__ == "__main__":
    main()
