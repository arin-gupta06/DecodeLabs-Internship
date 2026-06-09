

def Rule_based_Chatbot():
    print("Welcome to the Rule-based Chatbot!")
    str = input("Please enter your message: ")
    name = input("Please enter your name: ")
    str = str.lower()
    if "hello" in str or "hi" in str:
        print("Hello, " + name + "! How can I assist you today?")
    elif "how are you" in str:
        print("I'm doing well, thank you! How about you, " + name + "?")
    elif "what is your name" in str:
        print("My name is Chatbot. Nice to meet you, " + name + "!")
    else:
        print("I'm sorry, I didn't understand that. Can you please rephrase?")


Rule_based_Chatbot()