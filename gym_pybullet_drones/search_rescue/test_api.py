import os
import google.generativeai as genai

try:
    # Retrieve the API key from the environment variable.
    api_key = os.environ.get("GEMINI_API_KEY")

    # Check if the API key is set.  If not, raise an error.
    if not api_key:
        raise ValueError(
            "Gemini API key not found.  Please set the GEMINI_API_KEY environment variable."
        )

    genai.configure(api_key=api_key)

    # Create the model
    generation_config = {
        "temperature": 0.9,  # Adjust as needed
        "top_p": 1.0,
        "top_k": 1,
        "max_output_tokens": 2048,  # Adjust as needed
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro-latest",  # Or another suitable model
        generation_config=generation_config,
    )

    chat = model.start_chat()
    print("sending message")
    response = chat.send_message("Qual é meu parça?.")
    print("Receiving message")
    print(response.text)


except ValueError as e:
    print(f"Error: {e}")
    print(
        "Please ensure you have set the GEMINI_API_KEY environment variable "
        "with your Gemini API key."
    )
except Exception as e:
    print(f"An unexpected error occurred: {e}")