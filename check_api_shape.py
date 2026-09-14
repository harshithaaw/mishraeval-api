from gradio_client import Client

client = Client("harshithaaa06/mishraeval")
result = client.predict(
    user_message="Hello!!",
    bot_response="Hello!!",
    api_name="/run_prediction",
)
print(result)
print(type(result))
for i, item in enumerate(result):
    print(i, type(item), item)
