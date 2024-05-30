import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import vertexai.preview.generative_models as generative_models
import sys
from timeit import default_timer as timer


generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
}

class multimodalchat:
    def __init__(self, vidfn):
        self.responses = []
        vertexai.init(project="instant-cloud-422311", location="asia-east1")
        model = GenerativeModel(
          "gemini-1.5-flash-001" #"gemini-1.5-pro-001",
        )
        self.model = model
        self.initprompt = "Provide a summary of the media, including its key points and corresponding timestamps in Chinese"
        if vidfn.endswith('.m4a'):
            self.initprompt = "Provide full subtitle of the media in the VTT format"
        st = timer()
        part = self.setup_video( vidfn)
        print( f'media setup sent {timer() - st}')
        r = model.count_tokens([part])
        print(f"Prompt Token Count: {r.total_tokens}")
        print(f"Prompt Character Count: {r.total_billable_characters}")
 
        self.chatctx = self.setup_chatctx( part)

    def setup_chatctx(self,part):
        chatctx = self.model.start_chat(response_validation=False)
        st = timer()
        res = chatctx.send_message(
          [part, self.initprompt],
          generation_config=generation_config,
          safety_settings=safety_settings
        )
        print(f'\ntakes { timer() - st} to response')
        """
        {
          "text": "Hello, how can I help you today?",
          "tokens": ["Hello", ",", "how", "can", "I", "help", "you", "today", "?"],
          "logprobs": [-1.234, -0.567, -0.890, -0.456, -0.345, -0.234, -0.123, -0.012, -0.001],
          "chat_id": "1234567890",
          "conversation_id": "9876543210",
          "message_id": "1011121314",
          "model_version": "gemini-1.5-flash-001",
          "timestamp": "2023-10-26T18:30:00Z",
          "error": null,
          "warning": null
        }
        """
        self.responses.append( res)
        return chatctx

    def setup_video( self, vidfn):
        with open( vidfn, "rb") as f:
            mp4_data = f.read()
        mime_type = 'audio/mp4' if vidfn.endswith('m4a') else 'video/mp4'
        part1_1 = Part.from_data(
            mime_type=mime_type, #mime_type="image/png",
            #gcs_source="gs://my-bucket/videos/my-video.mp4"
            data=base64.b64encode(mp4_data).decode("utf-8")
        )
        return part1_1
    def interactive(self):
        while True:
            res = self.responses.pop(0)
            print(res.text)
            print(res.usage_metadata)
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("Chat ended.")
                break
            st = timer()
            res = self.chatctx.send_message(
              [user_input],
              generation_config=generation_config,
              safety_settings=safety_settings
            )
            print(f'\ntakes { timer() - st} to response')
            self.responses.append(res)
     

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} <media file>')
        sys.exit()
    vidfn = sys.argv[1]
    mmc = multimodalchat(vidfn)
    mmc.interactive()
