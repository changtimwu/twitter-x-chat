import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import vertexai.preview.generative_models as generative_models
import sys

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

class multimodelchat:
    def __init__(self, vidfn):
        self.responses = []
        vertexai.init(project="instant-cloud-422311", location="asia-east1")
        model = GenerativeModel(
          "gemini-1.5-flash-001" #"gemini-1.5-pro-001",
        )
        self.model = model
        self.initprompt = "Provide a summary of the video, including its key points and corresponding timestamps in Chinese"
        part = self.setup_video( vidfn)
        self.chatctx = setup_chatctx( part)

    def setup_chatctx(self,part):
      chatctx = self.model.start_chat(response_validation=False) 
      res = chatctx.send_message(
          [part, self.initprompt],
          generation_config=generation_config,
          safety_settings=safety_settings
      )
      self.responses.append( res)
      return chatctx

    def setup_video( self, vidfn):
        with open( vidfn, "rb") as f:
            mp4_data = f.read()
        part1_1 = Part.from_data(
            mime_type="video/mp4", #mime_type="image/png",
            #gcs_source="gs://my-bucket/videos/my-video.mp4"
            data=base64.b64encode(mp4_data).decode("utf-8")
        )
        return part1_1
    def interactive_chat(self):
        while True:
            res = self.responses.pop(0)
            print(res)
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("Chat ended.")
                break
            self.chatctx.send_message
            res = self.chatctx.send_message(
              [user_input],
              generation_config=generation_config,
              safety_settings=safety_settings
            )
            self.responses.append(res)
     

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} <video file>')
        sys.exit()
    vidfn = sys.argv[1]
    rmsg = chat_video(vidfn)
    print(rmsg)
