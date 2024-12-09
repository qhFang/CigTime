from openai import AzureOpenAI
import argparse


def generate_captions(client):

  response = client.chat.completions.create(
    model="vision",
    messages=[
      {"role": "system", 
       "content": "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture."
      },
      {
        "role": "user",
        "content": [
          {"type": "text", "text": ("We need to generate some instructions for human motion editing. The format of the instruction is as follows:\n"
                                    "Body part: Editing target\n"
                                    "There are two choices for the body parts, upper body and lower body.  The editing target are free-type, you can use any reasonable actions.\n"
                                    "I will provide some examples next:\n"
                                    "### Examples ###\n"
                                    "Upper body: A person throws a ball.\n"
                                    "Upper body: A person plays basketball with his right hand.\n"
                                    "Upper body: A person bends down to pick up things.\n"
                                    "Lower body: A person walks forward\n"
                                    "Lower body: A person jumps several times\n"
                                    "Lower body: A person kicks with his right leg.\n"
                                    "Lower body: A person kicks with his left leg.\n"
                                    "### End ###\n"
                                    "I hope you can generate 100 more instructions. Do not generate complex actions. You can output some instructions related to the left and right arms, left and right legs, or other parts of the body. However, be careful not to output actions that have high demands on the fingers, as our model cannot output finger movements. Don't reply to anything else, just provide new instructions.")},
        ],
      },
    ],
    max_tokens=4096,
  )
  instructions = response.choices[0].message.content
  return instructions

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--api_key", required=True, help="The API key for the Azure OpenAI service.")
  parser.add_argument("--api_version", default="2023-12-01-preview", help="The API version to use.")
  parser.add_argument("--output_file", required=True, type=str, help="The output file to write the generated instructions to.")

  args = parser.parse_args()
    
  azure_endpoint="https://gpt4v-0.openai.azure.com/"
  client = AzureOpenAI(
    api_key=args.api_key,
    api_version=args.api_version,
    azure_endpoint=azure_endpoint
  )

  instructions = generate_captions(client)
  with open(args.output_file, "w") as f:
    f.write(instructions)