import os
import discord
from discord.ext import commands
from pathlib import Path
from aitextgen import aitextgen
from datetime import datetime
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

token = ''
bot = commands.Bot(command_prefix='!')

ai = aitextgen(model_folder="trained_model")

filename = './model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
vectorizer_filename = "./vectorizer.sav"
loaded_vectorizer = pickle.load(open(vectorizer_filename, 'rb'))

openai.organization = ""
openai.api_key = ""
openai.Model.list()

@bot.command()
async def hello(ctx, arg=None):
  await ctx.send("hello! I'm test-bot!")

@bot.command()
async def bye(ctx, arg=None):
  await ctx.send("Goodbye! See you later.")

@bot.command()
async def inquiry(ctx, *arg):
    '''
    Function: generates text based on a prompt. 
    '''
    if arg is not None:
        the_prompt = ' '.join(arg)
        if isinstance(the_prompt, str):
            answer = ai.generate_one(prompt = the_prompt)
            answer=answer.split("?", 1)[1]
            await ctx.send(answer)
        else:
            await ctx.send("Prompt should be a string. Not sure where this went wrong.")

@bot.command()
async def short_advice(ctx, *arg):

    '''
    Function: generates short advice based on a prompt. 
    '''
    if arg is not None:
        the_prompt = ' '.join(arg)
        if isinstance(the_prompt, str):
          prompt = loaded_vectorizer.transform([the_prompt])
          sentiment=loaded_model.predict(prompt)

          prompt = "Write a sentence of advice for someone that is feeling " + sentiment + " and tells you: " + the_prompt
          prompt = prompt[0]

          response = openai.Completion.create(
          engine="text-curie-001",
          prompt=prompt,
          temperature=0.5,
          max_tokens=256,
          top_p=1.0,
          frequency_penalty=0.0,
          presence_penalty=0.0
        )
          await ctx.send(response['choices'][0]['text'])
        else:
            await ctx.send("Prompt should be a string. Not sure where this went wrong.")

@bot.command()
async def advice(ctx, *arg):

    '''
    Function: generates some advice based on a prompt. 
    '''
    if arg is not None:
        the_prompt = ' '.join(arg)
        if isinstance(the_prompt, str):
          prompt = loaded_vectorizer.transform([the_prompt])
          sentiment=loaded_model.predict(prompt)

          prompt = "Write some advice for someone that is feeling " + sentiment + " and tells you: " + the_prompt
          prompt = prompt[0]

          response = openai.Completion.create(
          engine="text-curie-001",
          prompt=prompt,
          temperature=0.5,
          max_tokens=256,
          top_p=1.0,
          frequency_penalty=0.0,
          presence_penalty=0.0
        )
          await ctx.send(response['choices'][0]['text'])
        else:
            await ctx.send("Prompt should be a string. Not sure where this went wrong.")

@bot.command()
async def long_advice(ctx, *arg):

    '''
    Function: generates a paragraph of advice based on a prompt. 
    '''
    if arg is not None:
        the_prompt = ' '.join(arg)
        if isinstance(the_prompt, str):
          prompt = loaded_vectorizer.transform([the_prompt])
          sentiment=loaded_model.predict(prompt)

          prompt = "Write a paragraph of advice for someone that is feeling " + sentiment + " and tells you: " + the_prompt
          prompt = prompt[0]

          response = openai.Completion.create(
          engine="text-curie-001",
          prompt=prompt,
          temperature=0.5,
          max_tokens=256,
          top_p=1.0,
          frequency_penalty=0.0,
          presence_penalty=0.0
        )
          await ctx.send(response['choices'][0]['text'])
        else:
            await ctx.send("Prompt should be a string. Not sure where this went wrong.")

@bot.command()
async def essay_advice(ctx, *arg):

    '''
    Function: generates an essay of advice based on a prompt. 
    '''
    if arg is not None:
        the_prompt = ' '.join(arg)
        if isinstance(the_prompt, str):
          prompt = loaded_vectorizer.transform([the_prompt])
          sentiment=loaded_model.predict(prompt)

          prompt = "Write an essay of advice for someone that is feeling " + sentiment + " and tells you: " + the_prompt
          prompt = prompt[0]

          response = openai.Completion.create(
          engine="text-curie-001",
          prompt=prompt,
          temperature=0.5,
          max_tokens=256,
          top_p=1.0,
          frequency_penalty=0.0,
          presence_penalty=0.0
        )
          await ctx.send(response['choices'][0]['text'])
        else:
            await ctx.send("Prompt should be a string. Not sure where this went wrong.")

@bot.command()
async def story(ctx, *arg):

    '''
    Function: generates a terrible story. 
    '''
    prompt = "Write a good story about unicorns"
    prompt = prompt[0]

    response = openai.Completion.create(
    engine="text-curie-001",
    prompt=prompt,
    temperature=0.4,
    max_tokens=256,
    top_p=0.9,
    frequency_penalty=0.0,
    presence_penalty=0.0
  )
    await ctx.send(response['choices'][0]['text'])
       

bot.run(token)