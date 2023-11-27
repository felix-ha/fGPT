# fGPT

> A language model trained from scratch

Test the model here: https://felixgpt.streamlit.app/
	
Repo is work in progress: At the moment the model was trained only on 2672 stories for 10 epochs. 

# Overview

The model architecture is a GPT-2 style encoder only tranformer (`dim_embedings=768` and `n_blocks=4`) with context of 1004 tokens and has 38M parameters.

The model was trained on the [tiny stories](https://arxiv.org/abs/2305.07759) dataset using spaCy englisch words tokenizer. 

It is inspired from Andrej Karpathy's  [nanoGPT](https://github.com/karpathy/nanoGPT) and his series [nn form zero to hero](https://github.com/karpathy/nn-zero-to-hero).

# Training



# Results

Here are the prompts that were evaluated in chapter 4.4 of the original paper. 

Prompt | fGPT | 33M 4 layers of original paper
-------- | -------- | --------
Alice was so tired when she got back home so she went | Answer from fGPT   | straight to bed.
Jack and Lily saw a rainbow after a rainy day. They were amazed by the colors. Jack said, "Look, Lily. A rainbow has   | Answer from fGPT   | red, orange, yellow, green, blue, and purple!
Jack and Lily liked to watch the moon at night. They noticed that the moon changed its shape every night. Sometimes the moon was big and round, and sometimes it was   | Answer from fGPT   | small and thin.
Jack wanted to read a book, so he went to   | Answer from fGPT   | the library.
"Can cows fly?", Alice asked her mother.   | Answer from fGPT   | "No, they can't fly," her mother said.
"What do birds like to eat?", Tom asked his mother.   | Answer from fGPT   | His mother smiled and said, "They like to eat worms and bugs."
"What language do they speak in France?", Tom asked his mother   | Answer from fGPT   | "They speak French," his mother replied.
If I throw a ball up in the air, eventually it will  | Answer from fGPT   | come down."
It was winter and cold outside so his mother told him, "You should   | Answer from fGPT   | wear your warm coat so you don't get cold.
Lily likes cats and dogs. She asked her mom for a dog and her mom said no, so instead she asked   | Answer from fGPT   | her dad for a cat.
Jack told Mary, "If you give me your banana, I’ll give you my apple". Mary gave Jack her Banana so   | Answer from fGPT   | he could give her the apple.
On weekends Jack went to visit his grandmother whereas on weekdays he would go to school. Last weekend, when Jack was on his way to   | Answer from fGPT   | his grandmother's house
Lily and Ben were having an argument. Ben said that cake is much better than ice cream and Lily said that   | Answer from fGPT   | ice cream is better than cake
Lily and Ben are having an argument. They are trying to decide between the park and the swimming pool. Ben says, "I want to go to the park". Lily says   | Answer from fGPT   | , "No, I want to go to the pool".
Jack's mother was not home, and his father was at home. When Jack came home, he said hello to   | Answer from fGPT   | his mother, but she didn't answer. Jack was confused.
Lily doesn't like swimming. When her father wants to take her to the swimming pool, she says   | Answer from fGPT   | , "No, I don’t want to go. I want to stay here."
Both Ben and Lily wanted cake. Father said that there was only one piece of cake left. They   | Answer from fGPT   | started to fight over the cake. They pulled and pushed and shouted.
Ben went to visit Lily in her house, but she was not at home. Ben knocked on the door,   | Answer from fGPT   | but no one answered. He knocked again and again, but still  no one came.
"Hi Jane, have you seen Alice? I cannot find her anywhere", said Jack.   | Answer from fGPT   | Jane smiled and said, "Don' t worry  Jack, I'll help you find her". Jack and Jane looked around the park, but they couldn't find Alice.
 Max had two dogs. One was white and the other was black. Max walked up the street and saw a kid with a dog. He told the kid, "I see you have a Brown dog. I also have | Answer from fGPT   | two dogs.
Anne had a piece of candy in her left pocket and a piece of chocolate in her right pocket. Annes mom asked her, "Anne, what is that you have in your left pocket?"   | Answer from fGPT   | Anne replied, "It's a piece of candy, Mommy. It's so yummy!"
Alice had both an apple and a carrot in her bag. She took the apple out of the bag and gave it to Jack. She reached into the bag again and took   | Answer from fGPT   | out the carrot.
Alice and Jack walked up the street and met a girl in a red dress. The girl said to them, "Hi, I am Jane. What are your names?"   | Answer from fGPT   | Alice said, "I'm Alice and this is Jack."
Diva was hungry, and wanted to bake a cake, but she did not have any sugar at home, so she decided to go ask around. She started walking and met a squirrel. She asked the squirrel, "Would you happen   | Answer from fGPT   | to have some sugar?"
